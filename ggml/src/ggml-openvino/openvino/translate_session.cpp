#include "translate_session.hpp"

#include <cstdint>
#include <cstdlib>
#include <map>
#include <memory>
#include <openvino/core/node.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/broadcast.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/convert.hpp>
#include <openvino/op/cos.hpp>
#include <openvino/op/divide.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/range.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/result.hpp>
#include <openvino/op/sin.hpp>
#include <openvino/op/squeeze.hpp>
#include <openvino/op/transpose.hpp>
#include <openvino/op/unsqueeze.hpp>
#include <openvino/pass/constant_folding.hpp>
#include <openvino/pass/make_stateful.hpp>

#include "ggml-openvino/openvino/node_context.hpp"
#include "ggml-openvino/openvino/utils.hpp"
#include "input_model.hpp"
#include "pass/fuse_to_sdpa.hpp"

namespace ov {
namespace frontend {
namespace ggml {

using namespace ov::op;

namespace {
ov::pass::MakeStateful::ParamResPairs get_kv_param_res_pairs(
    const std::shared_ptr<ov::Model>& model, const std::map<std::string, std::string>& kv_param_res_names) {
    ov::pass::MakeStateful::ParamResPairs pairs;
    const auto& params = model->get_parameters();
    const auto& results = model->get_results();

    for (const auto& param_res : kv_param_res_names) {
        const auto& param_name = param_res.first;
        const auto& res_name = param_res.second;

        auto param_it = std::find_if(params.begin(), params.end(), [&](const std::shared_ptr<v0::Parameter>& node) {
            return node->get_friendly_name() == param_name;
        });

        OPENVINO_ASSERT(param_it != params.end(), "The tensor name ", param_name,
                        " is not associated with any of "
                        "Parameters in the network.");

        auto res_it = std::find_if(results.begin(), results.end(), [&](const std::shared_ptr<v0::Result>& node) {
            return node->get_friendly_name() == res_name;
        });

        OPENVINO_ASSERT(res_it != results.end(), "The tensor name ", res_name,
                        " is not associated with any of "
                        "Results in the network.");

        std::shared_ptr<ov::op::v0::Parameter> param = *param_it;
        std::shared_ptr<ov::op::v0::Result> res = *res_it;
        pairs.emplace_back(param, res);
    }
    return pairs;
}

void add_token_len(TensorMap& tensor_map) {
    auto inp_tokens = tensor_map.at("inp_tokens").get_node_shared_ptr();
    auto token_len = get_dimensions(inp_tokens, {2});
    token_len->set_friendly_name("token_len");
    tensor_map.insert({"token_len", token_len->output(0)});
}

void add_kv_update_indices(TensorMap& tensor_map, GgmlDecoder& ggml_model_decoder) {
    // cache_k layout: [S, N, H] (seq, num_heads, head_size)
    // cache_v layout: [N, H, S] (num_heads, head_size, seq)
    // When writing to cache_v, cache should be reshaped to [N*H, S] and v-curr should be flattened
    auto past_token_len = tensor_map.at("past_token_len").get_node_shared_ptr();
    auto token_len = tensor_map.at("token_len").get_node_shared_ptr();

    Output<Node> update_indices_k;
    Output<Node> update_indices_v;

    auto zero = ov::op::v0::Constant::create(ov::element::i64, {1}, {0});
    auto zero_scalar = ov::op::v0::Constant::create(ov::element::i64, {}, {0});
    auto one = ov::op::v0::Constant::create(ov::element::i64, {1}, {1});
    auto one_scalar = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto two = ov::op::v0::Constant::create(ov::element::i64, {1}, {2});

    auto past_token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(past_token_len, zero);
    auto token_len_scalar = std::make_shared<ov::op::v0::Squeeze>(token_len, zero);
    auto total_token_len_scalar = std::make_shared<ov::op::v1::Add>(past_token_len_scalar, token_len_scalar);

    Output<Node> update_indices = std::make_shared<ov::op::v4::Range>(
        past_token_len_scalar, total_token_len_scalar, one_scalar, ov::element::i64);
    if (ggml_model_decoder.is_static()) {
        update_indices = past_token_len;
    }

    update_indices_k = std::make_shared<ov::op::v0::Unsqueeze>(update_indices, one);
    update_indices_k.get_node_shared_ptr()->set_friendly_name("update_indices_k");
    tensor_map.insert({"update_indices_k", update_indices_k});

    auto total_head_size = ggml_model_decoder.get_num_heads_kv() * ggml_model_decoder.get_head_size();
    auto total_head_size_node = ov::op::v0::Constant::create(ov::element::i64, {1}, {total_head_size});
    auto total_head_size_scalar = std::make_shared<ov::op::v0::Squeeze>(total_head_size_node, zero);

    // 1D tensor of shape [total_head_size], values starting from 0
    auto range_row =
        std::make_shared<ov::op::v4::Range>(zero_scalar, total_head_size_scalar, one_scalar, ov::element::i64);
    auto range_row_reshaped =
        std::make_shared<ov::op::v0::Unsqueeze>(range_row, ov::op::v0::Constant::create(ov::element::i64, {2}, {1, 2}));
    auto row_indices = std::make_shared<ov::op::v3::Broadcast>(
        range_row_reshaped,
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{total_head_size_node, token_len, one}, 0));

    // 1D tensor of shape [token_len], values starting from past_token_len
    auto range_col = update_indices;
    auto range_col_reshaped =
        std::make_shared<ov::op::v0::Unsqueeze>(range_col, ov::op::v0::Constant::create(ov::element::i64, {2}, {0, 2}));
    auto col_indices = std::make_shared<ov::op::v3::Broadcast>(
        range_col_reshaped,
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{total_head_size_node, token_len, one}, 0));

    // Stack row_indices and col_indices along last axis: [total_head_size, token_len, 2]
    update_indices_v = std::make_shared<ov::op::v0::Concat>(OutputVector{row_indices, col_indices}, 2);
    update_indices_v = std::make_shared<ov::op::v1::Reshape>(
        update_indices_v, ov::op::v0::Constant::create(ov::element::i64, {2}, std::vector<int64_t>{-1, 2}), false);
    update_indices_v.get_node_shared_ptr()->set_friendly_name("update_indices_v");
    tensor_map.insert({"update_indices_v", update_indices_v});
}

void add_rope_sin_cos(TensorMap& tensor_map, GgmlDecoder& ggml_model_decoder) {
    int32_t* rope_params = ggml_model_decoder.get_rope_params();
    auto inp_pos = tensor_map.at("inp_pos").get_node_shared_ptr();
    std::shared_ptr<ov::Node> rope_freqs_weight;
    if (tensor_map.find("rope_freqs_weight") != tensor_map.end()) {
        rope_freqs_weight = tensor_map.at("rope_freqs.weight").get_node_shared_ptr();
    }

    auto sin_cos = make_sin_cos(rope_params, inp_pos, rope_freqs_weight);
    auto sin_theta = sin_cos.first;
    auto cos_theta = sin_cos.second;

    cos_theta.get_node_shared_ptr()->set_friendly_name("rope_cos");
    sin_theta.get_node_shared_ptr()->set_friendly_name("rope_sin");
    tensor_map.insert({"rope_cos", cos_theta});
    tensor_map.insert({"rope_sin", sin_theta});
}

// Create common patterns
void preprocess(TensorMap& tensor_map, GgmlDecoder& ggml_model_decoder) {
    add_token_len(tensor_map);
    add_kv_update_indices(tensor_map, ggml_model_decoder);
    add_rope_sin_cos(tensor_map, ggml_model_decoder);
}

}  // namespace

TranslateSession::TranslateSession(const frontend::InputModel::Ptr& input_model,
                                   const std::unordered_map<std::string, CreatorFunction>& translator_map,
                                   bool naive) :
    m_input_model(input_model),
    m_translator_map(translator_map),
    m_ov_model(nullptr),
    m_naive(naive) {}

std::shared_ptr<Model> TranslateSession::get_converted_model() {
    if (m_ov_model) {
        return m_ov_model;
    }
    m_ov_model = translate_graph(m_input_model);
    return m_ov_model;
}

std::shared_ptr<Model> TranslateSession::translate_graph(const frontend::InputModel::Ptr& input_model) {
    ov::ParameterVector params;
    ov::ResultVector results;
    auto tensor_map = std::make_shared<TensorMap>();
    std::shared_ptr<Model> resulting_model;

    const auto& ggml_model = std::dynamic_pointer_cast<InputModel>(input_model);
    std::shared_ptr<GgmlDecoder> ggml_model_decoder = ggml_model->get_model_decoder();

    for (const auto& it : ggml_model_decoder->get_model_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : ggml_model_decoder->get_model_extra_inputs()) {
        params.push_back(std::dynamic_pointer_cast<ov::op::v0::Parameter>(it.second));
        (*tensor_map)[it.first] = it.second;
    }

    for (const auto& it : ggml_model_decoder->get_model_weights()) {
        (*tensor_map)[it.first] = it.second;
    }

    auto node_visitor = [&](std::shared_ptr<GgmlDecoder> node) {
        auto operation_type = node->get_op_type();
        if (operation_type == "GGML_OP_NONE") {
            return;
        }

        ov::OutputVector converted_outputs;
        auto it = m_translator_map.find(operation_type);
        FRONT_END_OP_CONVERSION_CHECK(it != m_translator_map.end(),
                                      "Translation for operation type ",
                                      operation_type,
                                      " is not implemented.");
        NodeContext node_context(node, tensor_map, this);
        converted_outputs = it->second(node_context);

        const auto& node_output_names = node->get_output_names();
        FRONT_END_OP_CONVERSION_CHECK(node_output_names.size() == converted_outputs.size(),
                                      "Number of ",
                                      operation_type,
                                      " outputs greater than number of converted outputs, which are ",
                                      node_output_names.size(),
                                      " and ",
                                      converted_outputs.size(),
                                      " respectively.");

        for (size_t i = 0; i < node_output_names.size(); ++i) {
            auto output_name = node_output_names[i];
            if (i < converted_outputs.size() && converted_outputs[i].get_node_shared_ptr() != nullptr) {
                (*tensor_map)[output_name] = converted_outputs[i];
            }
        }
    };

    if (!m_naive) {
        preprocess(*tensor_map, *ggml_model_decoder);
    }
    ggml_model_decoder->visit_subgraph(node_visitor);

    for (const auto& name : ggml_model_decoder->get_model_output_names()) {
        FRONT_END_GENERAL_CHECK(tensor_map->find(name) != tensor_map->end(),
                                "Output name not found in tensor map: ",
                                name);
        auto result = std::make_shared<v0::Result>(tensor_map->at(name));
        result->set_friendly_name(name);
        results.push_back(result);
    }

    resulting_model = std::make_shared<Model>(results, params);

    apply_transformations(resulting_model);
    return resulting_model;
}

void TranslateSession::apply_transformations(const std::shared_ptr<Model>& model) {
    auto ggml_model_decoder = std::dynamic_pointer_cast<InputModel>(m_input_model)->get_model_decoder();

    ov::pass::Manager manager;
    manager.set_per_pass_validation(true);
    manager.register_pass<ov::pass::ConstantFolding>();

    if (!ggml_model_decoder->is_static()) {
        const auto kv_param_res_names = ggml_model_decoder->get_kv_param_res_names();
        const auto kv_param_res_pairs = get_kv_param_res_pairs(model, kv_param_res_names);
        manager.register_pass<ov::pass::MakeStateful>(kv_param_res_pairs);
    }

    manager.register_pass<pass::FuseToSDPA>();
    manager.run_passes(model);
}

}  // namespace ggml
}  // namespace frontend
}  // namespace ov
