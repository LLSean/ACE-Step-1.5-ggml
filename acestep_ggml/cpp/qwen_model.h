#ifndef ACESTEP_QWEN_MODEL_H
#define ACESTEP_QWEN_MODEL_H

#include "qwen_config.h"
#include "safetensors.h"

#include <string>
#include <vector>

struct ggml_context;
struct ggml_tensor;

namespace ace_qwen {

struct Layer {
    ggml_tensor * input_norm = nullptr;
    ggml_tensor * post_attn_norm = nullptr;
    ggml_tensor * w_q = nullptr;
    ggml_tensor * w_k = nullptr;
    ggml_tensor * w_v = nullptr;
    ggml_tensor * w_o = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_down = nullptr;
};

struct Model {
    bool loaded = false;
    Config cfg;
    ggml_context * ctx = nullptr;
    void * ctx_buffer = nullptr;
    size_t ctx_buffer_size = 0;

    ggml_tensor * tok_embeddings = nullptr;
    ggml_tensor * norm = nullptr;
    std::vector<Layer> layers;
};

bool load_model_from_dir(const std::string & dir, Model & out, std::string & error);
void free_model(Model & model);

ggml_tensor * forward_text_encoder(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    bool causal);

ggml_tensor * forward_text_encoder_embeddings(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    int32_t n_tokens);

ggml_tensor * forward_text_encoder_layers(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    bool causal,
    int32_t n_layers,
    bool apply_final_norm);


}  // namespace ace_qwen

#endif  // ACESTEP_QWEN_MODEL_H
