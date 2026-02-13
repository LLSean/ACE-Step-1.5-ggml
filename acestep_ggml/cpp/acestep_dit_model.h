#ifndef ACESTEP_DIT_MODEL_H
#define ACESTEP_DIT_MODEL_H

#include "acestep_dit_config.h"

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ace_dit {

struct TimestepWeights {
    ggml_tensor * w1 = nullptr;
    ggml_tensor * b1 = nullptr;
    ggml_tensor * w2 = nullptr;
    ggml_tensor * b2 = nullptr;
    ggml_tensor * w_proj = nullptr;
    ggml_tensor * b_proj = nullptr;
};

struct AttnWeights {
    ggml_tensor * w_q = nullptr;
    ggml_tensor * w_k = nullptr;
    ggml_tensor * w_v = nullptr;
    ggml_tensor * w_o = nullptr;
    ggml_tensor * q_norm = nullptr;
    ggml_tensor * k_norm = nullptr;
};

struct MLPWeights {
    ggml_tensor * w_gate = nullptr;
    ggml_tensor * w_up = nullptr;
    ggml_tensor * w_down = nullptr;
};

struct Layer {
    ggml_tensor * self_attn_norm = nullptr;
    AttnWeights self_attn;
    ggml_tensor * cross_attn_norm = nullptr;
    AttnWeights cross_attn;
    ggml_tensor * mlp_norm = nullptr;
    MLPWeights mlp;
    ggml_tensor * scale_shift_table = nullptr; // [hidden, 6]
    bool use_cross_attention = true;
    bool use_sliding_window = false;
};

struct EncoderLayer {
    ggml_tensor * input_norm = nullptr;
    AttnWeights self_attn;
    ggml_tensor * post_attn_norm = nullptr;
    MLPWeights mlp;
    bool use_sliding_window = false;
};

struct Model {
    Config cfg{};
    ggml_context * ctx = nullptr;
    void * ctx_buffer = nullptr;
    size_t ctx_buffer_size = 0;
    bool loaded = false;

    ggml_tensor * proj_in_w = nullptr;   // [in_channels*patch, hidden]
    ggml_tensor * proj_in_b = nullptr;   // [hidden]
    ggml_tensor * proj_out_w = nullptr;  // [hidden, out_channels*patch]
    ggml_tensor * proj_out_b = nullptr;  // [out_channels]

    ggml_tensor * condition_w = nullptr;
    ggml_tensor * condition_b = nullptr;
    ggml_tensor * text_projector_w = nullptr;     // [text_hidden, hidden]
    ggml_tensor * lyric_embed_w = nullptr;        // [text_hidden, hidden]
    ggml_tensor * lyric_embed_b = nullptr;        // [hidden]
    ggml_tensor * lyric_norm = nullptr;
    std::vector<EncoderLayer> lyric_layers;
    ggml_tensor * timbre_embed_w = nullptr;       // [timbre_hidden, hidden]
    ggml_tensor * timbre_embed_b = nullptr;       // [hidden]
    ggml_tensor * timbre_norm = nullptr;
    ggml_tensor * timbre_special_token = nullptr; // [hidden]
    std::vector<EncoderLayer> timbre_layers;

    ggml_tensor * norm_out = nullptr;
    ggml_tensor * out_scale_shift_table = nullptr; // [hidden, 2]

    TimestepWeights time_embed;
    TimestepWeights time_embed_r;

    std::vector<Layer> layers;
};

void free_model(Model & model);
bool load_model_from_dir(const std::string & dir, Model & out, std::string & error);

ggml_tensor * forward_dit(
    ggml_context * ctx,
    const Model & model,
    const float * hidden_states,            // [seq_len, audio_acoustic_hidden_dim]
    const float * context_latents,          // [seq_len, in_channels - audio_acoustic_hidden_dim]
    const float * encoder_hidden_states,    // [enc_len, hidden_size] or nullptr
    const int32_t * attention_mask,         // [seq_len] or nullptr
    const int32_t * encoder_attention_mask, // [enc_len] or nullptr
    int32_t seq_len,
    int32_t enc_len,
    float timestep,
    float timestep_r,
    ggml_tensor ** input_x0_out = nullptr,     // optional: receives packed input tensor [in_channels, seq_len_padded]
    ggml_tensor ** input_enc_out = nullptr);   // optional: receives encoder input tensor [hidden_size, enc_len]

ggml_tensor * forward_lyric_encoder(
    ggml_context * ctx,
    const Model & model,
    const float * lyric_hidden_states,    // [n_lyric, text_hidden_dim]
    const int32_t * lyric_attention_mask, // [n_lyric] or nullptr
    int32_t n_lyric);

ggml_tensor * forward_timbre_encoder(
    ggml_context * ctx,
    const Model & model,
    const float * refer_audio_hidden_states,    // [refer_len, timbre_hidden_dim]
    const int32_t * refer_audio_attention_mask, // [refer_len] or nullptr
    int32_t refer_len);

}  // namespace ace_dit

#endif  // ACESTEP_DIT_MODEL_H
