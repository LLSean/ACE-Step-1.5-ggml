#ifndef ACESTEP_QWEN_CONFIG_H
#define ACESTEP_QWEN_CONFIG_H

#include <cstdint>
#include <string>

namespace ace_qwen {

struct Config {
    int32_t vocab_size = 0;
    int32_t hidden_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t intermediate_size = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 0.0f;
    float rope_theta = 1000000.0f;
    std::string dtype;
};

bool load_config(const std::string & path, Config & cfg, std::string & error);

}  // namespace ace_qwen

#endif  // ACESTEP_QWEN_CONFIG_H
