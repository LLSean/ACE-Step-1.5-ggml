#ifndef ACESTEP_DIT_CONFIG_H
#define ACESTEP_DIT_CONFIG_H

#include <cstdint>
#include <string>
#include <vector>

namespace ace_dit {

struct Config {
    int32_t hidden_size = 0;
    int32_t intermediate_size = 0;
    int32_t num_hidden_layers = 0;
    int32_t num_attention_heads = 0;
    int32_t num_key_value_heads = 0;
    int32_t head_dim = 0;
    int32_t max_position_embeddings = 0;
    float rms_norm_eps = 1e-6f;
    float rope_theta = 1000000.0f;
    int32_t patch_size = 0;
    int32_t in_channels = 0;
    int32_t audio_acoustic_hidden_dim = 0;
    int32_t text_hidden_dim = 0;
    int32_t num_lyric_encoder_hidden_layers = 0;
    int32_t timbre_hidden_dim = 0;
    int32_t num_timbre_encoder_hidden_layers = 0;
    int32_t timbre_fix_frame = 0;
    bool use_sliding_window = false;
    int32_t sliding_window = 0;
    std::vector<std::string> layer_types;
    std::string dtype;
};

bool load_config(const std::string & path, Config & cfg, std::string & error);

}  // namespace ace_dit

#endif  // ACESTEP_DIT_CONFIG_H
