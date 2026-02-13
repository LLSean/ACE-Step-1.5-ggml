#ifndef ACESTEP_VAE_MODEL_H
#define ACESTEP_VAE_MODEL_H

#include "ggml.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace ace_vae {

struct Config {
    int32_t audio_channels = 2;
    int32_t encoder_hidden_size = 128;
    int32_t decoder_channels = 128;
    int32_t decoder_input_channels = 64;
    int32_t sampling_rate = 48000;
    std::vector<int32_t> downsampling_ratios;
    std::vector<int32_t> upsampling_ratios;
    std::vector<int32_t> channel_multiples;
    int32_t hop_length = 1;
};

struct Snake {
    ggml_tensor * alpha = nullptr; // [1, C, 1]
    ggml_tensor * beta = nullptr;  // [1, C, 1]
};

struct Conv1d {
    ggml_tensor * weight = nullptr; // conv: [K, in, out], conv_t: [K, out, in]
    ggml_tensor * bias = nullptr;   // [out] or nullptr
    bool transposed = false;
    int32_t stride = 1;
    int32_t padding = 0;
    int32_t dilation = 1;
};

struct ResidualUnit {
    Snake snake1;
    Conv1d conv1; // kernel=7, dilation in {1,3,9}
    Snake snake2;
    Conv1d conv2; // kernel=1
};

struct DecoderBlock {
    Snake snake1;
    Conv1d conv_t1; // transposed conv
    ResidualUnit res1;
    ResidualUnit res2;
    ResidualUnit res3;
};

struct EncoderBlock {
    ResidualUnit res1;
    ResidualUnit res2;
    ResidualUnit res3;
    Snake snake1;
    Conv1d conv1; // downsample conv
};

struct Model {
    Config cfg{};
    ggml_context * ctx = nullptr;
    void * ctx_buffer = nullptr;
    size_t ctx_buffer_size = 0;
    bool loaded = false;

    Conv1d enc_conv1; // encoder.conv1
    std::vector<EncoderBlock> enc_blocks;
    Snake enc_snake1; // encoder.snake1
    Conv1d enc_conv2; // encoder.conv2

    Conv1d conv1; // decoder.conv1
    std::vector<DecoderBlock> blocks;
    Snake snake1; // decoder.snake1
    Conv1d conv2; // decoder.conv2
};

void free_model(Model & model);
bool load_model_from_dir(const std::string & dir, Model & out, std::string & error);

// latents: [n_frames, decoder_input_channels]
// output : [n_samples, audio_channels] where n_samples = n_frames * hop_length
ggml_tensor * forward_decode(
    ggml_context * ctx,
    const Model & model,
    const float * latents,
    int32_t n_frames,
    ggml_tensor ** latent_input_out = nullptr);

// audio: [n_samples, audio_channels]
// output latent mode: [n_frames, decoder_input_channels]
ggml_tensor * forward_encode(
    ggml_context * ctx,
    const Model & model,
    const float * audio,
    int32_t n_samples);

} // namespace ace_vae

#endif // ACESTEP_VAE_MODEL_H
