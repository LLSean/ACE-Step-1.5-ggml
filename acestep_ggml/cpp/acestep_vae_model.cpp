#include "acestep_vae_model.h"

#include "json_min.h"
#include "safetensors.h"

#include "ggml-cpu.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

namespace ace_vae {

namespace {

static bool starts_with(const std::string & s, const std::string & p) {
    return s.size() >= p.size() && s.compare(0, p.size(), p) == 0;
}

static bool ends_with(const std::string & s, const std::string & suf) {
    return s.size() >= suf.size() && s.compare(s.size() - suf.size(), suf.size(), suf) == 0;
}

static bool env_enabled(const char * key) {
    const char * value = std::getenv(key);
    return value && value[0] && std::strcmp(value, "0") != 0;
}

static int64_t numel(const std::vector<int64_t> & shape) {
    int64_t n = 1;
    for (int64_t d : shape) {
        n *= d;
    }
    return n;
}

static bool read_text_file(const std::string & path, std::string & out, std::string & error) {
    std::ifstream ifs(path);
    if (!ifs) {
        error = "failed to open file: " + path;
        return false;
    }
    std::ostringstream oss;
    oss << ifs.rdbuf();
    out = oss.str();
    return true;
}

static bool load_config(const std::string & path, Config & cfg, std::string & error) {
    std::string text;
    if (!read_text_file(path, text, error)) {
        return false;
    }

    ace_json::Value root;
    try {
        ace_json::Parser parser(text);
        root = parser.parse();
    } catch (const std::exception & e) {
        error = std::string("failed to parse VAE config: ") + e.what();
        return false;
    }

    if (!root.is_object()) {
        error = "VAE config is not a JSON object";
        return false;
    }
    const auto & obj = root.as_object();

    auto get_int = [&](const char * key, int32_t & out_val) -> bool {
        auto it = obj.find(key);
        if (it == obj.end() || !it->second.is_number()) {
            error = std::string("missing or invalid integer key: ") + key;
            return false;
        }
        out_val = static_cast<int32_t>(it->second.as_int());
        return true;
    };

    auto get_array_int = [&](const char * key, std::vector<int32_t> & out_vals) -> bool {
        auto it = obj.find(key);
        if (it == obj.end() || !it->second.is_array()) {
            error = std::string("missing or invalid array key: ") + key;
            return false;
        }
        out_vals.clear();
        for (const auto & v : it->second.as_array()) {
            if (!v.is_number()) {
                error = std::string("non-numeric value in array: ") + key;
                return false;
            }
            out_vals.push_back(static_cast<int32_t>(v.as_int()));
        }
        return true;
    };

    if (!get_int("audio_channels", cfg.audio_channels) ||
        !get_int("encoder_hidden_size", cfg.encoder_hidden_size) ||
        !get_int("decoder_channels", cfg.decoder_channels) ||
        !get_int("decoder_input_channels", cfg.decoder_input_channels) ||
        !get_int("sampling_rate", cfg.sampling_rate) ||
        !get_array_int("downsampling_ratios", cfg.downsampling_ratios) ||
        !get_array_int("channel_multiples", cfg.channel_multiples)) {
        return false;
    }

    if (cfg.downsampling_ratios.empty() || cfg.channel_multiples.empty()) {
        error = "invalid VAE config: empty ratios or channel multiples";
        return false;
    }

    cfg.upsampling_ratios = cfg.downsampling_ratios;
    std::reverse(cfg.upsampling_ratios.begin(), cfg.upsampling_ratios.end());
    cfg.hop_length = 1;
    for (int32_t r : cfg.downsampling_ratios) {
        cfg.hop_length *= r;
    }
    return true;
}

static bool read_tensor_as_f32(
    const ace_safetensors::File & st,
    const ace_safetensors::TensorInfo & info,
    std::vector<float> & out,
    std::string & error) {

    const int64_t n = numel(info.shape);
    if (n <= 0) {
        error = "invalid tensor shape for " + info.name;
        return false;
    }

    std::vector<uint8_t> raw(info.nbytes());
    if (!st.read_tensor(info, raw.data(), raw.size(), error)) {
        return false;
    }

    out.resize(static_cast<size_t>(n));
    if (info.dtype == "F32") {
        if (raw.size() != static_cast<size_t>(n) * sizeof(float)) {
            error = "F32 tensor byte-size mismatch for " + info.name;
            return false;
        }
        std::memcpy(out.data(), raw.data(), raw.size());
        return true;
    }
    if (info.dtype == "BF16") {
        const auto * src = reinterpret_cast<const ggml_bf16_t *>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = ggml_bf16_to_fp32(src[i]);
        }
        return true;
    }
    if (info.dtype == "F16") {
        const auto * src = reinterpret_cast<const ggml_fp16_t *>(raw.data());
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = ggml_fp16_to_fp32(src[i]);
        }
        return true;
    }

    error = "unsupported VAE dtype for " + info.name + ": " + info.dtype;
    return false;
}

static ggml_tensor * make_tensor_1d_f32(
    ggml_context * ctx,
    const std::string & name,
    int64_t ne0,
    const std::vector<float> & data,
    std::string & error) {

    if (static_cast<int64_t>(data.size()) != ne0) {
        error = "size mismatch creating tensor: " + name;
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, ne0);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    std::memcpy(t->data, data.data(), data.size() * sizeof(float));
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * make_tensor_3d_f32(
    ggml_context * ctx,
    const std::string & name,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    const std::vector<float> & data,
    std::string & error) {

    if (static_cast<int64_t>(data.size()) != ne0 * ne1 * ne2) {
        error = "size mismatch creating tensor: " + name;
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, ne0, ne1, ne2);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    std::memcpy(t->data, data.data(), data.size() * sizeof(float));
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * make_tensor_3d_f16(
    ggml_context * ctx,
    const std::string & name,
    int64_t ne0,
    int64_t ne1,
    int64_t ne2,
    const std::vector<float> & data,
    std::string & error) {

    if (static_cast<int64_t>(data.size()) != ne0 * ne1 * ne2) {
        error = "size mismatch creating tensor: " + name;
        return nullptr;
    }
    ggml_tensor * t = ggml_new_tensor_3d(ctx, GGML_TYPE_F16, ne0, ne1, ne2);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    auto * dst = static_cast<ggml_fp16_t *>(t->data);
    for (size_t i = 0; i < data.size(); ++i) {
        dst[i] = ggml_fp32_to_fp16(data[i]);
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static bool load_tensor_1d_f32(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    ggml_tensor * & out,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return false;
    }
    if (info->shape.size() != 1) {
        error = "invalid tensor shape for " + name;
        return false;
    }
    std::vector<float> data;
    if (!read_tensor_as_f32(st, *info, data, error)) {
        return false;
    }
    out = make_tensor_1d_f32(ctx, name, info->shape[0], data, error);
    return out != nullptr;
}

static bool load_tensor_3d_f32_reversed(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    ggml_tensor * & out,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return false;
    }
    if (info->shape.size() != 3) {
        error = "invalid tensor shape for " + name;
        return false;
    }
    std::vector<float> data;
    if (!read_tensor_as_f32(st, *info, data, error)) {
        return false;
    }
    // Safetensors/PyTorch layout [A, B, C] -> ggml tensor [C, B, A].
    out = make_tensor_3d_f32(ctx, name, info->shape[2], info->shape[1], info->shape[0], data, error);
    return out != nullptr;
}

static bool load_snake(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & prefix,
    Snake & out,
    std::string & error) {

    if (!load_tensor_3d_f32_reversed(ctx, st, prefix + ".alpha", out.alpha, error)) {
        return false;
    }
    if (!load_tensor_3d_f32_reversed(ctx, st, prefix + ".beta", out.beta, error)) {
        return false;
    }
    return true;
}

static bool load_conv_weight_norm(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & prefix,
    bool transposed,
    bool with_bias,
    int32_t stride,
    int32_t padding,
    int32_t dilation,
    Conv1d & out,
    std::string & error) {

    const auto * info_g = st.find(prefix + ".weight_g");
    const auto * info_v = st.find(prefix + ".weight_v");
    if (!info_g || !info_v) {
        error = "missing weight-norm tensors for " + prefix;
        return false;
    }
    if (info_v->shape.size() != 3 || info_g->shape.size() != 3) {
        error = "invalid weight-norm tensor shape for " + prefix;
        return false;
    }

    std::vector<float> g_data;
    std::vector<float> v_data;
    if (!read_tensor_as_f32(st, *info_g, g_data, error)) {
        return false;
    }
    if (!read_tensor_as_f32(st, *info_v, v_data, error)) {
        return false;
    }

    const int64_t d0 = info_v->shape[0];
    const int64_t d1 = info_v->shape[1];
    const int64_t d2 = info_v->shape[2];
    if (static_cast<int64_t>(g_data.size()) != d0) {
        error = "weight_g size mismatch for " + prefix;
        return false;
    }

    std::vector<float> w_data(v_data.size());
    const int64_t row = d1 * d2;
    for (int64_t i = 0; i < d0; ++i) {
        const int64_t base = i * row;
        double ss = 0.0;
        for (int64_t j = 0; j < row; ++j) {
            const float v = v_data[static_cast<size_t>(base + j)];
            ss += static_cast<double>(v) * static_cast<double>(v);
        }
        const float scale = g_data[static_cast<size_t>(i)] / std::sqrt(static_cast<float>(ss) + 1e-12f);
        for (int64_t j = 0; j < row; ++j) {
            w_data[static_cast<size_t>(base + j)] = v_data[static_cast<size_t>(base + j)] * scale;
        }
    }

    const bool force_f32_transposed = transposed && env_enabled("ACE_GGML_VAE_TRANSPOSE_CONV_F32");
    if (force_f32_transposed) {
        out.weight = make_tensor_3d_f32(ctx, prefix + ".weight", d2, d1, d0, w_data, error);
    } else {
        out.weight = make_tensor_3d_f16(ctx, prefix + ".weight", d2, d1, d0, w_data, error);
    }
    if (!out.weight) {
        return false;
    }

    if (with_bias) {
        if (!load_tensor_1d_f32(ctx, st, prefix + ".bias", out.bias, error)) {
            return false;
        }
    }

    out.transposed = transposed;
    out.stride = stride;
    out.padding = padding;
    out.dilation = dilation;
    return true;
}

static bool load_residual_unit(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & prefix,
    int32_t dilation,
    ResidualUnit & out,
    std::string & error) {

    if (!load_snake(ctx, st, prefix + ".snake1", out.snake1, error)) {
        return false;
    }
    const int32_t pad = ((7 - 1) * dilation) / 2;
    if (!load_conv_weight_norm(ctx, st, prefix + ".conv1", false, true, 1, pad, dilation, out.conv1, error)) {
        return false;
    }
    if (!load_snake(ctx, st, prefix + ".snake2", out.snake2, error)) {
        return false;
    }
    if (!load_conv_weight_norm(ctx, st, prefix + ".conv2", false, true, 1, 0, 1, out.conv2, error)) {
        return false;
    }
    return true;
}

static bool load_decoder_block(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & prefix,
    int32_t stride,
    DecoderBlock & out,
    std::string & error) {

    if (!load_snake(ctx, st, prefix + ".snake1", out.snake1, error)) {
        return false;
    }
    const int32_t pad = (stride + 1) / 2; // ceil(stride / 2)
    if (!load_conv_weight_norm(ctx, st, prefix + ".conv_t1", true, true, stride, pad, 1, out.conv_t1, error)) {
        return false;
    }
    if (!load_residual_unit(ctx, st, prefix + ".res_unit1", 1, out.res1, error)) {
        return false;
    }
    if (!load_residual_unit(ctx, st, prefix + ".res_unit2", 3, out.res2, error)) {
        return false;
    }
    if (!load_residual_unit(ctx, st, prefix + ".res_unit3", 9, out.res3, error)) {
        return false;
    }
    return true;
}

static bool load_encoder_block(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & prefix,
    int32_t stride,
    EncoderBlock & out,
    std::string & error) {

    if (!load_residual_unit(ctx, st, prefix + ".res_unit1", 1, out.res1, error)) {
        return false;
    }
    if (!load_residual_unit(ctx, st, prefix + ".res_unit2", 3, out.res2, error)) {
        return false;
    }
    if (!load_residual_unit(ctx, st, prefix + ".res_unit3", 9, out.res3, error)) {
        return false;
    }
    if (!load_snake(ctx, st, prefix + ".snake1", out.snake1, error)) {
        return false;
    }
    const int32_t pad = (stride + 1) / 2; // ceil(stride / 2)
    if (!load_conv_weight_norm(ctx, st, prefix + ".conv1", false, true, stride, pad, 1, out.conv1, error)) {
        return false;
    }
    return true;
}

static ggml_tensor * add_bias_3d(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    if (!bias) {
        return x;
    }
    ggml_tensor * b3 = ggml_reshape_3d(ctx, bias, 1, bias->ne[0], 1);
    ggml_tensor * b = ggml_repeat(ctx, b3, x);
    return ggml_add(ctx, x, b);
}

static ggml_tensor * center_crop_3d(ggml_context * ctx, ggml_tensor * x, int64_t target_len);

static ggml_tensor * snake_forward(ggml_context * ctx, const Snake & snake, ggml_tensor * x) {
    ggml_tensor * alpha = ggml_exp(ctx, snake.alpha);
    ggml_tensor * beta = ggml_exp(ctx, snake.beta);
    ggml_tensor * alpha_rep = ggml_repeat(ctx, alpha, x);
    ggml_tensor * beta_rep = ggml_repeat(ctx, beta, x);
    ggml_tensor * s = ggml_mul(ctx, alpha_rep, x);
    s = ggml_sin(ctx, s);
    s = ggml_sqr(ctx, s);
    s = ggml_div(ctx, s, beta_rep);
    return ggml_add(ctx, x, s);
}

static ggml_tensor * conv_forward(ggml_context * ctx, const Conv1d & conv, ggml_tensor * x) {
    ggml_tensor * y = nullptr;
    if (conv.transposed) {
        // ggml conv_transpose_1d currently supports p0 == 0. Emulate PyTorch padding by post-cropping.
        y = ggml_conv_transpose_1d(ctx, conv.weight, x, conv.stride, 0, conv.dilation);
        if (conv.padding > 0) {
            const int64_t in_len = x->ne[0];
            const int64_t kernel = conv.weight->ne[0];
            const int64_t target_len =
                (in_len - 1) * conv.stride - 2 * conv.padding + conv.dilation * (kernel - 1) + 1;
            if (target_len > 0 && target_len < y->ne[0]) {
                y = center_crop_3d(ctx, y, target_len);
            }
        }
    } else {
        y = ggml_conv_1d(ctx, conv.weight, x, conv.stride, conv.padding, conv.dilation);
    }
    return add_bias_3d(ctx, y, conv.bias);
}

static ggml_tensor * center_crop_3d(ggml_context * ctx, ggml_tensor * x, int64_t target_len) {
    if (x->ne[0] <= target_len) {
        return x;
    }
    const int64_t diff = x->ne[0] - target_len;
    const int64_t start = diff / 2;
    const size_t offset = static_cast<size_t>(start) * static_cast<size_t>(x->nb[0]);
    return ggml_view_3d(ctx, x, target_len, x->ne[1], x->ne[2], x->nb[1], x->nb[2], offset);
}

static ggml_tensor * residual_forward(ggml_context * ctx, const ResidualUnit & ru, ggml_tensor * x) {
    ggml_tensor * residual = x;
    ggml_tensor * y = conv_forward(ctx, ru.conv1, snake_forward(ctx, ru.snake1, x));
    y = conv_forward(ctx, ru.conv2, snake_forward(ctx, ru.snake2, y));

    const int64_t len = std::min<int64_t>(residual->ne[0], y->ne[0]);
    residual = center_crop_3d(ctx, residual, len);
    y = center_crop_3d(ctx, y, len);
    return ggml_add(ctx, residual, y);
}

static ggml_tensor * decoder_block_forward(ggml_context * ctx, const DecoderBlock & block, ggml_tensor * x) {
    x = snake_forward(ctx, block.snake1, x);
    x = conv_forward(ctx, block.conv_t1, x);
    x = residual_forward(ctx, block.res1, x);
    x = residual_forward(ctx, block.res2, x);
    x = residual_forward(ctx, block.res3, x);
    return x;
}

} // namespace

void free_model(Model & model) {
    if (model.ctx) {
        ggml_free(model.ctx);
        model.ctx = nullptr;
    }
    if (model.ctx_buffer) {
        std::free(model.ctx_buffer);
        model.ctx_buffer = nullptr;
    }
    model.ctx_buffer_size = 0;
    model.enc_blocks.clear();
    model.blocks.clear();
    model.loaded = false;
}

bool load_model_from_dir(const std::string & dir, Model & out, std::string & error) {
    const std::string cfg_path = dir + "/config.json";
    const std::string st_path = dir + "/diffusion_pytorch_model.safetensors";

    Config cfg;
    if (!load_config(cfg_path, cfg, error)) {
        return false;
    }

    ace_safetensors::File st;
    if (!st.load(st_path, error)) {
        return false;
    }

    // Keep encoder/decoder tensors.
    size_t n_tensors = 0;
    size_t stored_bytes = 0;
    for (const auto & info : st.tensors) {
        if (!starts_with(info.name, "decoder.") && !starts_with(info.name, "encoder.")) {
            continue;
        }
        if (ends_with(info.name, ".weight_g")) {
            continue;
        }
        const int64_t n = numel(info.shape);
        stored_bytes += static_cast<size_t>(n) * sizeof(float);
        ++n_tensors;
    }

    const size_t overhead = ggml_tensor_overhead() * (n_tensors + 256);
    const size_t ctx_size = stored_bytes + overhead + (64ULL * 1024ULL * 1024ULL);

    void * buffer = std::malloc(ctx_size);
    if (!buffer) {
        error = "failed to allocate VAE model buffer";
        return false;
    }

    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = buffer;
    params.no_alloc = false;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::free(buffer);
        error = "ggml_init failed for VAE model";
        return false;
    }

    Model model;
    model.cfg = cfg;
    model.ctx = ctx;
    model.ctx_buffer = buffer;
    model.ctx_buffer_size = ctx_size;

    if (!load_conv_weight_norm(ctx, st, "encoder.conv1", false, true, 1, 3, 1, model.enc_conv1, error)) {
        free_model(model);
        return false;
    }
    const size_t n_enc_blocks = cfg.downsampling_ratios.size();
    model.enc_blocks.resize(n_enc_blocks);
    for (size_t i = 0; i < n_enc_blocks; ++i) {
        std::string prefix = "encoder.block." + std::to_string(i);
        if (!load_encoder_block(ctx, st, prefix, cfg.downsampling_ratios[i], model.enc_blocks[i], error)) {
            free_model(model);
            return false;
        }
    }
    if (!load_snake(ctx, st, "encoder.snake1", model.enc_snake1, error)) {
        free_model(model);
        return false;
    }
    if (!load_conv_weight_norm(ctx, st, "encoder.conv2", false, true, 1, 1, 1, model.enc_conv2, error)) {
        free_model(model);
        return false;
    }

    if (!load_conv_weight_norm(ctx, st, "decoder.conv1", false, true, 1, 3, 1, model.conv1, error)) {
        free_model(model);
        return false;
    }

    const size_t n_blocks = cfg.upsampling_ratios.size();
    model.blocks.resize(n_blocks);
    for (size_t i = 0; i < n_blocks; ++i) {
        std::string prefix = "decoder.block." + std::to_string(i);
        if (!load_decoder_block(ctx, st, prefix, cfg.upsampling_ratios[i], model.blocks[i], error)) {
            free_model(model);
            return false;
        }
    }

    if (!load_snake(ctx, st, "decoder.snake1", model.snake1, error)) {
        free_model(model);
        return false;
    }
    if (!load_conv_weight_norm(ctx, st, "decoder.conv2", false, false, 1, 3, 1, model.conv2, error)) {
        free_model(model);
        return false;
    }

    model.loaded = true;
    out = std::move(model);
    return true;
}

ggml_tensor * forward_decode(
    ggml_context * ctx,
    const Model & model,
    const float * latents,
    int32_t n_frames,
    ggml_tensor ** latent_input_out) {

    if (n_frames <= 0) {
        return nullptr;
    }

    const int32_t latent_channels = model.cfg.decoder_input_channels;
    const bool no_alloc = ggml_get_no_alloc(ctx);
    if (!latents && !no_alloc) {
        return nullptr;
    }

    // ggml conv1d input layout: [time, channels, batch]
    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_frames, latent_channels, 1);
    if (!x) {
        return nullptr;
    }
    if (latent_input_out) {
        *latent_input_out = x;
    }
    ggml_set_input(x);
    if (x->data) {
        if (!latents) {
            return nullptr;
        }
        float * x_data = static_cast<float *>(x->data);
        for (int32_t t = 0; t < n_frames; ++t) {
            for (int32_t c = 0; c < latent_channels; ++c) {
                x_data[t + c * n_frames] = latents[static_cast<size_t>(t) * latent_channels + c];
            }
        }
    }

    x = conv_forward(ctx, model.conv1, x);
    for (const auto & block : model.blocks) {
        x = decoder_block_forward(ctx, block, x);
    }
    x = snake_forward(ctx, model.snake1, x);
    x = conv_forward(ctx, model.conv2, x);
    return x;
}

ggml_tensor * forward_encode(
    ggml_context * ctx,
    const Model & model,
    const float * audio,
    int32_t n_samples) {

    if (!audio || n_samples <= 0) {
        return nullptr;
    }

    const int32_t audio_channels = model.cfg.audio_channels;
    const int32_t latent_channels = model.cfg.decoder_input_channels;

    ggml_tensor * x = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, n_samples, audio_channels, 1);
    if (!x || !x->data) {
        return nullptr;
    }
    float * x_data = static_cast<float *>(x->data);
    for (int32_t t = 0; t < n_samples; ++t) {
        for (int32_t c = 0; c < audio_channels; ++c) {
            x_data[t + c * n_samples] = audio[static_cast<size_t>(t) * audio_channels + c];
        }
    }

    x = conv_forward(ctx, model.enc_conv1, x);
    for (const auto & block : model.enc_blocks) {
        x = residual_forward(ctx, block.res1, x);
        x = residual_forward(ctx, block.res2, x);
        x = residual_forward(ctx, block.res3, x);
        x = snake_forward(ctx, block.snake1, x);
        x = conv_forward(ctx, block.conv1, x);
    }
    x = snake_forward(ctx, model.enc_snake1, x);
    x = conv_forward(ctx, model.enc_conv2, x);

    // Posterior mode: keep the mean part from [mean, scale] channel concat.
    if (x->ne[1] < latent_channels) {
        return nullptr;
    }
    return ggml_view_3d(ctx, x, x->ne[0], latent_channels, x->ne[2], x->nb[1], x->nb[2], 0);
}

} // namespace ace_vae
