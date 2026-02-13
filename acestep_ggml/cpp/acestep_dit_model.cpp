#include "acestep_dit_model.h"

#include "safetensors.h"

#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <vector>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace ace_dit {

static ggml_type ggml_type_from_dtype(const std::string & dtype, std::string & error) {
    if (dtype == "BF16") return GGML_TYPE_BF16;
    if (dtype == "F16") return GGML_TYPE_F16;
    if (dtype == "F32") return GGML_TYPE_F32;
    error = "unsupported dtype: " + dtype;
    return GGML_TYPE_COUNT;
}

static ggml_type parse_quant_type(const char * value) {
    if (!value || !value[0]) {
        return GGML_TYPE_COUNT;
    }
    std::string s(value);
    std::transform(s.begin(), s.end(), s.begin(), [](unsigned char c) { return static_cast<char>(std::toupper(c)); });
    if (s == "Q8" || s == "Q8_0") return GGML_TYPE_Q8_0;
    if (s == "Q6" || s == "Q6_K") return GGML_TYPE_Q6_K;
    if (s == "Q4" || s == "Q4_K" || s == "Q4_K_M") return GGML_TYPE_Q4_K;
    return GGML_TYPE_COUNT;
}

static ggml_type get_quant_type_from_env() {
    ggml_type t = parse_quant_type(std::getenv("ACE_GGML_DIT_WEIGHT_QTYPE"));
    if (t != GGML_TYPE_COUNT) {
        return t;
    }
    return parse_quant_type(std::getenv("ACE_GGML_WEIGHT_QTYPE"));
}

static std::string resolve_gguf_path(const std::string & dir) {
    const char * keys[] = {
        "ACE_GGML_DIT_GGUF",
        "ACE_GGML_DIT_GGUF_PATH",
    };
    for (const char * key : keys) {
        const char * value = std::getenv(key);
        if (value && value[0] && std::filesystem::exists(value)) {
            return std::string(value);
        }
    }

    const std::filesystem::path p(dir);
    if (p.extension() == ".gguf" && std::filesystem::exists(p)) {
        return p.string();
    }
    if (std::filesystem::is_directory(p)) {
        const std::filesystem::path candidate = p / "model.gguf";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }
    return "";
}

static bool load_gguf_file(
    const std::string & path,
    gguf_context * & out_gguf,
    ggml_context * & out_gguf_ctx,
    std::string & error) {

    out_gguf = nullptr;
    out_gguf_ctx = nullptr;

    gguf_init_params params{};
    params.no_alloc = false;
    params.ctx = &out_gguf_ctx;

    out_gguf = gguf_init_from_file(path.c_str(), params);
    if (!out_gguf || !out_gguf_ctx) {
        if (out_gguf) {
            gguf_free(out_gguf);
            out_gguf = nullptr;
        }
        if (out_gguf_ctx) {
            ggml_free(out_gguf_ctx);
            out_gguf_ctx = nullptr;
        }
        error = "failed to load gguf file: " + path;
        return false;
    }
    return true;
}

static int64_t tensor_num_elements(const ace_safetensors::TensorInfo & info) {
    int64_t n = 1;
    for (int64_t d : info.shape) {
        if (d <= 0) {
            return 0;
        }
        n *= d;
    }
    return n;
}

static bool read_tensor_as_f32(
    const ace_safetensors::File & st,
    const ace_safetensors::TensorInfo & info,
    std::vector<float> & out,
    std::string & error) {

    const int64_t n = tensor_num_elements(info);
    if (n <= 0) {
        error = "invalid tensor size for " + info.name;
        return false;
    }
    std::vector<uint8_t> raw(info.nbytes());
    if (!st.read_tensor(info, raw.data(), raw.size(), error)) {
        return false;
    }

    out.resize(static_cast<size_t>(n));
    if (info.dtype == "F32") {
        if (raw.size() != static_cast<size_t>(n) * sizeof(float)) {
            error = "f32 tensor size mismatch: " + info.name;
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

    error = "unsupported dtype for f32 conversion: " + info.dtype;
    return false;
}

static bool try_quantize_matrix(
    ggml_context * ctx,
    const std::string & name,
    int64_t ne0,
    int64_t ne1,
    const std::vector<float> & f32,
    ggml_tensor * & out,
    std::string & error) {

    const ggml_type qtype = get_quant_type_from_env();
    if (qtype == GGML_TYPE_COUNT || !ggml_is_quantized(qtype) || ggml_quantize_requires_imatrix(qtype)) {
        return false;
    }

    const int64_t blk = ggml_blck_size(qtype);
    if (blk <= 0 || (ne0 % blk) != 0) {
        return false;
    }
    if (static_cast<int64_t>(f32.size()) != ne0 * ne1) {
        error = "quantize size mismatch: " + name;
        return false;
    }

    ggml_tensor * qt = ggml_new_tensor_2d(ctx, qtype, ne0, ne1);
    if (!qt || !qt->data) {
        error = "failed to allocate quantized tensor: " + name;
        return false;
    }
    size_t written = ggml_quantize_chunk(qtype, f32.data(), qt->data, 0, ne1, ne0, nullptr);
    if (written == 0) {
        error = "failed to quantize tensor: " + name;
        return false;
    }
    ggml_set_name(qt, name.c_str());
    out = qt;
    return true;
}

static ggml_tensor * load_tensor_1d(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return nullptr;
    }
    if (info->shape.size() != 1) {
        error = "invalid tensor shape for " + name;
        return nullptr;
    }
    std::string dtype_err;
    ggml_type type = ggml_type_from_dtype(info->dtype, dtype_err);
    if (type == GGML_TYPE_COUNT) {
        error = dtype_err;
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, type, info->shape[0]);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    if (!st.read_tensor(*info, t->data, info->nbytes(), error)) {
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_tensor_2d_transposed(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return nullptr;
    }
    if (info->shape.size() != 2) {
        error = "invalid tensor shape for " + name;
        return nullptr;
    }
    std::string dtype_err;
    ggml_type type = ggml_type_from_dtype(info->dtype, dtype_err);
    if (type == GGML_TYPE_COUNT) {
        error = dtype_err;
        return nullptr;
    }

    int64_t out_dim = info->shape[0];
    int64_t in_dim = info->shape[1];
    {
        std::vector<float> f32;
        if (read_tensor_as_f32(st, *info, f32, error)) {
            ggml_tensor * qt = nullptr;
            if (try_quantize_matrix(ctx, name, in_dim, out_dim, f32, qt, error)) {
                return qt;
            }
            if (!error.empty()) {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, in_dim, out_dim);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    if (!st.read_tensor(*info, t->data, info->nbytes(), error)) {
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_tensor_3d_as_2d(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return nullptr;
    }
    if (info->shape.size() != 3) {
        error = "invalid tensor shape for " + name;
        return nullptr;
    }
    if (info->shape[0] != 1) {
        error = "unexpected tensor shape for " + name;
        return nullptr;
    }
    std::string dtype_err;
    ggml_type type = ggml_type_from_dtype(info->dtype, dtype_err);
    if (type == GGML_TYPE_COUNT) {
        error = dtype_err;
        return nullptr;
    }

    const int64_t ne1 = info->shape[1];
    const int64_t ne0 = info->shape[2];
    {
        std::vector<float> f32;
        if (read_tensor_as_f32(st, *info, f32, error)) {
            ggml_tensor * qt = nullptr;
            if (try_quantize_matrix(ctx, name, ne0, ne1, f32, qt, error)) {
                return qt;
            }
            if (!error.empty()) {
                return nullptr;
            }
        } else {
            return nullptr;
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, ne0, ne1);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    if (!st.read_tensor(*info, t->data, info->nbytes(), error)) {
        return nullptr;
    }
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_conv1d_weight_as_linear(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return nullptr;
    }
    if (info->shape.size() != 3) {
        error = "invalid tensor shape for " + name;
        return nullptr;
    }
    std::string dtype_err;
    ggml_type type = ggml_type_from_dtype(info->dtype, dtype_err);
    if (type == GGML_TYPE_COUNT) {
        error = dtype_err;
        return nullptr;
    }

    const int64_t out_channels = info->shape[0];
    const int64_t in_channels = info->shape[1];
    const int64_t kernel = info->shape[2];
    const int64_t in_flat = in_channels * kernel;

    {
        std::vector<float> src_f32;
        if (!read_tensor_as_f32(st, *info, src_f32, error)) {
            return nullptr;
        }
        std::vector<float> mat(static_cast<size_t>(out_channels * in_flat), 0.0f);
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t in = 0; in < in_channels; ++in) {
                for (int64_t k = 0; k < kernel; ++k) {
                    const int64_t src = (out * in_channels + in) * kernel + k;
                    const int64_t dst = (in + k * in_channels) + out * in_flat;
                    mat[static_cast<size_t>(dst)] = src_f32[static_cast<size_t>(src)];
                }
            }
        }
        ggml_tensor * qt = nullptr;
        if (try_quantize_matrix(ctx, name, in_flat, out_channels, mat, qt, error)) {
            return qt;
        }
        if (!error.empty()) {
            return nullptr;
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, in_flat, out_channels);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }

    std::vector<uint8_t> tmp(info->nbytes());
    if (!st.read_tensor(*info, tmp.data(), tmp.size(), error)) {
        return nullptr;
    }

    const size_t elem_size = ggml_type_size(type);
    for (int64_t out = 0; out < out_channels; ++out) {
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src = (out * in_channels + in) * kernel + k;
                const int64_t dst = (in + k * in_channels) + out * in_flat;
                std::memcpy(static_cast<uint8_t *>(t->data) + dst * elem_size,
                            tmp.data() + src * elem_size,
                            elem_size);
            }
        }
    }

    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_convtranspose1d_weight_as_linear(
    ggml_context * ctx,
    const ace_safetensors::File & st,
    const std::string & name,
    std::string & error) {

    const auto * info = st.find(name);
    if (!info) {
        error = "missing tensor: " + name;
        return nullptr;
    }
    if (info->shape.size() != 3) {
        error = "invalid tensor shape for " + name;
        return nullptr;
    }
    std::string dtype_err;
    ggml_type type = ggml_type_from_dtype(info->dtype, dtype_err);
    if (type == GGML_TYPE_COUNT) {
        error = dtype_err;
        return nullptr;
    }

    const int64_t in_channels = info->shape[0];
    const int64_t out_channels = info->shape[1];
    const int64_t kernel = info->shape[2];
    const int64_t out_flat = out_channels * kernel;

    {
        std::vector<float> src_f32;
        if (!read_tensor_as_f32(st, *info, src_f32, error)) {
            return nullptr;
        }
        std::vector<float> mat(static_cast<size_t>(in_channels * out_flat), 0.0f);
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t out = 0; out < out_channels; ++out) {
                for (int64_t k = 0; k < kernel; ++k) {
                    const int64_t src = (in * out_channels + out) * kernel + k;
                    const int64_t dst = in + (out + k * out_channels) * in_channels;
                    mat[static_cast<size_t>(dst)] = src_f32[static_cast<size_t>(src)];
                }
            }
        }
        ggml_tensor * qt = nullptr;
        if (try_quantize_matrix(ctx, name, in_channels, out_flat, mat, qt, error)) {
            return qt;
        }
        if (!error.empty()) {
            return nullptr;
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, type, in_channels, out_flat);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }

    std::vector<uint8_t> tmp(info->nbytes());
    if (!st.read_tensor(*info, tmp.data(), tmp.size(), error)) {
        return nullptr;
    }

    const size_t elem_size = ggml_type_size(type);
    for (int64_t in = 0; in < in_channels; ++in) {
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src = (in * out_channels + out) * kernel + k;
                const int64_t dst = in + (out + k * out_channels) * in_channels;
                std::memcpy(static_cast<uint8_t *>(t->data) + dst * elem_size,
                            tmp.data() + src * elem_size,
                            elem_size);
            }
        }
    }

    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_tensor_1d_from_gguf(
    ggml_context * ctx,
    ggml_context * gguf_ctx,
    const std::string & name,
    std::string & error) {

    ggml_tensor * src = ggml_get_tensor(gguf_ctx, name.c_str());
    if (!src) {
        error = "missing tensor in gguf: " + name;
        return nullptr;
    }
    if (src->ne[1] != 1 || src->ne[2] != 1 || src->ne[3] != 1) {
        error = "invalid 1d tensor shape in gguf: " + name;
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_1d(ctx, src->type, src->ne[0]);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (!src->data || ggml_nbytes(src) < nbytes) {
        error = "invalid tensor data size in gguf: " + name;
        return nullptr;
    }
    std::memcpy(t->data, src->data, nbytes);
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_tensor_2d_from_gguf(
    ggml_context * ctx,
    ggml_context * gguf_ctx,
    const std::string & name,
    std::string & error) {

    ggml_tensor * src = ggml_get_tensor(gguf_ctx, name.c_str());
    if (!src) {
        error = "missing tensor in gguf: " + name;
        return nullptr;
    }
    if (src->ne[2] != 1 || src->ne[3] != 1) {
        error = "invalid 2d tensor shape in gguf: " + name;
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, src->type, src->ne[0], src->ne[1]);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (!src->data || ggml_nbytes(src) < nbytes) {
        error = "invalid tensor data size in gguf: " + name;
        return nullptr;
    }
    std::memcpy(t->data, src->data, nbytes);
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_tensor_3d_as_2d_from_gguf(
    ggml_context * ctx,
    ggml_context * gguf_ctx,
    const std::string & name,
    std::string & error) {

    ggml_tensor * src = ggml_get_tensor(gguf_ctx, name.c_str());
    if (!src) {
        error = "missing tensor in gguf: " + name;
        return nullptr;
    }
    if (src->ne[2] != 1 || src->ne[3] != 1) {
        error = "invalid 3d-as-2d tensor shape in gguf: " + name;
        return nullptr;
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, src->type, src->ne[0], src->ne[1]);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    const size_t nbytes = ggml_nbytes(t);
    if (!src->data || ggml_nbytes(src) < nbytes) {
        error = "invalid tensor data size in gguf: " + name;
        return nullptr;
    }
    std::memcpy(t->data, src->data, nbytes);
    ggml_set_name(t, name.c_str());
    return t;
}

static bool read_gguf_tensor_as_f32(
    ggml_context * gguf_ctx,
    const std::string & name,
    std::vector<float> & out,
    int64_t & ne0,
    int64_t & ne1,
    int64_t & ne2,
    std::string & error) {

    ggml_tensor * src = ggml_get_tensor(gguf_ctx, name.c_str());
    if (!src) {
        error = "missing tensor in gguf: " + name;
        return false;
    }
    if (src->ne[3] != 1) {
        error = "unsupported tensor rank in gguf: " + name;
        return false;
    }
    ne0 = src->ne[0];
    ne1 = src->ne[1];
    ne2 = src->ne[2];
    const int64_t n = ne0 * ne1 * std::max<int64_t>(1, ne2);
    out.resize(static_cast<size_t>(n));

    if (!src->data) {
        error = "tensor has no data in gguf: " + name;
        return false;
    }

    if (src->type == GGML_TYPE_F32) {
        std::memcpy(out.data(), src->data, static_cast<size_t>(n) * sizeof(float));
        return true;
    }
    if (src->type == GGML_TYPE_F16) {
        const auto * p = static_cast<const ggml_fp16_t *>(src->data);
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = ggml_fp16_to_fp32(p[i]);
        }
        return true;
    }
    if (src->type == GGML_TYPE_BF16) {
        const auto * p = static_cast<const ggml_bf16_t *>(src->data);
        for (int64_t i = 0; i < n; ++i) {
            out[static_cast<size_t>(i)] = ggml_bf16_to_fp32(p[i]);
        }
        return true;
    }

    error = "unsupported gguf tensor type for conversion: " + name;
    return false;
}

static ggml_tensor * load_conv1d_weight_as_linear_from_gguf(
    ggml_context * ctx,
    ggml_context * gguf_ctx,
    const std::string & name,
    std::string & error) {

    std::vector<float> src_f32;
    int64_t kernel = 0;
    int64_t in_channels = 0;
    int64_t out_channels = 0;
    if (!read_gguf_tensor_as_f32(gguf_ctx, name, src_f32, kernel, in_channels, out_channels, error)) {
        return nullptr;
    }
    if (kernel <= 0 || in_channels <= 0 || out_channels <= 0) {
        error = "invalid conv1d tensor shape in gguf: " + name;
        return nullptr;
    }

    const int64_t in_flat = in_channels * kernel;
    std::vector<float> mat(static_cast<size_t>(out_channels * in_flat), 0.0f);
    for (int64_t out = 0; out < out_channels; ++out) {
        for (int64_t in = 0; in < in_channels; ++in) {
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src = k + in * kernel + out * kernel * in_channels;
                const int64_t dst = (in + k * in_channels) + out * in_flat;
                mat[static_cast<size_t>(dst)] = src_f32[static_cast<size_t>(src)];
            }
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_flat, out_channels);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    std::memcpy(t->data, mat.data(), mat.size() * sizeof(float));
    ggml_set_name(t, name.c_str());
    return t;
}

static ggml_tensor * load_convtranspose1d_weight_as_linear_from_gguf(
    ggml_context * ctx,
    ggml_context * gguf_ctx,
    const std::string & name,
    std::string & error) {

    std::vector<float> src_f32;
    int64_t kernel = 0;
    int64_t out_channels = 0;
    int64_t in_channels = 0;
    if (!read_gguf_tensor_as_f32(gguf_ctx, name, src_f32, kernel, out_channels, in_channels, error)) {
        return nullptr;
    }
    if (kernel <= 0 || in_channels <= 0 || out_channels <= 0) {
        error = "invalid convtranspose tensor shape in gguf: " + name;
        return nullptr;
    }

    const int64_t out_flat = out_channels * kernel;
    std::vector<float> mat(static_cast<size_t>(in_channels * out_flat), 0.0f);
    for (int64_t in = 0; in < in_channels; ++in) {
        for (int64_t out = 0; out < out_channels; ++out) {
            for (int64_t k = 0; k < kernel; ++k) {
                const int64_t src = k + out * kernel + in * kernel * out_channels;
                const int64_t dst = in + (out + k * out_channels) * in_channels;
                mat[static_cast<size_t>(dst)] = src_f32[static_cast<size_t>(src)];
            }
        }
    }

    ggml_tensor * t = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_channels, out_flat);
    if (!t || !t->data) {
        error = "failed to allocate tensor: " + name;
        return nullptr;
    }
    std::memcpy(t->data, mat.data(), mat.size() * sizeof(float));
    ggml_set_name(t, name.c_str());
    return t;
}

static bool compute_model_memory(const ace_safetensors::File & st, size_t & bytes) {
    bytes = 0;
    for (const auto & t : st.tensors) {
        bytes += t.nbytes();
    }
    return true;
}

static bool compute_model_memory_gguf(const gguf_context * gguf, ggml_context * gguf_ctx, size_t & bytes) {
    bytes = 0;
    const int64_t n_tensors = gguf_get_n_tensors(gguf);
    for (int64_t i = 0; i < n_tensors; ++i) {
        const char * name = gguf_get_tensor_name(gguf, i);
        ggml_tensor * t = ggml_get_tensor(gguf_ctx, name);
        if (!t) {
            continue;
        }
        bytes += ggml_nbytes(t);
    }
    return true;
}

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
    model.layers.clear();
    model.loaded = false;
}

bool load_model_from_dir(const std::string & dir, Model & out, std::string & error) {
    const std::filesystem::path p(dir);
    const bool input_is_gguf = p.extension() == ".gguf";
    const std::filesystem::path model_root = input_is_gguf ? p.parent_path() : p;
    std::string cfg_path = (model_root / "config.json").string();
    std::string st_path = (model_root / "model.safetensors").string();
    const std::string gguf_path = resolve_gguf_path(dir);
    const bool use_gguf = !gguf_path.empty();

    Config cfg;
    if (!load_config(cfg_path, cfg, error)) {
        return false;
    }

    gguf_context * gguf = nullptr;
    ggml_context * gguf_ctx = nullptr;
    ace_safetensors::File st;
    if (use_gguf) {
        if (!load_gguf_file(gguf_path, gguf, gguf_ctx, error)) {
            return false;
        }
    } else {
        if (!st.load(st_path, error)) {
            return false;
        }
    }

    size_t weight_bytes = 0;
    if (use_gguf) {
        compute_model_memory_gguf(gguf, gguf_ctx, weight_bytes);
    } else {
        compute_model_memory(st, weight_bytes);
    }

    const size_t n_layers = static_cast<size_t>(cfg.num_hidden_layers);
    const size_t n_lyric_layers = static_cast<size_t>(std::max(0, cfg.num_lyric_encoder_hidden_layers));
    const size_t n_timbre_layers = static_cast<size_t>(std::max(0, cfg.num_timbre_encoder_hidden_layers));
    const size_t n_tensors = 72 + n_layers * 19 + n_lyric_layers * 11 + n_timbre_layers * 11;
    size_t overhead = ggml_tensor_overhead() * n_tensors;
    size_t ctx_size = weight_bytes + overhead + (32 * 1024 * 1024);

    void * buffer = std::malloc(ctx_size);
    if (!buffer) {
        error = "failed to allocate model buffer";
        return false;
    }

    ggml_init_params params{};
    params.mem_size = ctx_size;
    params.mem_buffer = buffer;
    params.no_alloc = false;

    ggml_context * ctx = ggml_init(params);
    if (!ctx) {
        std::free(buffer);
        error = "ggml_init failed";
        return false;
    }

    Model model;
    model.cfg = cfg;
    model.ctx = ctx;
    model.ctx_buffer = buffer;
    model.ctx_buffer_size = ctx_size;

    auto cleanup_source = [&]() {
        if (gguf) {
            gguf_free(gguf);
            gguf = nullptr;
        }
        if (gguf_ctx) {
            ggml_free(gguf_ctx);
            gguf_ctx = nullptr;
        }
    };
    auto fail = [&]() -> bool {
        free_model(model);
        cleanup_source();
        return false;
    };
    auto has_tensor = [&](const std::string & name) -> bool {
        if (use_gguf) {
            return ggml_get_tensor(gguf_ctx, name.c_str()) != nullptr;
        }
        return st.find(name) != nullptr;
    };
    auto load_1d = [&](const std::string & name) -> ggml_tensor * {
        if (use_gguf) {
            return load_tensor_1d_from_gguf(ctx, gguf_ctx, name, error);
        }
        return load_tensor_1d(ctx, st, name, error);
    };
    auto load_2d = [&](const std::string & name) -> ggml_tensor * {
        if (use_gguf) {
            return load_tensor_2d_from_gguf(ctx, gguf_ctx, name, error);
        }
        return load_tensor_2d_transposed(ctx, st, name, error);
    };
    auto load_3d_as_2d = [&](const std::string & name) -> ggml_tensor * {
        if (use_gguf) {
            return load_tensor_3d_as_2d_from_gguf(ctx, gguf_ctx, name, error);
        }
        return load_tensor_3d_as_2d(ctx, st, name, error);
    };
    auto load_proj_in = [&](const std::string & name) -> ggml_tensor * {
        if (use_gguf) {
            return load_conv1d_weight_as_linear_from_gguf(ctx, gguf_ctx, name, error);
        }
        return load_conv1d_weight_as_linear(ctx, st, name, error);
    };
    auto load_proj_out = [&](const std::string & name) -> ggml_tensor * {
        if (use_gguf) {
            return load_convtranspose1d_weight_as_linear_from_gguf(ctx, gguf_ctx, name, error);
        }
        return load_convtranspose1d_weight_as_linear(ctx, st, name, error);
    };

    model.proj_in_w = load_proj_in("decoder.proj_in.1.weight");
    if (!model.proj_in_w) { return fail(); }
    model.proj_in_b = load_1d("decoder.proj_in.1.bias");
    if (!model.proj_in_b) { return fail(); }

    model.proj_out_w = load_proj_out("decoder.proj_out.1.weight");
    if (!model.proj_out_w) { return fail(); }
    model.proj_out_b = load_1d("decoder.proj_out.1.bias");
    if (!model.proj_out_b) { return fail(); }

    model.condition_w = load_2d("decoder.condition_embedder.weight");
    if (!model.condition_w) { return fail(); }
    model.condition_b = load_1d("decoder.condition_embedder.bias");
    if (!model.condition_b) { return fail(); }

    if (has_tensor("encoder.text_projector.weight")) {
        model.text_projector_w = load_2d("encoder.text_projector.weight");
        if (!model.text_projector_w) { return fail(); }
    }
    if (has_tensor("encoder.lyric_encoder.embed_tokens.weight")) {
        model.lyric_embed_w = load_2d("encoder.lyric_encoder.embed_tokens.weight");
        if (!model.lyric_embed_w) { return fail(); }
    }
    if (has_tensor("encoder.lyric_encoder.embed_tokens.bias")) {
        model.lyric_embed_b = load_1d("encoder.lyric_encoder.embed_tokens.bias");
        if (!model.lyric_embed_b) { return fail(); }
    }
    if (has_tensor("encoder.lyric_encoder.norm.weight")) {
        model.lyric_norm = load_1d("encoder.lyric_encoder.norm.weight");
        if (!model.lyric_norm) { return fail(); }
    }

    if (n_lyric_layers > 0) {
        model.lyric_layers.resize(n_lyric_layers);
        for (size_t i = 0; i < n_lyric_layers; ++i) {
            EncoderLayer layer;
            std::ostringstream prefix;
            prefix << "encoder.lyric_encoder.layers." << i << ".";
            const std::string p = prefix.str();

            layer.input_norm = load_1d(p + "input_layernorm.weight");
            if (!layer.input_norm) { return fail(); }
            layer.self_attn.w_q = load_2d(p + "self_attn.q_proj.weight");
            if (!layer.self_attn.w_q) { return fail(); }
            layer.self_attn.w_k = load_2d(p + "self_attn.k_proj.weight");
            if (!layer.self_attn.w_k) { return fail(); }
            layer.self_attn.w_v = load_2d(p + "self_attn.v_proj.weight");
            if (!layer.self_attn.w_v) { return fail(); }
            layer.self_attn.w_o = load_2d(p + "self_attn.o_proj.weight");
            if (!layer.self_attn.w_o) { return fail(); }
            layer.self_attn.q_norm = load_1d(p + "self_attn.q_norm.weight");
            if (!layer.self_attn.q_norm) { return fail(); }
            layer.self_attn.k_norm = load_1d(p + "self_attn.k_norm.weight");
            if (!layer.self_attn.k_norm) { return fail(); }

            layer.post_attn_norm = load_1d(p + "post_attention_layernorm.weight");
            if (!layer.post_attn_norm) { return fail(); }
            layer.mlp.w_gate = load_2d(p + "mlp.gate_proj.weight");
            if (!layer.mlp.w_gate) { return fail(); }
            layer.mlp.w_up = load_2d(p + "mlp.up_proj.weight");
            if (!layer.mlp.w_up) { return fail(); }
            layer.mlp.w_down = load_2d(p + "mlp.down_proj.weight");
            if (!layer.mlp.w_down) { return fail(); }

            if (i < cfg.layer_types.size()) {
                layer.use_sliding_window = (cfg.layer_types[i] == "sliding_attention");
            }
            model.lyric_layers[i] = layer;
        }
    }

    if (has_tensor("encoder.timbre_encoder.embed_tokens.weight")) {
        model.timbre_embed_w = load_2d("encoder.timbre_encoder.embed_tokens.weight");
        if (!model.timbre_embed_w) { return fail(); }
    }
    if (has_tensor("encoder.timbre_encoder.embed_tokens.bias")) {
        model.timbre_embed_b = load_1d("encoder.timbre_encoder.embed_tokens.bias");
        if (!model.timbre_embed_b) { return fail(); }
    }
    if (has_tensor("encoder.timbre_encoder.norm.weight")) {
        model.timbre_norm = load_1d("encoder.timbre_encoder.norm.weight");
        if (!model.timbre_norm) { return fail(); }
    }
    if (has_tensor("encoder.timbre_encoder.special_token")) {
        ggml_tensor * t = load_3d_as_2d("encoder.timbre_encoder.special_token");
        if (!t) { return fail(); }
        model.timbre_special_token = ggml_reshape_1d(ctx, t, t->ne[0]);
    }

    if (n_timbre_layers > 0) {
        model.timbre_layers.resize(n_timbre_layers);
        for (size_t i = 0; i < n_timbre_layers; ++i) {
            EncoderLayer layer;
            std::ostringstream prefix;
            prefix << "encoder.timbre_encoder.layers." << i << ".";
            const std::string p = prefix.str();

            layer.input_norm = load_1d(p + "input_layernorm.weight");
            if (!layer.input_norm) { return fail(); }
            layer.self_attn.w_q = load_2d(p + "self_attn.q_proj.weight");
            if (!layer.self_attn.w_q) { return fail(); }
            layer.self_attn.w_k = load_2d(p + "self_attn.k_proj.weight");
            if (!layer.self_attn.w_k) { return fail(); }
            layer.self_attn.w_v = load_2d(p + "self_attn.v_proj.weight");
            if (!layer.self_attn.w_v) { return fail(); }
            layer.self_attn.w_o = load_2d(p + "self_attn.o_proj.weight");
            if (!layer.self_attn.w_o) { return fail(); }
            layer.self_attn.q_norm = load_1d(p + "self_attn.q_norm.weight");
            if (!layer.self_attn.q_norm) { return fail(); }
            layer.self_attn.k_norm = load_1d(p + "self_attn.k_norm.weight");
            if (!layer.self_attn.k_norm) { return fail(); }

            layer.post_attn_norm = load_1d(p + "post_attention_layernorm.weight");
            if (!layer.post_attn_norm) { return fail(); }
            layer.mlp.w_gate = load_2d(p + "mlp.gate_proj.weight");
            if (!layer.mlp.w_gate) { return fail(); }
            layer.mlp.w_up = load_2d(p + "mlp.up_proj.weight");
            if (!layer.mlp.w_up) { return fail(); }
            layer.mlp.w_down = load_2d(p + "mlp.down_proj.weight");
            if (!layer.mlp.w_down) { return fail(); }

            if (i < cfg.layer_types.size()) {
                layer.use_sliding_window = (cfg.layer_types[i] == "sliding_attention");
            }
            model.timbre_layers[i] = layer;
        }
    }

    model.norm_out = load_1d("decoder.norm_out.weight");
    if (!model.norm_out) { return fail(); }
    model.out_scale_shift_table = load_3d_as_2d("decoder.scale_shift_table");
    if (!model.out_scale_shift_table) { return fail(); }

    model.time_embed.w1 = load_2d("decoder.time_embed.linear_1.weight");
    if (!model.time_embed.w1) { return fail(); }
    model.time_embed.b1 = load_1d("decoder.time_embed.linear_1.bias");
    if (!model.time_embed.b1) { return fail(); }
    model.time_embed.w2 = load_2d("decoder.time_embed.linear_2.weight");
    if (!model.time_embed.w2) { return fail(); }
    model.time_embed.b2 = load_1d("decoder.time_embed.linear_2.bias");
    if (!model.time_embed.b2) { return fail(); }
    model.time_embed.w_proj = load_2d("decoder.time_embed.time_proj.weight");
    if (!model.time_embed.w_proj) { return fail(); }
    model.time_embed.b_proj = load_1d("decoder.time_embed.time_proj.bias");
    if (!model.time_embed.b_proj) { return fail(); }

    model.time_embed_r.w1 = load_2d("decoder.time_embed_r.linear_1.weight");
    if (!model.time_embed_r.w1) { return fail(); }
    model.time_embed_r.b1 = load_1d("decoder.time_embed_r.linear_1.bias");
    if (!model.time_embed_r.b1) { return fail(); }
    model.time_embed_r.w2 = load_2d("decoder.time_embed_r.linear_2.weight");
    if (!model.time_embed_r.w2) { return fail(); }
    model.time_embed_r.b2 = load_1d("decoder.time_embed_r.linear_2.bias");
    if (!model.time_embed_r.b2) { return fail(); }
    model.time_embed_r.w_proj = load_2d("decoder.time_embed_r.time_proj.weight");
    if (!model.time_embed_r.w_proj) { return fail(); }
    model.time_embed_r.b_proj = load_1d("decoder.time_embed_r.time_proj.bias");
    if (!model.time_embed_r.b_proj) { return fail(); }

    model.layers.resize(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        Layer layer;
        std::ostringstream prefix;
        prefix << "decoder.layers." << i << ".";
        const std::string p = prefix.str();

        layer.self_attn_norm = load_1d(p + "self_attn_norm.weight");
        if (!layer.self_attn_norm) { return fail(); }
        layer.self_attn.w_q = load_2d(p + "self_attn.q_proj.weight");
        if (!layer.self_attn.w_q) { return fail(); }
        layer.self_attn.w_k = load_2d(p + "self_attn.k_proj.weight");
        if (!layer.self_attn.w_k) { return fail(); }
        layer.self_attn.w_v = load_2d(p + "self_attn.v_proj.weight");
        if (!layer.self_attn.w_v) { return fail(); }
        layer.self_attn.w_o = load_2d(p + "self_attn.o_proj.weight");
        if (!layer.self_attn.w_o) { return fail(); }
        layer.self_attn.q_norm = load_1d(p + "self_attn.q_norm.weight");
        if (!layer.self_attn.q_norm) { return fail(); }
        layer.self_attn.k_norm = load_1d(p + "self_attn.k_norm.weight");
        if (!layer.self_attn.k_norm) { return fail(); }

        layer.cross_attn_norm = load_1d(p + "cross_attn_norm.weight");
        if (!layer.cross_attn_norm) { return fail(); }
        layer.cross_attn.w_q = load_2d(p + "cross_attn.q_proj.weight");
        if (!layer.cross_attn.w_q) { return fail(); }
        layer.cross_attn.w_k = load_2d(p + "cross_attn.k_proj.weight");
        if (!layer.cross_attn.w_k) { return fail(); }
        layer.cross_attn.w_v = load_2d(p + "cross_attn.v_proj.weight");
        if (!layer.cross_attn.w_v) { return fail(); }
        layer.cross_attn.w_o = load_2d(p + "cross_attn.o_proj.weight");
        if (!layer.cross_attn.w_o) { return fail(); }
        layer.cross_attn.q_norm = load_1d(p + "cross_attn.q_norm.weight");
        if (!layer.cross_attn.q_norm) { return fail(); }
        layer.cross_attn.k_norm = load_1d(p + "cross_attn.k_norm.weight");
        if (!layer.cross_attn.k_norm) { return fail(); }

        layer.mlp_norm = load_1d(p + "mlp_norm.weight");
        if (!layer.mlp_norm) { return fail(); }
        layer.mlp.w_gate = load_2d(p + "mlp.gate_proj.weight");
        if (!layer.mlp.w_gate) { return fail(); }
        layer.mlp.w_up = load_2d(p + "mlp.up_proj.weight");
        if (!layer.mlp.w_up) { return fail(); }
        layer.mlp.w_down = load_2d(p + "mlp.down_proj.weight");
        if (!layer.mlp.w_down) { return fail(); }

        layer.scale_shift_table = load_3d_as_2d(p + "scale_shift_table");
        if (!layer.scale_shift_table) { return fail(); }

        if (i < cfg.layer_types.size()) {
            layer.use_sliding_window = (cfg.layer_types[i] == "sliding_attention");
        }
        model.layers[i] = layer;
    }

    cleanup_source();
    model.loaded = true;
    out = std::move(model);
    return true;
}

static ggml_tensor * cast_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static ggml_tensor * rms_norm(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * weight,
    float eps) {

    ggml_tensor * y = ggml_rms_norm(ctx, x, eps);
    ggml_tensor * w = cast_f32(ctx, weight);
    return ggml_mul(ctx, y, w);
}

static ggml_tensor * add_bias(
    ggml_context * ctx,
    ggml_tensor * x,
    ggml_tensor * bias) {

    ggml_tensor * b = cast_f32(ctx, bias);
    ggml_tensor * b_rep = ggml_repeat(ctx, b, x);
    return ggml_add(ctx, x, b_rep);
}

static ggml_tensor * repeat_kv_interleave(
    ggml_context * ctx,
    ggml_tensor * t,
    int32_t n_rep) {

    if (n_rep <= 1) {
        return t;
    }
    ggml_tensor * t4 = ggml_reshape_4d(ctx, t, t->ne[0], t->ne[1], 1, t->ne[2]);
    ggml_tensor * shape = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], n_rep, t->ne[2]);
    ggml_tensor * rep = ggml_repeat(ctx, t4, shape);
    return ggml_reshape_3d(ctx, rep, t->ne[0], t->ne[1], t->ne[2] * n_rep);
}

static ggml_tensor * build_attention_mask(
    ggml_context * ctx,
    int64_t q_len,
    int64_t k_len,
    const int32_t * attention_mask,
    bool causal,
    bool sliding,
    int32_t sliding_window) {

    if (!attention_mask && !causal && !sliding) {
        return nullptr;
    }

    ggml_tensor * mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_len, q_len, 1);
    float * mdata = static_cast<float *>(mask->data);
    for (int64_t q = 0; q < q_len; ++q) {
        for (int64_t k = 0; k < k_len; ++k) {
            bool allow = true;
            if (causal && k > q) {
                allow = false;
            }
            if (sliding) {
                const int64_t diff = static_cast<int64_t>(q) - static_cast<int64_t>(k);
                if (causal) {
                    if (diff < 0 || diff > sliding_window) {
                        allow = false;
                    }
                } else {
                    if (std::llabs(diff) > sliding_window) {
                        allow = false;
                    }
                }
            }
            if (attention_mask && k < k_len && attention_mask[k] == 0) {
                allow = false;
            }
            mdata[k + q * k_len] = allow ? 0.0f : -INFINITY;
        }
    }
    ggml_set_input(mask);
    return mask;
}

static ggml_tensor * attention(
    ggml_context * ctx,
    const Config & cfg,
    const AttnWeights & w,
    ggml_tensor * query_input,
    ggml_tensor * key_value_input,
    ggml_tensor * pos,
    const int32_t * attention_mask,
    int32_t q_len,
    int32_t k_len,
    bool causal,
    bool sliding,
    int32_t sliding_window,
    bool use_rope) {

    const int32_t n_heads = cfg.num_attention_heads;
    const int32_t n_kv = cfg.num_key_value_heads;
    const int32_t head_dim = cfg.head_dim;

    ggml_tensor * q = ggml_mul_mat(ctx, w.w_q, query_input);
    ggml_tensor * k = ggml_mul_mat(ctx, w.w_k, key_value_input);
    ggml_tensor * v = ggml_mul_mat(ctx, w.w_v, key_value_input);

    q = ggml_reshape_3d(ctx, q, head_dim, n_heads, q_len);
    k = ggml_reshape_3d(ctx, k, head_dim, n_kv, k_len);
    v = ggml_reshape_3d(ctx, v, head_dim, n_kv, k_len);

    q = rms_norm(ctx, q, w.q_norm, cfg.rms_norm_eps);
    k = rms_norm(ctx, k, w.k_norm, cfg.rms_norm_eps);

    if (use_rope) {
        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, cfg.max_position_embeddings,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, cfg.max_position_embeddings,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
    }

    ggml_tensor * q4 = ggml_reshape_4d(ctx, q, head_dim, n_heads, q_len, 1);
    q4 = ggml_permute(ctx, q4, 0, 2, 1, 3);
    q4 = ggml_cont(ctx, q4);
    q = ggml_reshape_3d(ctx, q4, head_dim, q_len, n_heads);

    ggml_tensor * k4 = ggml_reshape_4d(ctx, k, head_dim, n_kv, k_len, 1);
    k4 = ggml_permute(ctx, k4, 0, 2, 1, 3);
    k4 = ggml_cont(ctx, k4);
    k = ggml_reshape_3d(ctx, k4, head_dim, k_len, n_kv);

    ggml_tensor * v4a = ggml_reshape_4d(ctx, v, head_dim, n_kv, k_len, 1);
    v4a = ggml_permute(ctx, v4a, 0, 2, 1, 3);
    v4a = ggml_cont(ctx, v4a);
    v = ggml_reshape_3d(ctx, v4a, head_dim, k_len, n_kv);

    if (n_kv != n_heads) {
        const int32_t n_rep = n_heads / n_kv;
        k = repeat_kv_interleave(ctx, k, n_rep);
        v = repeat_kv_interleave(ctx, v, n_rep);
    }

    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    ggml_tensor * mask = build_attention_mask(ctx, q_len, k_len, attention_mask, causal, sliding, sliding_window);

    // stable-diffusion.cpp style: scale + add mask then softmax.
    // Avoid inplace ops here; some ggml builds can hit graph parent traversal issues.
    ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
    ggml_mul_mat_set_prec(kq, GGML_PREC_F32);
    kq = ggml_cont(ctx, kq);
    kq = ggml_scale(ctx, kq, scale);
    if (mask) {
        kq = ggml_add(ctx, kq, mask);
    }
    ggml_tensor * attn = ggml_soft_max(ctx, kq);

    ggml_tensor * v4 = ggml_reshape_4d(ctx, v, head_dim, k_len, n_heads, 1);
    v4 = ggml_permute(ctx, v4, 1, 0, 2, 3);
    v4 = ggml_cont(ctx, v4);
    ggml_tensor * v3 = ggml_reshape_3d(ctx, v4, k_len, head_dim, n_heads);
    ggml_tensor * attn_out = ggml_mul_mat(ctx, v3, attn);

    ggml_tensor * attn4 = ggml_reshape_4d(ctx, attn_out, head_dim, q_len, n_heads, 1);
    attn4 = ggml_permute(ctx, attn4, 0, 2, 1, 3);
    attn4 = ggml_cont(ctx, attn4);
    ggml_tensor * attn2 = ggml_reshape_2d(ctx, attn4, head_dim * n_heads, q_len);
    ggml_tensor * out = ggml_mul_mat(ctx, w.w_o, attn2);
    return out;
}

static ggml_tensor * build_timestep_freq(
    ggml_context * ctx,
    float t,
    int32_t dim,
    float scale) {

    ggml_tensor * freq = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, dim, 1);
    float * data = static_cast<float *>(freq->data);
    const float t_scaled = t * scale;
    const int32_t half = dim / 2;
    const float log_max = std::log(10000.0f);
    for (int32_t i = 0; i < half; ++i) {
        const float exponent = -log_max * static_cast<float>(i) / static_cast<float>(half);
        const float f = std::exp(exponent);
        const float arg = t_scaled * f;
        data[i] = std::cos(arg);
        data[i + half] = std::sin(arg);
    }
    if (dim % 2 != 0) {
        data[dim - 1] = 0.0f;
    }
    ggml_set_input(freq);
    return freq;
}

static void timestep_forward(
    ggml_context * ctx,
    const TimestepWeights & tw,
    float t,
    int32_t hidden,
    ggml_tensor ** out_temb,
    ggml_tensor ** out_proj) {

    ggml_tensor * t_freq = build_timestep_freq(ctx, t, 256, 1000.0f);
    ggml_tensor * temb = ggml_mul_mat(ctx, tw.w1, t_freq);
    temb = add_bias(ctx, temb, tw.b1);
    temb = ggml_silu(ctx, temb);
    temb = ggml_mul_mat(ctx, tw.w2, temb);
    temb = add_bias(ctx, temb, tw.b2);

    ggml_tensor * proj = ggml_silu(ctx, temb);
    proj = ggml_mul_mat(ctx, tw.w_proj, proj);
    proj = add_bias(ctx, proj, tw.b_proj);
    proj = ggml_reshape_2d(ctx, proj, hidden, 6);

    *out_temb = temb;
    *out_proj = proj;
}

static ggml_tensor * view_col(ggml_context * ctx, ggml_tensor * t, int col) {
    const size_t elem_size = ggml_type_size(t->type);
    const size_t offset = static_cast<size_t>(col) * static_cast<size_t>(t->ne[0]) * elem_size;
    return ggml_view_1d(ctx, t, t->ne[0], offset);
}

ggml_tensor * forward_dit(
    ggml_context * ctx,
    const Model & model,
    const float * hidden_states,
    const float * context_latents,
    const float * encoder_hidden_states,
    const int32_t * attention_mask,
    const int32_t * encoder_attention_mask,
    int32_t seq_len,
    int32_t enc_len,
    float timestep,
    float timestep_r,
    ggml_tensor ** input_x0_out,
    ggml_tensor ** input_enc_out) {

    const Config & cfg = model.cfg;
    const int32_t audio_dim = cfg.audio_acoustic_hidden_dim;
    const int32_t ctx_dim = cfg.in_channels - audio_dim;
    const int32_t patch = cfg.patch_size;

    if (input_x0_out) {
        *input_x0_out = nullptr;
    }
    if (input_enc_out) {
        *input_enc_out = nullptr;
    }

    int32_t pad_len = 0;
    if (seq_len % patch != 0) {
        pad_len = patch - (seq_len % patch);
    }
    const int32_t seq_len_padded = seq_len + pad_len;
    const int32_t seq_len_p = seq_len_padded / patch;

    ggml_tensor * x0 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.in_channels, seq_len_padded);
    if (!x0) {
        return nullptr;
    }
    ggml_set_input(x0);
    if (input_x0_out) {
        *input_x0_out = x0;
    }
    if ((hidden_states || context_latents) && x0->data == nullptr) {
        // no_alloc contexts require caller-side tensor_set path.
        return nullptr;
    }
    if (x0->data) {
        float * xdata = static_cast<float *>(x0->data);
        std::memset(xdata, 0, static_cast<size_t>(cfg.in_channels) * seq_len_padded * sizeof(float));
        for (int32_t t = 0; t < seq_len; ++t) {
            if (context_latents) {
                for (int32_t c = 0; c < ctx_dim; ++c) {
                    xdata[c + t * cfg.in_channels] = context_latents[t * ctx_dim + c];
                }
            }
            if (hidden_states) {
                for (int32_t c = 0; c < audio_dim; ++c) {
                    xdata[ctx_dim + c + t * cfg.in_channels] = hidden_states[t * audio_dim + c];
                }
            }
        }
    }

    ggml_tensor * x3 = ggml_reshape_3d(ctx, x0, cfg.in_channels, patch, seq_len_p);
    ggml_tensor * x_patch = ggml_reshape_2d(ctx, x3, cfg.in_channels * patch, seq_len_p);
    ggml_tensor * x = ggml_mul_mat(ctx, model.proj_in_w, x_patch);
    x = add_bias(ctx, x, model.proj_in_b);

    ggml_tensor * enc = nullptr;
    if (enc_len > 0) {
        if (!encoder_hidden_states && input_enc_out == nullptr) {
            return nullptr;
        }
        ggml_tensor * enc_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden_size, enc_len);
        if (!enc_in) {
            return nullptr;
        }
        ggml_set_input(enc_in);
        if (input_enc_out) {
            *input_enc_out = enc_in;
        }
        if (encoder_hidden_states && enc_in->data == nullptr) {
            return nullptr;
        }
        if (enc_in->data) {
            float * edata = static_cast<float *>(enc_in->data);
            if (encoder_hidden_states) {
                for (int32_t t = 0; t < enc_len; ++t) {
                    for (int32_t c = 0; c < cfg.hidden_size; ++c) {
                        edata[c + t * cfg.hidden_size] = encoder_hidden_states[t * cfg.hidden_size + c];
                    }
                }
            } else {
                std::memset(edata, 0, static_cast<size_t>(cfg.hidden_size) * static_cast<size_t>(enc_len) * sizeof(float));
            }
        }
        enc = ggml_mul_mat(ctx, model.condition_w, enc_in);
        enc = add_bias(ctx, enc, model.condition_b);
    }

    ggml_tensor * temb_t = nullptr;
    ggml_tensor * proj_t = nullptr;
    ggml_tensor * temb_r = nullptr;
    ggml_tensor * proj_r = nullptr;
    timestep_forward(ctx, model.time_embed, timestep, cfg.hidden_size, &temb_t, &proj_t);
    timestep_forward(ctx, model.time_embed_r, timestep - timestep_r, cfg.hidden_size, &temb_r, &proj_r);

    ggml_tensor * temb = ggml_add(ctx, temb_t, temb_r);         // [hidden, 1]
    ggml_tensor * proj = ggml_add(ctx, proj_t, proj_r);         // [hidden, 6]

    ggml_tensor * ones = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, cfg.hidden_size);
    float * ones_data = static_cast<float *>(ones->data);
    for (int32_t i = 0; i < cfg.hidden_size; ++i) {
        ones_data[i] = 1.0f;
    }
    ggml_set_input(ones);

    std::vector<int32_t> patch_mask;
    const int32_t * attn_mask_ptr = nullptr;
    if (attention_mask) {
        patch_mask.assign(seq_len_p, 0);
        for (int32_t p = 0; p < seq_len_p; ++p) {
            int32_t val = 0;
            for (int32_t k = 0; k < patch; ++k) {
                int32_t idx = p * patch + k;
                if (idx < seq_len && attention_mask[idx] != 0) {
                    val = 1;
                    break;
                }
            }
            patch_mask[p] = val;
        }
        attn_mask_ptr = patch_mask.data();
    }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, seq_len_p);
    for (int32_t i = 0; i < seq_len_p; ++i) {
        ggml_set_i32_1d(pos, i, i);
    }
    ggml_set_input(pos);

    size_t max_layers = model.layers.size();
    if (const char * env_layers = std::getenv("ACE_GGML_DIT_MAX_LAYERS")) {
        char * endp = nullptr;
        const long long val = std::strtoll(env_layers, &endp, 10);
        if (endp && endp != env_layers && val > 0) {
            max_layers = std::min(max_layers, static_cast<size_t>(val));
        }
    }

    for (size_t i = 0; i < max_layers; ++i) {
        const Layer & layer = model.layers[i];

        ggml_tensor * scale_shift = ggml_add(ctx, cast_f32(ctx, layer.scale_shift_table), proj);
        ggml_tensor * shift_msa = view_col(ctx, scale_shift, 0);
        ggml_tensor * scale_msa = view_col(ctx, scale_shift, 1);
        ggml_tensor * gate_msa = view_col(ctx, scale_shift, 2);
        ggml_tensor * c_shift = view_col(ctx, scale_shift, 3);
        ggml_tensor * c_scale = view_col(ctx, scale_shift, 4);
        ggml_tensor * c_gate = view_col(ctx, scale_shift, 5);

        ggml_tensor * norm = rms_norm(ctx, x, layer.self_attn_norm, cfg.rms_norm_eps);
        ggml_tensor * scale_msa1 = ggml_add(ctx, scale_msa, ones);
        ggml_tensor * scale_msa_b = ggml_repeat(ctx, scale_msa1, norm);
        ggml_tensor * shift_msa_b = ggml_repeat(ctx, shift_msa, norm);
        ggml_tensor * norm_msa = ggml_add(ctx, ggml_mul(ctx, norm, scale_msa_b), shift_msa_b);

        ggml_tensor * attn_out = attention(
            ctx,
            cfg,
            layer.self_attn,
            norm_msa,
            norm_msa,
            pos,
            attn_mask_ptr,
            seq_len_p,
            seq_len_p,
            false,
            layer.use_sliding_window,
            cfg.sliding_window,
            true);

        ggml_tensor * gate_b = ggml_repeat(ctx, gate_msa, attn_out);
        ggml_tensor * gated_attn = ggml_mul(ctx, attn_out, gate_b);
        x = ggml_add(ctx, x, gated_attn);

        if (layer.use_cross_attention && enc) {
            ggml_tensor * cross_norm = rms_norm(ctx, x, layer.cross_attn_norm, cfg.rms_norm_eps);
            const int32_t * enc_mask = encoder_attention_mask ? encoder_attention_mask : nullptr;
            ggml_tensor * cross_out = attention(
                ctx,
                cfg,
                layer.cross_attn,
                cross_norm,
                enc,
                nullptr,
                enc_mask,
                seq_len_p,
                enc_len,
                false,
                false,
                0,
                false);
            x = ggml_add(ctx, x, cross_out);
        }

        ggml_tensor * mlp_norm = rms_norm(ctx, x, layer.mlp_norm, cfg.rms_norm_eps);
        ggml_tensor * c_scale1 = ggml_add(ctx, c_scale, ones);
        ggml_tensor * c_scale_b = ggml_repeat(ctx, c_scale1, mlp_norm);
        ggml_tensor * c_shift_b = ggml_repeat(ctx, c_shift, mlp_norm);
        ggml_tensor * mlp_in = ggml_add(ctx, ggml_mul(ctx, mlp_norm, c_scale_b), c_shift_b);

        ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.w_gate, mlp_in);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.mlp.w_up, mlp_in);
        ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * down = ggml_mul_mat(ctx, layer.mlp.w_down, act);
        ggml_tensor * c_gate_b = ggml_repeat(ctx, c_gate, down);
        ggml_tensor * gated_down = ggml_mul(ctx, down, c_gate_b);
        x = ggml_add(ctx, x, gated_down);
    }

    ggml_tensor * temb_vec = ggml_reshape_1d(ctx, temb, cfg.hidden_size);
    ggml_tensor * out_shape = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, cfg.hidden_size, 2);
    ggml_set_input(out_shape);
    ggml_tensor * temb_rep = ggml_repeat(ctx, temb_vec, out_shape);
    ggml_tensor * out_scale_shift = ggml_add(ctx, cast_f32(ctx, model.out_scale_shift_table), temb_rep);
    ggml_tensor * out_shift = view_col(ctx, out_scale_shift, 0);
    ggml_tensor * out_scale = view_col(ctx, out_scale_shift, 1);

    ggml_tensor * norm_out = rms_norm(ctx, x, model.norm_out, cfg.rms_norm_eps);
    ggml_tensor * out_scale1 = ggml_add(ctx, out_scale, ones);
    ggml_tensor * out_scale_b = ggml_repeat(ctx, out_scale1, norm_out);
    ggml_tensor * out_shift_b = ggml_repeat(ctx, out_shift, norm_out);
    ggml_tensor * y = ggml_add(ctx, ggml_mul(ctx, norm_out, out_scale_b), out_shift_b);

    ggml_tensor * y_lin = ggml_mul_mat(ctx, model.proj_out_w, y);
    ggml_tensor * y3 = ggml_reshape_3d(ctx, y_lin, cfg.audio_acoustic_hidden_dim, patch, seq_len_p);
    ggml_tensor * y2 = ggml_reshape_2d(ctx, y3, cfg.audio_acoustic_hidden_dim, seq_len_p * patch);
    ggml_tensor * y_bias = add_bias(ctx, y2, model.proj_out_b);

    if (seq_len_p * patch == seq_len) {
        return y_bias;
    }
    return ggml_view_2d(ctx, y_bias, cfg.audio_acoustic_hidden_dim, seq_len, y_bias->nb[1], 0);
}

ggml_tensor * forward_lyric_encoder(
    ggml_context * ctx,
    const Model & model,
    const float * lyric_hidden_states,
    const int32_t * lyric_attention_mask,
    int32_t n_lyric) {

    if (!lyric_hidden_states || n_lyric <= 0) {
        return nullptr;
    }

    const Config & cfg = model.cfg;
    const int32_t in_dim = cfg.text_hidden_dim > 0 ? cfg.text_hidden_dim : 1024;
    const int32_t out_dim = cfg.hidden_size;

    ggml_tensor * proj_w = model.lyric_embed_w ? model.lyric_embed_w : model.text_projector_w;
    ggml_tensor * proj_b = model.lyric_embed_w ? model.lyric_embed_b : nullptr;
    if (!proj_w) {
        return nullptr;
    }
    if (proj_w->ne[0] != in_dim || proj_w->ne[1] != out_dim) {
        return nullptr;
    }
    if (proj_b && proj_b->ne[0] != out_dim) {
        return nullptr;
    }

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, n_lyric);
    if (!x_in || !x_in->data) {
        return nullptr;
    }
    std::memcpy(x_in->data, lyric_hidden_states, static_cast<size_t>(in_dim) * static_cast<size_t>(n_lyric) * sizeof(float));

    ggml_tensor * x = ggml_mul_mat(ctx, proj_w, x_in);
    if (proj_b) {
        x = add_bias(ctx, x, proj_b);
    }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_lyric);
    for (int32_t i = 0; i < n_lyric; ++i) {
        ggml_set_i32_1d(pos, i, i);
    }

    size_t max_layers = model.lyric_layers.size();
    if (const char * env_layers = std::getenv("ACE_GGML_LYRIC_MAX_LAYERS")) {
        char * endp = nullptr;
        const long long val = std::strtoll(env_layers, &endp, 10);
        if (endp && endp != env_layers && val >= 0) {
            max_layers = std::min(max_layers, static_cast<size_t>(val));
        }
    }

    for (size_t i = 0; i < max_layers; ++i) {
        const EncoderLayer & layer = model.lyric_layers[i];

        ggml_tensor * xn = rms_norm(ctx, x, layer.input_norm, cfg.rms_norm_eps);
        ggml_tensor * attn_out = attention(
            ctx,
            cfg,
            layer.self_attn,
            xn,
            xn,
            pos,
            lyric_attention_mask,
            n_lyric,
            n_lyric,
            false,
            layer.use_sliding_window,
            cfg.sliding_window,
            true);
        ggml_tensor * h = ggml_add(ctx, x, attn_out);

        ggml_tensor * hn = rms_norm(ctx, h, layer.post_attn_norm, cfg.rms_norm_eps);
        ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.w_gate, hn);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.mlp.w_up, hn);
        ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * down = ggml_mul_mat(ctx, layer.mlp.w_down, act);
        x = ggml_add(ctx, h, down);
    }

    if (model.lyric_norm) {
        x = rms_norm(ctx, x, model.lyric_norm, cfg.rms_norm_eps);
    }
    return x;
}

ggml_tensor * forward_timbre_encoder(
    ggml_context * ctx,
    const Model & model,
    const float * refer_audio_hidden_states,
    const int32_t * refer_audio_attention_mask,
    int32_t refer_len) {

    if (!refer_audio_hidden_states || refer_len <= 0) {
        return nullptr;
    }

    const Config & cfg = model.cfg;
    const int32_t in_dim =
        cfg.timbre_hidden_dim > 0 ? cfg.timbre_hidden_dim :
        (cfg.audio_acoustic_hidden_dim > 0 ? cfg.audio_acoustic_hidden_dim : 64);
    const int32_t out_dim = cfg.hidden_size;

    ggml_tensor * proj_w = model.timbre_embed_w;
    ggml_tensor * proj_b = model.timbre_embed_b;
    if (!proj_w) {
        return nullptr;
    }
    if (proj_w->ne[0] != in_dim || proj_w->ne[1] != out_dim) {
        return nullptr;
    }
    if (proj_b && proj_b->ne[0] != out_dim) {
        return nullptr;
    }

    ggml_tensor * x_in = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, in_dim, refer_len);
    if (!x_in || !x_in->data) {
        return nullptr;
    }
    std::memcpy(
        x_in->data,
        refer_audio_hidden_states,
        static_cast<size_t>(in_dim) * static_cast<size_t>(refer_len) * sizeof(float));

    ggml_tensor * x = ggml_mul_mat(ctx, proj_w, x_in);
    if (proj_b) {
        x = add_bias(ctx, x, proj_b);
    }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, refer_len);
    for (int32_t i = 0; i < refer_len; ++i) {
        ggml_set_i32_1d(pos, i, i);
    }

    size_t max_layers = model.timbre_layers.size();
    if (const char * env_layers = std::getenv("ACE_GGML_TIMBRE_MAX_LAYERS")) {
        char * endp = nullptr;
        const long long val = std::strtoll(env_layers, &endp, 10);
        if (endp && endp != env_layers && val >= 0) {
            max_layers = std::min(max_layers, static_cast<size_t>(val));
        }
    }

    for (size_t i = 0; i < max_layers; ++i) {
        const EncoderLayer & layer = model.timbre_layers[i];

        ggml_tensor * xn = rms_norm(ctx, x, layer.input_norm, cfg.rms_norm_eps);
        ggml_tensor * attn_out = attention(
            ctx,
            cfg,
            layer.self_attn,
            xn,
            xn,
            pos,
            refer_audio_attention_mask,
            refer_len,
            refer_len,
            false,
            layer.use_sliding_window,
            cfg.sliding_window,
            true);
        ggml_tensor * h = ggml_add(ctx, x, attn_out);

        ggml_tensor * hn = rms_norm(ctx, h, layer.post_attn_norm, cfg.rms_norm_eps);
        ggml_tensor * gate = ggml_mul_mat(ctx, layer.mlp.w_gate, hn);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.mlp.w_up, hn);
        ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * down = ggml_mul_mat(ctx, layer.mlp.w_down, act);
        x = ggml_add(ctx, h, down);
    }

    if (model.timbre_norm) {
        x = rms_norm(ctx, x, model.timbre_norm, cfg.rms_norm_eps);
    }

    // Match Python: use first time step as timbre embedding.
    return ggml_view_2d(ctx, x, out_dim, 1, x->nb[1], 0);
}

}  // namespace ace_dit
