#include "qwen_model.h"

#include "ggml.h"
#include "ggml-cpu.h"
#include "gguf.h"

#include <cmath>
#include <cstring>
#include <sstream>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <algorithm>
#include <cctype>
#include <filesystem>

namespace ace_qwen {

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
    ggml_type t = parse_quant_type(std::getenv("ACE_GGML_QWEN_WEIGHT_QTYPE"));
    if (t != GGML_TYPE_COUNT) {
        return t;
    }
    return parse_quant_type(std::getenv("ACE_GGML_WEIGHT_QTYPE"));
}

static std::string resolve_gguf_path(const std::string & dir) {
    const char * keys[] = {
        "ACE_GGML_QWEN_GGUF",
        "ACE_GGML_TEXT_ENCODER_GGUF",
        "ACE_GGML_LM_GGUF",
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

    const ggml_type qtype = get_quant_type_from_env();
    if (qtype != GGML_TYPE_COUNT && ggml_is_quantized(qtype) && !ggml_quantize_requires_imatrix(qtype)) {
        const int64_t blk = ggml_blck_size(qtype);
        if (blk > 0 && (in_dim % blk) == 0) {
            std::vector<float> f32;
            if (!read_tensor_as_f32(st, *info, f32, error)) {
                return nullptr;
            }
            ggml_tensor * qt = ggml_new_tensor_2d(ctx, qtype, in_dim, out_dim);
            if (!qt || !qt->data) {
                error = "failed to allocate quantized tensor: " + name;
                return nullptr;
            }
            size_t written = ggml_quantize_chunk(qtype, f32.data(), qt->data, 0, out_dim, in_dim, nullptr);
            if (written == 0) {
                error = "failed to quantize tensor: " + name;
                return nullptr;
            }
            ggml_set_name(qt, name.c_str());
            return qt;
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
    const size_t n_tensors = 2 + n_layers * 12; // embeddings + final norm + per-layer

    size_t overhead = ggml_tensor_overhead() * n_tensors;
    size_t ctx_size = weight_bytes + overhead + (16 * 1024 * 1024);

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

    auto release_gguf = [&]() {
        if (gguf) {
            gguf_free(gguf);
            gguf = nullptr;
        }
        if (gguf_ctx) {
            ggml_free(gguf_ctx);
            gguf_ctx = nullptr;
        }
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

    model.tok_embeddings = load_2d("embed_tokens.weight");
    if (!model.tok_embeddings) {
        free_model(model);
        release_gguf();
        return false;
    }
    model.norm = load_1d("norm.weight");
    if (!model.norm) {
        free_model(model);
        release_gguf();
        return false;
    }

    model.layers.resize(n_layers);
    for (size_t i = 0; i < n_layers; ++i) {
        Layer layer;
        std::ostringstream prefix;
        prefix << "layers." << i << ".";
        const std::string p = prefix.str();

        layer.input_norm = load_1d(p + "input_layernorm.weight");
        if (!layer.input_norm) { free_model(model); release_gguf(); return false; }
        layer.post_attn_norm = load_1d(p + "post_attention_layernorm.weight");
        if (!layer.post_attn_norm) { free_model(model); release_gguf(); return false; }

        layer.w_q = load_2d(p + "self_attn.q_proj.weight");
        if (!layer.w_q) { free_model(model); release_gguf(); return false; }
        layer.w_k = load_2d(p + "self_attn.k_proj.weight");
        if (!layer.w_k) { free_model(model); release_gguf(); return false; }
        layer.w_v = load_2d(p + "self_attn.v_proj.weight");
        if (!layer.w_v) { free_model(model); release_gguf(); return false; }
        layer.w_o = load_2d(p + "self_attn.o_proj.weight");
        if (!layer.w_o) { free_model(model); release_gguf(); return false; }
        layer.q_norm = load_1d(p + "self_attn.q_norm.weight");
        if (!layer.q_norm) { free_model(model); release_gguf(); return false; }
        layer.k_norm = load_1d(p + "self_attn.k_norm.weight");
        if (!layer.k_norm) { free_model(model); release_gguf(); return false; }

        layer.w_gate = load_2d(p + "mlp.gate_proj.weight");
        if (!layer.w_gate) { free_model(model); release_gguf(); return false; }
        layer.w_up = load_2d(p + "mlp.up_proj.weight");
        if (!layer.w_up) { free_model(model); release_gguf(); return false; }
        layer.w_down = load_2d(p + "mlp.down_proj.weight");
        if (!layer.w_down) { free_model(model); release_gguf(); return false; }

        model.layers[i] = layer;
    }

    release_gguf();
    model.loaded = true;
    out = std::move(model);
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
    model.tok_embeddings = nullptr;
    model.norm = nullptr;
    model.loaded = false;
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

static ggml_tensor * repeat_kv_interleave(
    ggml_context * ctx,
    ggml_tensor * t,
    int32_t n_rep) {

    if (n_rep <= 1) {
        return t;
    }
    // t shape: [head_dim, n_tokens, n_kv]
    ggml_tensor * t4 = ggml_reshape_4d(ctx, t, t->ne[0], t->ne[1], 1, t->ne[2]);
    ggml_tensor * shape = ggml_new_tensor_4d(ctx, t->type, t->ne[0], t->ne[1], n_rep, t->ne[2]);
    ggml_tensor * rep = ggml_repeat(ctx, t4, shape);
    return ggml_reshape_3d(ctx, rep, t->ne[0], t->ne[1], t->ne[2] * n_rep);
}

ggml_tensor * forward_text_encoder_layers(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    bool causal,
    int32_t n_layers,
    bool apply_final_norm) {

    const Config & cfg = model.cfg;
    const int32_t n_heads = cfg.num_attention_heads;
    const int32_t n_kv = cfg.num_key_value_heads;
    const int32_t head_dim = cfg.head_dim;
    const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    const int32_t total_layers = cfg.num_hidden_layers;
    int32_t run_layers = n_layers;
    if (run_layers < 0) {
        run_layers = total_layers;
    } else if (run_layers > total_layers) {
        run_layers = total_layers;
    }

    ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        ggml_set_i32_1d(tokens, i, token_ids[i]);
    }

    ggml_tensor * pos = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        ggml_set_i32_1d(pos, i, i);
    }

    ggml_tensor * x = ggml_get_rows(ctx, model.tok_embeddings, tokens);
    x = cast_f32(ctx, x);

    for (int32_t i = 0; i < run_layers; ++i) {
        const Layer & layer = model.layers[static_cast<size_t>(i)];

        ggml_tensor * xn = rms_norm(ctx, x, layer.input_norm, cfg.rms_norm_eps);

        ggml_tensor * q = ggml_mul_mat(ctx, layer.w_q, xn);
        ggml_tensor * k = ggml_mul_mat(ctx, layer.w_k, xn);
        ggml_tensor * v = ggml_mul_mat(ctx, layer.w_v, xn);

        q = ggml_cont(ctx, q);
        k = ggml_cont(ctx, k);
        v = ggml_cont(ctx, v);

        // shape for RoPE: [head_dim, n_heads/n_kv, n_tokens]
        q = ggml_reshape_3d(ctx, q, head_dim, n_heads, n_tokens);
        k = ggml_reshape_3d(ctx, k, head_dim, n_kv, n_tokens);
        v = ggml_reshape_3d(ctx, v, head_dim, n_kv, n_tokens);

        q = rms_norm(ctx, q, layer.q_norm, cfg.rms_norm_eps);
        k = rms_norm(ctx, k, layer.k_norm, cfg.rms_norm_eps);

        q = ggml_rope_ext(ctx, q, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, cfg.max_position_embeddings,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);
        k = ggml_rope_ext(ctx, k, pos, nullptr, head_dim, GGML_ROPE_TYPE_NEOX, cfg.max_position_embeddings,
                          cfg.rope_theta, 1.0f, 0.0f, 1.0f, 0.0f, 0.0f);

        // permute to [head_dim, n_tokens, n_heads]
        ggml_tensor * q4 = ggml_reshape_4d(ctx, q, head_dim, n_heads, n_tokens, 1);
        q4 = ggml_permute(ctx, q4, 0, 2, 1, 3);
        q4 = ggml_cont(ctx, q4);
        q = ggml_reshape_3d(ctx, q4, head_dim, n_tokens, n_heads);

        ggml_tensor * k4 = ggml_reshape_4d(ctx, k, head_dim, n_kv, n_tokens, 1);
        k4 = ggml_permute(ctx, k4, 0, 2, 1, 3);
        k4 = ggml_cont(ctx, k4);
        k = ggml_reshape_3d(ctx, k4, head_dim, n_tokens, n_kv);

        ggml_tensor * v4a = ggml_reshape_4d(ctx, v, head_dim, n_kv, n_tokens, 1);
        v4a = ggml_permute(ctx, v4a, 0, 2, 1, 3);
        v4a = ggml_cont(ctx, v4a);
        v = ggml_reshape_3d(ctx, v4a, head_dim, n_tokens, n_kv);

        if (n_kv != n_heads) {
            const int32_t n_rep = n_heads / n_kv;
            k = repeat_kv_interleave(ctx, k, n_rep);
            v = repeat_kv_interleave(ctx, v, n_rep);
        }

        ggml_tensor * mask = nullptr;
        if (causal || attention_mask) {
            const int64_t k_len = n_tokens;
            const int64_t q_len = n_tokens;
            mask = ggml_new_tensor_3d(ctx, GGML_TYPE_F32, k_len, q_len, 1);
            float * mdata = static_cast<float *>(mask->data);
            for (int64_t qidx = 0; qidx < q_len; ++qidx) {
                for (int64_t kidx = 0; kidx < k_len; ++kidx) {
                    bool allow = true;
                    if (causal && kidx > qidx) {
                        allow = false;
                    }
                    if (attention_mask && kidx < n_tokens && attention_mask[kidx] == 0) {
                        allow = false;
                    }
                    float val = allow ? 0.0f : -INFINITY;
                    mdata[kidx + qidx * k_len] = val;
                }
            }
        }

        ggml_tensor * kq = ggml_mul_mat(ctx, k, q);
        kq = ggml_cont(ctx, kq);

        ggml_tensor * attn = ggml_soft_max_ext(ctx, kq, mask, scale, 0.0f);

        ggml_tensor * v4 = ggml_reshape_4d(ctx, v, head_dim, n_tokens, n_heads, 1);
        v4 = ggml_permute(ctx, v4, 1, 0, 2, 3); // [n_tokens, head_dim, n_heads, 1]
        v4 = ggml_cont(ctx, v4);
        ggml_tensor * v3 = ggml_reshape_3d(ctx, v4, n_tokens, head_dim, n_heads);

        ggml_tensor * attn_out = ggml_mul_mat(ctx, v3, attn);

        ggml_tensor * attn4 = ggml_reshape_4d(ctx, attn_out, head_dim, n_tokens, n_heads, 1);
        attn4 = ggml_permute(ctx, attn4, 0, 2, 1, 3); // [head_dim, n_heads, n_tokens, 1]
        attn4 = ggml_cont(ctx, attn4);
        ggml_tensor * attn2 = ggml_reshape_2d(ctx, attn4, head_dim * n_heads, n_tokens);

        ggml_tensor * attn_out_proj = ggml_mul_mat(ctx, layer.w_o, attn2);
        ggml_tensor * h = ggml_add(ctx, x, attn_out_proj);

        ggml_tensor * hn = rms_norm(ctx, h, layer.post_attn_norm, cfg.rms_norm_eps);
        ggml_tensor * gate = ggml_mul_mat(ctx, layer.w_gate, hn);
        ggml_tensor * up = ggml_mul_mat(ctx, layer.w_up, hn);
        ggml_tensor * act = ggml_mul(ctx, ggml_silu(ctx, gate), up);
        ggml_tensor * down = ggml_mul_mat(ctx, layer.w_down, act);

        x = ggml_add(ctx, h, down);
    }

    if (apply_final_norm && run_layers == total_layers) {
        return rms_norm(ctx, x, model.norm, cfg.rms_norm_eps);
    }
    return x;
}

ggml_tensor * forward_text_encoder(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    bool causal) {
    return forward_text_encoder_layers(ctx, model, token_ids, attention_mask, n_tokens, causal,
                                       model.cfg.num_hidden_layers, true);
}

ggml_tensor * forward_text_encoder_embeddings(
    ggml_context * ctx,
    const Model & model,
    const int32_t * token_ids,
    int32_t n_tokens) {

    ggml_tensor * tokens = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_tokens);
    for (int i = 0; i < n_tokens; ++i) {
        ggml_set_i32_1d(tokens, i, token_ids[i]);
    }

    ggml_tensor * x = ggml_get_rows(ctx, model.tok_embeddings, tokens);
    x = cast_f32(ctx, x);
    return x;
}

}  // namespace ace_qwen
