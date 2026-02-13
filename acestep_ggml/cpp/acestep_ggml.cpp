#include "acestep_ggml.h"
#include "qwen_config.h"
#include "safetensors.h"
#include "qwen_model.h"
#include "acestep_dit_model.h"
#include "acestep_vae_model.h"

#include "ggml-cpu.h"
#include "ggml-backend.h"

#include <new>
#include <string>
#include <vector>
#include <cstring>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <filesystem>
#include <random>
#include <cmath>
#include <iterator>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <cctype>

struct ace_ggml_context {
    int32_t n_threads = 0;
    bool use_metal = false;
    size_t compute_buffer_bytes = 0;
    std::string last_error;

    ace_qwen::Model text_encoder;
    ace_dit::Model dit;
    ace_vae::Model vae;
    ggml_backend_t vae_backend = nullptr;
    ggml_backend_t vae_cpu_backend = nullptr;
    ggml_backend_sched_t vae_sched = nullptr;
    ggml_backend_sched_t dit_sched = nullptr;
    ggml_backend_buffer_t dit_model_host_buffer = nullptr;
    ggml_backend_buffer_t vae_model_host_buffer = nullptr;
    bool vae_backend_is_metal = false;
    bool dit_weights_on_metal = false;
    bool vae_weights_on_metal = false;
    bool dit_warned_metal_fallback = false;
    bool vae_warned_metal_fallback = false;

    // Reusable buffers to avoid large per-call allocations in chunked decode paths.
    std::vector<uint8_t> vae_compute_buffer;
    std::vector<uint8_t> vae_graph_buffer;

    // Cached decode graph for repeated chunk sizes (common in tiled VAE decode).
    ggml_context * vae_decode_compute_ctx = nullptr;
    ggml_context * vae_decode_graph_ctx = nullptr;
    ggml_cgraph * vae_decode_graph = nullptr;
    ggml_tensor * vae_decode_input = nullptr;
    ggml_tensor * vae_decode_output = nullptr;
    std::vector<float> vae_decode_input_planar;
    std::vector<float> vae_decode_backend_output;
    int32_t vae_decode_cached_frames = -1;
    size_t vae_decode_cached_graph_size = 0;
    bool vae_decode_graph_allocated = false;
};

static ace_ggml_status ace_set_error(ace_ggml_context * ctx, ace_ggml_status code, const char * message) {
    if (ctx && message) {
        ctx->last_error = message;
    }
    return code;
}

static size_t ace_read_graph_size_env(size_t fallback, const char * key, const char * fallback_key);

static void ace_assign_buffer_to_ctx_tensors(ggml_context * ggctx, ggml_backend_buffer_t buffer) {
    if (!ggctx || !buffer) {
        return;
    }
    for (ggml_tensor * t = ggml_get_first_tensor(ggctx); t != nullptr; t = ggml_get_next_tensor(ggctx, t)) {
        if (t->buffer == nullptr && t->data != nullptr) {
            t->buffer = buffer;
        }
        if (t->view_src && t->buffer == nullptr && t->view_src->buffer) {
            t->buffer = t->view_src->buffer;
        }
    }
}

static void ace_reset_vae_decode_cache(ace_ggml_context * ctx) {
    if (!ctx) {
        return;
    }
    if (ctx->vae_decode_graph_ctx) {
        ggml_free(ctx->vae_decode_graph_ctx);
        ctx->vae_decode_graph_ctx = nullptr;
    }
    if (ctx->vae_decode_compute_ctx) {
        ggml_free(ctx->vae_decode_compute_ctx);
        ctx->vae_decode_compute_ctx = nullptr;
    }
    ctx->vae_decode_graph = nullptr;
    ctx->vae_decode_input = nullptr;
    ctx->vae_decode_output = nullptr;
    ctx->vae_decode_cached_frames = -1;
    ctx->vae_decode_cached_graph_size = 0;
    ctx->vae_decode_graph_allocated = false;
}

ace_ggml_status ace_ggml_create(const ace_ggml_init_params * params, ace_ggml_context ** out_ctx) {
    if (!out_ctx) {
        return ACE_GGML_ERR_INVALID_ARG;
    }

    ace_ggml_context * ctx = new (std::nothrow) ace_ggml_context();
    if (!ctx) {
        return ACE_GGML_ERR;
    }

    if (params) {
        ctx->n_threads = params->n_threads;
        ctx->use_metal = params->use_metal != 0;
        ctx->compute_buffer_bytes = params->compute_buffer_bytes;
    }

    if (ctx->compute_buffer_bytes == 0) {
        ctx->compute_buffer_bytes = 512ULL * 1024ULL * 1024ULL;
    }

    if (ctx->use_metal) {
        // Uses ggml backend registry; returns nullptr if Metal backend is not built/available.
        ctx->vae_backend = ggml_backend_init_by_name("Metal", nullptr);
        if (ctx->vae_backend != nullptr) {
            // CPU fallback backend used by scheduler for unsupported ops.
            ctx->vae_cpu_backend = ggml_backend_cpu_init();
            if (ctx->vae_cpu_backend && ctx->n_threads > 0) {
                ggml_backend_cpu_set_n_threads(ctx->vae_cpu_backend, ctx->n_threads);
            }
            if (ctx->vae_cpu_backend) {
                ggml_backend_t backends[] = {ctx->vae_backend, ctx->vae_cpu_backend};
                ctx->vae_sched = ggml_backend_sched_new(
                    backends,
                    nullptr,
                    2,
                    GGML_DEFAULT_GRAPH_SIZE,
                    false,
                    true);
                if (ctx->vae_sched) {
                    ctx->vae_backend_is_metal = true;
                    size_t dit_sched_graph_size = ace_read_graph_size_env(
                        65536,
                        "ACE_GGML_DIT_SCHED_GRAPH_SIZE",
                        "ACE_GGML_DIT_GRAPH_SIZE");
                    if (dit_sched_graph_size < 4096) {
                        dit_sched_graph_size = 4096;
                    }
                    if (dit_sched_graph_size > 1048576) {
                        dit_sched_graph_size = 1048576;
                    }
                    ctx->dit_sched = ggml_backend_sched_new(
                        backends,
                        nullptr,
                        2,
                        dit_sched_graph_size,
                        false,
                        true);
                    if (!ctx->dit_sched) {
                        std::fprintf(
                            stderr,
                            "ace_ggml: failed to create DiT metal scheduler, falling back to CPU DiT compute path\n");
                    } else if (std::getenv("ACE_GGML_DEBUG_GRAPH")) {
                        std::fprintf(stderr, "ace_ggml: DiT metal scheduler graph_size=%zu\n", dit_sched_graph_size);
                    }
                } else {
                    ctx->last_error = "Metal backend initialized but scheduler creation failed; falling back to CPU graph compute path";
                    ggml_backend_free(ctx->vae_cpu_backend);
                    ctx->vae_cpu_backend = nullptr;
                    ggml_backend_free(ctx->vae_backend);
                    ctx->vae_backend = nullptr;
                    ctx->use_metal = false;
                }
            } else {
                ctx->last_error = "Metal backend initialized but CPU fallback backend failed; falling back to CPU graph compute path";
                ggml_backend_free(ctx->vae_backend);
                ctx->vae_backend = nullptr;
                ctx->use_metal = false;
            }
        } else {
            ctx->last_error = "Metal backend unavailable, falling back to CPU graph compute path";
            ctx->use_metal = false;
        }
    }

    *out_ctx = ctx;
    return ACE_GGML_OK;
}

void ace_ggml_destroy(ace_ggml_context * ctx) {
    if (!ctx) {
        return;
    }
    ace_reset_vae_decode_cache(ctx);
    ace_qwen::free_model(ctx->text_encoder);
    if (ctx->dit_model_host_buffer) {
        ggml_backend_buffer_free(ctx->dit_model_host_buffer);
        ctx->dit_model_host_buffer = nullptr;
    }
    ace_dit::free_model(ctx->dit);
    if (ctx->vae_model_host_buffer) {
        ggml_backend_buffer_free(ctx->vae_model_host_buffer);
        ctx->vae_model_host_buffer = nullptr;
    }
    ace_vae::free_model(ctx->vae);
    if (ctx->vae_sched) {
        ggml_backend_sched_free(ctx->vae_sched);
        ctx->vae_sched = nullptr;
    }
    if (ctx->dit_sched) {
        ggml_backend_sched_free(ctx->dit_sched);
        ctx->dit_sched = nullptr;
    }
    if (ctx->vae_cpu_backend) {
        ggml_backend_free(ctx->vae_cpu_backend);
        ctx->vae_cpu_backend = nullptr;
    }
    if (ctx->vae_backend) {
        ggml_backend_free(ctx->vae_backend);
        ctx->vae_backend = nullptr;
    }
    delete ctx;
}

const char * ace_ggml_last_error(const ace_ggml_context * ctx) {
    if (!ctx) {
        return "ace_ggml_last_error: null context";
    }
    return ctx->last_error.c_str();
}

static ace_ggml_status load_qwen_dir(ace_ggml_context * ctx, const char * model_dir, ace_qwen::Model & model) {
    std::string error;
    if (!ace_qwen::load_model_from_dir(model_dir, model, error)) {
        return ace_set_error(ctx, ACE_GGML_ERR_IO, error.c_str());
    }
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_load_lm(ace_ggml_context * ctx, const char * model_dir) {
    if (!ctx || !model_dir) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    return load_qwen_dir(ctx, model_dir, ctx->text_encoder);
}

ace_ggml_status ace_ggml_load_text_encoder(ace_ggml_context * ctx, const char * model_dir) {
    if (!ctx || !model_dir) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    return load_qwen_dir(ctx, model_dir, ctx->text_encoder);
}

ace_ggml_status ace_ggml_load_dit(ace_ggml_context * ctx, const char * model_dir) {
    if (!ctx || !model_dir) {
        return ACE_GGML_ERR_INVALID_ARG;
    }

    if (ctx->dit_sched) {
        ggml_backend_sched_reset(ctx->dit_sched);
    }
    if (ctx->dit_model_host_buffer) {
        ggml_backend_buffer_free(ctx->dit_model_host_buffer);
        ctx->dit_model_host_buffer = nullptr;
    }
    ctx->dit_weights_on_metal = false;
    ctx->dit_warned_metal_fallback = false;
    ace_dit::free_model(ctx->dit);

    std::string error;
    if (!ace_dit::load_model_from_dir(model_dir, ctx->dit, error)) {
        return ace_set_error(ctx, ACE_GGML_ERR_IO, error.c_str());
    }

    if (!ctx->dit.ctx_buffer || ctx->dit.ctx_buffer_size == 0 || !ctx->dit.ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "DiT model loaded but context buffer is invalid");
    }

    bool map_force_on = false;
    bool map_disabled = false;
    if (const char * mode_value = std::getenv("ACE_GGML_DIT_METAL_WEIGHT_MAP")) {
        std::string normalized(mode_value);
        std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
            return static_cast<char>(std::tolower(ch));
        });
        if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no" || normalized == "disable") {
            map_disabled = true;
        } else if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes" || normalized == "force") {
            map_force_on = true;
        }
    }

    if (ctx->vae_backend_is_metal && ctx->vae_backend && !map_disabled) {
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vae_backend);
        bool can_try_map = dev != nullptr;
        ggml_backend_dev_props props{};
        bool have_props = false;
        if (dev) {
            ggml_backend_dev_get_props(dev, &props);
            have_props = true;
        }

        if (can_try_map && !map_force_on && have_props) {
            if (!props.caps.buffer_from_host_ptr) {
                can_try_map = false;
            } else {
                const size_t min_free_mb = ace_read_graph_size_env(4096, "ACE_GGML_DIT_METAL_MIN_FREE_MB", nullptr);
                const size_t min_free_bytes = min_free_mb * 1024ULL * 1024ULL;
                if (props.memory_free > 0) {
                    bool enough_free = props.memory_free > ctx->dit.ctx_buffer_size;
                    if (enough_free) {
                        const size_t free_after_map = props.memory_free - ctx->dit.ctx_buffer_size;
                        enough_free = free_after_map >= min_free_bytes;
                    }
                    if (!enough_free) {
                        can_try_map = false;
                        std::fprintf(
                            stderr,
                            "ace_ggml: skip metal host_ptr map for DiT (auto): free=%.2f MiB model=%.2f MiB min_free=%.2f MiB\n",
                            static_cast<double>(props.memory_free) / 1024.0 / 1024.0,
                            static_cast<double>(ctx->dit.ctx_buffer_size) / 1024.0 / 1024.0,
                            static_cast<double>(min_free_bytes) / 1024.0 / 1024.0);
                    }
                }
            }
        }

        if (can_try_map && dev) {
            ctx->dit_model_host_buffer = ggml_backend_dev_buffer_from_host_ptr(
                dev,
                ctx->dit.ctx_buffer,
                ctx->dit.ctx_buffer_size,
                ctx->dit.ctx_buffer_size);
            ctx->dit_weights_on_metal = (ctx->dit_model_host_buffer != nullptr);
        }
        if (!ctx->dit_model_host_buffer && map_force_on) {
            std::fprintf(stderr, "ace_ggml: DiT metal host_ptr map requested but unavailable, falling back to CPU model buffer\n");
        }
    }

    if (!ctx->dit_model_host_buffer) {
        ctx->dit_model_host_buffer = ggml_backend_cpu_buffer_from_ptr(ctx->dit.ctx_buffer, ctx->dit.ctx_buffer_size);
    }
    if (!ctx->dit_model_host_buffer) {
        return ace_set_error(ctx, ACE_GGML_ERR, "failed to create host buffer for DiT model context");
    }

    ggml_backend_buffer_set_usage(ctx->dit_model_host_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ace_assign_buffer_to_ctx_tensors(ctx->dit.ctx, ctx->dit_model_host_buffer);
    return ACE_GGML_OK;
}

static size_t ace_read_graph_size_env(size_t fallback, const char * key, const char * fallback_key) {
    auto parse = [](const char * value, size_t * out) -> bool {
        if (!value) {
            return false;
        }
        char * endp = nullptr;
        const long long v = std::strtoll(value, &endp, 10);
        if (endp && endp != value && v > 0) {
            *out = static_cast<size_t>(v);
            return true;
        }
        return false;
    };

    size_t result = fallback;
    if (!parse(std::getenv(key), &result) && fallback_key) {
        parse(std::getenv(fallback_key), &result);
    }
    return result;
}

static bool ace_env_enabled(const char * key) {
    const char * value = std::getenv(key);
    if (!value || !*value) {
        return false;
    }
    return std::strcmp(value, "0") != 0;
}

enum class ace_toggle_mode {
    AUTO,
    ON,
    OFF,
};

static ace_toggle_mode ace_read_toggle_mode_env(const char * key, ace_toggle_mode fallback) {
    const char * value = std::getenv(key);
    if (!value || !*value) {
        return fallback;
    }
    std::string normalized(value);
    std::transform(normalized.begin(), normalized.end(), normalized.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    if (normalized == "1" || normalized == "true" || normalized == "on" || normalized == "yes" || normalized == "force") {
        return ace_toggle_mode::ON;
    }
    if (normalized == "0" || normalized == "false" || normalized == "off" || normalized == "no" || normalized == "disable") {
        return ace_toggle_mode::OFF;
    }
    if (normalized == "auto" || normalized == "default") {
        return ace_toggle_mode::AUTO;
    }
    return fallback;
}

static bool ace_metal_decode_requires_weight_map() {
    return ace_read_toggle_mode_env("ACE_GGML_VAE_METAL_REQUIRE_WEIGHT_MAP", ace_toggle_mode::OFF) == ace_toggle_mode::ON;
}

static bool ace_allow_unsafe_vae_metal_decode() {
    return ace_read_toggle_mode_env("ACE_GGML_ALLOW_UNSAFE_VAE_METAL", ace_toggle_mode::OFF) == ace_toggle_mode::ON;
}

static bool ace_should_use_metal_decode(ace_ggml_context * ctx) {
    if (!ctx || !ctx->vae_backend_is_metal || !ctx->vae_sched || !ctx->vae_backend || !ctx->vae_model_host_buffer) {
        return false;
    }
    if (!ace_allow_unsafe_vae_metal_decode()) {
        static bool warned_once = false;
        const ace_toggle_mode requested = ace_read_toggle_mode_env("ACE_GGML_VAE_METAL_DECODE", ace_toggle_mode::OFF);
        if (requested == ace_toggle_mode::ON && !warned_once) {
            std::fprintf(
                stderr,
                "ace_ggml: ignore ACE_GGML_VAE_METAL_DECODE=on; set ACE_GGML_ALLOW_UNSAFE_VAE_METAL=1 to enable unsafe VAE metal decode\n");
            warned_once = true;
        }
        return false;
    }
    // Stable-diffusion.cpp-style behavior: allow runtime backend execution even
    // when model params stay on CPU buffers. Keep opt-in strict mode for regressions.
    if (ace_metal_decode_requires_weight_map() && !ctx->vae_weights_on_metal) {
        return false;
    }

    const ace_toggle_mode mode = ace_read_toggle_mode_env("ACE_GGML_VAE_METAL_DECODE", ace_toggle_mode::OFF);
    if (mode == ace_toggle_mode::OFF) {
        return false;
    }
    if (mode == ace_toggle_mode::ON) {
        return true;
    }

    const size_t min_free_mb = ace_read_graph_size_env(
        2048,
        "ACE_GGML_VAE_METAL_DECODE_MIN_FREE_MB",
        "ACE_GGML_VAE_METAL_MIN_FREE_MB");
    if (min_free_mb == 0) {
        return true;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vae_backend);
    if (!dev) {
        return false;
    }

    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    const size_t min_free_bytes = min_free_mb * 1024ULL * 1024ULL;
    if (props.memory_free > 0 && props.memory_free < min_free_bytes) {
        if (!ctx->vae_warned_metal_fallback) {
            std::fprintf(
                stderr,
                "ace_ggml: disable metal vae decode (auto) due to low free memory: free=%.2f MiB < min=%.2f MiB\n",
                static_cast<double>(props.memory_free) / 1024.0 / 1024.0,
                static_cast<double>(min_free_bytes) / 1024.0 / 1024.0);
            ctx->vae_warned_metal_fallback = true;
        }
        return false;
    }

    return true;
}

static bool ace_should_use_metal_dit(ace_ggml_context * ctx) {
    // Keep DiT backend gating strict for stability: scheduler path currently
    // assumes metal-mapped model params for robust buffer assignment.
    if (!ctx || !ctx->vae_backend_is_metal || !ctx->dit_sched || !ctx->vae_backend || !ctx->dit_weights_on_metal) {
        return false;
    }

    const ace_toggle_mode mode = ace_read_toggle_mode_env("ACE_GGML_DIT_METAL_FORWARD", ace_toggle_mode::AUTO);
    if (mode == ace_toggle_mode::OFF) {
        return false;
    }
    if (mode == ace_toggle_mode::ON) {
        return true;
    }

    const size_t min_free_mb = ace_read_graph_size_env(
        2048,
        "ACE_GGML_DIT_METAL_FORWARD_MIN_FREE_MB",
        "ACE_GGML_DIT_METAL_MIN_FREE_MB");
    if (min_free_mb == 0) {
        return true;
    }

    ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vae_backend);
    if (!dev) {
        return false;
    }

    ggml_backend_dev_props props{};
    ggml_backend_dev_get_props(dev, &props);
    const size_t min_free_bytes = min_free_mb * 1024ULL * 1024ULL;
    if (props.memory_free > 0 && props.memory_free < min_free_bytes) {
        if (!ctx->dit_warned_metal_fallback) {
            std::fprintf(
                stderr,
                "ace_ggml: disable metal dit forward (auto) due to low free memory: free=%.2f MiB < min=%.2f MiB\n",
                static_cast<double>(props.memory_free) / 1024.0 / 1024.0,
                static_cast<double>(min_free_bytes) / 1024.0 / 1024.0);
            ctx->dit_warned_metal_fallback = true;
        }
        return false;
    }

    return true;
}

ace_ggml_status ace_ggml_load_vae(ace_ggml_context * ctx, const char * model_dir) {
    if (!ctx || !model_dir) {
        return ACE_GGML_ERR_INVALID_ARG;
    }

    auto has_vae_config = [](const std::string & dir) {
        const std::filesystem::path p(dir);
        const std::filesystem::path cfg_path = p / "config.json";
        if (!std::filesystem::exists(cfg_path)) {
            return false;
        }
        std::ifstream ifs(cfg_path);
        if (!ifs) {
            return false;
        }
        std::string cfg_text((std::istreambuf_iterator<char>(ifs)), std::istreambuf_iterator<char>());
        return cfg_text.find("\"audio_channels\"") != std::string::npos &&
               cfg_text.find("\"decoder_input_channels\"") != std::string::npos;
    };

    std::string vae_dir = model_dir;
    if (!has_vae_config(vae_dir)) {
        const std::string nested = vae_dir + "/vae";
        if (has_vae_config(nested)) {
            vae_dir = nested;
        }
    }

    std::string error;
    ace_reset_vae_decode_cache(ctx);
    if (ctx->vae_sched) {
        ggml_backend_sched_reset(ctx->vae_sched);
    }
    if (ctx->vae_model_host_buffer) {
        ggml_backend_buffer_free(ctx->vae_model_host_buffer);
        ctx->vae_model_host_buffer = nullptr;
    }
    ctx->vae_weights_on_metal = false;
    ctx->vae_warned_metal_fallback = false;
    ace_vae::free_model(ctx->vae);
    if (!ace_vae::load_model_from_dir(vae_dir, ctx->vae, error)) {
        return ace_set_error(ctx, ACE_GGML_ERR_IO, error.c_str());
    }
    if (!ctx->vae.ctx_buffer || ctx->vae.ctx_buffer_size == 0 || !ctx->vae.ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "VAE model loaded but context buffer is invalid");
    }
    const ace_toggle_mode map_mode = ace_read_toggle_mode_env("ACE_GGML_VAE_METAL_WEIGHT_MAP", ace_toggle_mode::AUTO);
    if (ctx->vae_backend_is_metal && ctx->vae_backend && map_mode != ace_toggle_mode::OFF) {
        ggml_backend_dev_t dev = ggml_backend_get_device(ctx->vae_backend);
        bool can_try_map = dev != nullptr;
        ggml_backend_dev_props props{};
        bool have_props = false;
        if (dev) {
            ggml_backend_dev_get_props(dev, &props);
            have_props = true;
        }

        if (can_try_map && map_mode == ace_toggle_mode::AUTO && have_props) {
            if (!props.caps.buffer_from_host_ptr) {
                can_try_map = false;
            } else {
                const size_t min_free_mb = ace_read_graph_size_env(4096, "ACE_GGML_VAE_METAL_MIN_FREE_MB", nullptr);
                const size_t min_free_bytes = min_free_mb * 1024ULL * 1024ULL;
                if (props.memory_free > 0) {
                    bool enough_free = props.memory_free > ctx->vae.ctx_buffer_size;
                    if (enough_free) {
                        const size_t free_after_map = props.memory_free - ctx->vae.ctx_buffer_size;
                        enough_free = free_after_map >= min_free_bytes;
                    }
                    if (!enough_free) {
                        can_try_map = false;
                        std::fprintf(
                            stderr,
                            "ace_ggml: skip metal host_ptr map (auto): free=%.2f MiB model=%.2f MiB min_free=%.2f MiB\n",
                            static_cast<double>(props.memory_free) / 1024.0 / 1024.0,
                            static_cast<double>(ctx->vae.ctx_buffer_size) / 1024.0 / 1024.0,
                            static_cast<double>(min_free_bytes) / 1024.0 / 1024.0);
                    }
                }
            }
        }

        if (can_try_map && dev) {
            ctx->vae_model_host_buffer = ggml_backend_dev_buffer_from_host_ptr(
                dev,
                ctx->vae.ctx_buffer,
                ctx->vae.ctx_buffer_size,
                ctx->vae.ctx_buffer_size);
            ctx->vae_weights_on_metal = (ctx->vae_model_host_buffer != nullptr);
        }
        if (!ctx->vae_model_host_buffer && map_mode == ace_toggle_mode::ON) {
            std::fprintf(stderr, "ace_ggml: metal host_ptr map requested but unavailable, falling back to CPU model buffer\n");
        }
    }
    if (!ctx->vae_model_host_buffer) {
        ctx->vae_model_host_buffer = ggml_backend_cpu_buffer_from_ptr(ctx->vae.ctx_buffer, ctx->vae.ctx_buffer_size);
    }
    if (!ctx->vae_model_host_buffer) {
        return ace_set_error(ctx, ACE_GGML_ERR, "failed to create host buffer for VAE model context");
    }
    ggml_backend_buffer_set_usage(ctx->vae_model_host_buffer, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    ace_assign_buffer_to_ctx_tensors(ctx->vae.ctx, ctx->vae_model_host_buffer);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_vae_get_info(
    ace_ggml_context * ctx,
    int32_t * latent_channels,
    int32_t * audio_channels,
    int32_t * hop_length) {

    if (!ctx) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    if (latent_channels) {
        *latent_channels = ctx->vae.cfg.decoder_input_channels;
    }
    if (audio_channels) {
        *audio_channels = ctx->vae.cfg.audio_channels;
    }
    if (hop_length) {
        *hop_length = ctx->vae.cfg.hop_length;
    }
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_vae_decode(
    ace_ggml_context * ctx,
    const float * latents,
    int32_t n_frames,
    float * out,
    size_t out_size) {

    if (!ctx || !latents || !out || n_frames <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    const int32_t latent_channels = ctx->vae.cfg.decoder_input_channels;
    const int32_t audio_channels = ctx->vae.cfg.audio_channels;
    const int32_t hop_length = ctx->vae.cfg.hop_length;
    const size_t expected = static_cast<size_t>(n_frames) * static_cast<size_t>(hop_length) *
                            static_cast<size_t>(audio_channels) * sizeof(float);
    if (out_size < expected) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    const bool use_backend = ace_should_use_metal_decode(ctx);
    const bool profile = ace_env_enabled("ACE_GGML_VAE_PROFILE");
    const auto t0 = std::chrono::steady_clock::now();
    ggml_context * compute_ctx = nullptr;
    ggml_context * graph_ctx = nullptr;
    ggml_tensor * out_f32 = nullptr;
    ggml_cgraph * graph = nullptr;
    ggml_status status = GGML_STATUS_FAILED;
    bool free_temp_contexts = false;
    size_t graph_size = 262144;
    graph_size = ace_read_graph_size_env(graph_size, "ACE_GGML_VAE_GRAPH_SIZE", "ACE_GGML_GRAPH_SIZE");
    auto t_compute_init = t0;
    auto t_graph_build = t0;
    auto t_graph_ready = t0;
    auto t_graph_done = t0;

    if (use_backend) {
        const bool rebuild_cache =
            ctx->vae_decode_compute_ctx == nullptr ||
            ctx->vae_decode_graph_ctx == nullptr ||
            ctx->vae_decode_graph == nullptr ||
            ctx->vae_decode_input == nullptr ||
            ctx->vae_decode_output == nullptr ||
            ctx->vae_decode_cached_frames != n_frames ||
            ctx->vae_decode_cached_graph_size != graph_size;

        if (rebuild_cache) {
            ace_reset_vae_decode_cache(ctx);
            ggml_backend_sched_reset(ctx->vae_sched);

            size_t meta_mb = ace_read_graph_size_env(256, "ACE_GGML_VAE_META_BUFFER_MB", nullptr);
            if (meta_mb < 64) {
                meta_mb = 64;
            }
            const size_t meta_bytes = meta_mb * 1024ULL * 1024ULL;
            if (ctx->vae_compute_buffer.size() != meta_bytes) {
                ctx->vae_compute_buffer.resize(meta_bytes);
            }

            ggml_init_params params{};
            params.mem_size = ctx->vae_compute_buffer.size();
            params.mem_buffer = ctx->vae_compute_buffer.data();
            params.no_alloc = true;

            ctx->vae_decode_compute_ctx = ggml_init(params);
            if (!ctx->vae_decode_compute_ctx) {
                return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
            }
            t_compute_init = std::chrono::steady_clock::now();

            ggml_tensor * latent_in = nullptr;
            ggml_tensor * out_tensor = ace_vae::forward_decode(
                ctx->vae_decode_compute_ctx,
                ctx->vae,
                nullptr,
                n_frames,
                &latent_in);
            if (!out_tensor || !latent_in) {
                ace_reset_vae_decode_cache(ctx);
                return ace_set_error(ctx, ACE_GGML_ERR, "vae decode graph build failed");
            }

            ctx->vae_decode_input = latent_in;
            ctx->vae_decode_output = out_tensor->type == GGML_TYPE_F32 ? out_tensor : ggml_cast(ctx->vae_decode_compute_ctx, out_tensor, GGML_TYPE_F32);
            if (!ctx->vae_decode_output) {
                ace_reset_vae_decode_cache(ctx);
                return ace_set_error(ctx, ACE_GGML_ERR, "vae decode cast failed");
            }
            ggml_set_output(ctx->vae_decode_output);
            t_graph_build = std::chrono::steady_clock::now();

            const size_t graph_buffer_size = ggml_graph_overhead_custom(graph_size, false);
            if (ctx->vae_graph_buffer.size() != graph_buffer_size) {
                ctx->vae_graph_buffer.resize(graph_buffer_size);
            }
            ggml_init_params gparams{};
            gparams.mem_size = ctx->vae_graph_buffer.size();
            gparams.mem_buffer = ctx->vae_graph_buffer.data();
            gparams.no_alloc = true;
            ctx->vae_decode_graph_ctx = ggml_init(gparams);
            if (!ctx->vae_decode_graph_ctx) {
                ace_reset_vae_decode_cache(ctx);
                return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init graph failed");
            }

            ctx->vae_decode_graph = ggml_new_graph_custom(ctx->vae_decode_graph_ctx, graph_size, false);
            if (!ctx->vae_decode_graph) {
                ace_reset_vae_decode_cache(ctx);
                return ace_set_error(ctx, ACE_GGML_ERR, "ggml_new_graph_custom failed");
            }
            ggml_build_forward_expand(ctx->vae_decode_graph, ctx->vae_decode_output);
            ggml_backend_sched_set_tensor_backend(ctx->vae_sched, ctx->vae_decode_output, ctx->vae_backend);
            t_graph_ready = std::chrono::steady_clock::now();

            if (!ggml_backend_sched_alloc_graph(ctx->vae_sched, ctx->vae_decode_graph)) {
                ace_reset_vae_decode_cache(ctx);
                return ace_set_error(ctx, ACE_GGML_ERR, "ggml_backend_sched_alloc_graph failed");
            }
            ctx->vae_decode_graph_allocated = true;
            ctx->vae_decode_cached_frames = n_frames;
            ctx->vae_decode_cached_graph_size = graph_size;
        } else {
            t_compute_init = std::chrono::steady_clock::now();
            t_graph_build = t_compute_init;
            t_graph_ready = t_compute_init;
        }

        compute_ctx = ctx->vae_decode_compute_ctx;
        graph_ctx = ctx->vae_decode_graph_ctx;
        graph = ctx->vae_decode_graph;
        out_f32 = ctx->vae_decode_output;

        const size_t planar_count = static_cast<size_t>(n_frames) * static_cast<size_t>(latent_channels);
        if (ctx->vae_decode_input_planar.size() != planar_count) {
            ctx->vae_decode_input_planar.resize(planar_count);
        }
        float * latents_planar = ctx->vae_decode_input_planar.data();
        for (int32_t t = 0; t < n_frames; ++t) {
            for (int32_t c = 0; c < latent_channels; ++c) {
                latents_planar[static_cast<size_t>(t) + static_cast<size_t>(c) * static_cast<size_t>(n_frames)] =
                    latents[static_cast<size_t>(t) * static_cast<size_t>(latent_channels) + static_cast<size_t>(c)];
            }
        }
        ggml_backend_tensor_set(ctx->vae_decode_input, latents_planar, 0, planar_count * sizeof(float));
        status = ggml_backend_sched_graph_compute(ctx->vae_sched, graph);
        if (status == GGML_STATUS_SUCCESS) {
            const int32_t copy_out_len = static_cast<int32_t>(out_f32->ne[0]);
            const int32_t copy_out_ch = static_cast<int32_t>(out_f32->ne[1]);
            const size_t copy_needed = static_cast<size_t>(copy_out_len) * static_cast<size_t>(copy_out_ch) * sizeof(float);
            if (ctx->vae_decode_backend_output.size() != static_cast<size_t>(copy_out_len) * static_cast<size_t>(copy_out_ch)) {
                ctx->vae_decode_backend_output.resize(static_cast<size_t>(copy_out_len) * static_cast<size_t>(copy_out_ch));
            }
            ggml_backend_tensor_get(out_f32, ctx->vae_decode_backend_output.data(), 0, copy_needed);
        }
    } else {
        if (ctx->vae_compute_buffer.size() != ctx->compute_buffer_bytes) {
            ctx->vae_compute_buffer.resize(ctx->compute_buffer_bytes);
        }
        ggml_init_params params{};
        params.mem_size = ctx->vae_compute_buffer.size();
        params.mem_buffer = ctx->vae_compute_buffer.data();
        params.no_alloc = false;

        compute_ctx = ggml_init(params);
        if (!compute_ctx) {
            return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
        }
        free_temp_contexts = true;
        t_compute_init = std::chrono::steady_clock::now();

        ggml_tensor * out_tensor = ace_vae::forward_decode(compute_ctx, ctx->vae, latents, n_frames);
        if (!out_tensor) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "vae decode failed");
        }
        out_f32 = out_tensor->type == GGML_TYPE_F32 ? out_tensor : ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
        t_graph_build = std::chrono::steady_clock::now();

        const size_t graph_buffer_size = ggml_graph_overhead_custom(graph_size, false);
        if (ctx->vae_graph_buffer.size() != graph_buffer_size) {
            ctx->vae_graph_buffer.resize(graph_buffer_size);
        }
        ggml_init_params gparams{};
        gparams.mem_size = ctx->vae_graph_buffer.size();
        gparams.mem_buffer = ctx->vae_graph_buffer.data();
        gparams.no_alloc = true;
        graph_ctx = ggml_init(gparams);
        if (!graph_ctx) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init graph failed");
        }

        graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
        if (!graph) {
            ggml_free(graph_ctx);
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "ggml_new_graph_custom failed");
        }
        ggml_build_forward_expand(graph, out_f32);
        t_graph_ready = std::chrono::steady_clock::now();

        status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    }

    if (status != GGML_STATUS_SUCCESS) {
        if (free_temp_contexts) {
            if (graph_ctx) {
                ggml_free(graph_ctx);
            }
            if (compute_ctx) {
                ggml_free(compute_ctx);
            }
        }
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }
    t_graph_done = std::chrono::steady_clock::now();

    if (!use_backend && (!out_f32 || !out_f32->data)) {
        if (free_temp_contexts) {
            if (graph_ctx) {
                ggml_free(graph_ctx);
            }
            if (compute_ctx) {
                ggml_free(compute_ctx);
            }
        }
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }

    if (!out_f32) {
        if (free_temp_contexts) {
            if (graph_ctx) {
                ggml_free(graph_ctx);
            }
            if (compute_ctx) {
                ggml_free(compute_ctx);
            }
        }
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor is null");
    }

    const int32_t out_len = static_cast<int32_t>(out_f32->ne[0]);
    const int32_t out_ch = static_cast<int32_t>(out_f32->ne[1]);
    if (out_ch != audio_channels || out_len <= 0) {
        if (free_temp_contexts) {
            if (graph_ctx) {
                ggml_free(graph_ctx);
            }
            if (compute_ctx) {
                ggml_free(compute_ctx);
            }
        }
        return ace_set_error(ctx, ACE_GGML_ERR, "unexpected vae output shape");
    }
    const size_t needed = static_cast<size_t>(out_len) * static_cast<size_t>(out_ch) * sizeof(float);
    if (out_size < needed) {
        if (free_temp_contexts) {
            if (graph_ctx) {
                ggml_free(graph_ctx);
            }
            if (compute_ctx) {
                ggml_free(compute_ctx);
            }
        }
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small for actual output");
    }

    const float * src = static_cast<const float *>(out_f32->data);
    if (use_backend) {
        if (ctx->vae_decode_backend_output.empty()) {
            return ace_set_error(ctx, ACE_GGML_ERR, "backend output copy is empty");
        }
        src = ctx->vae_decode_backend_output.data();
    }
    for (int32_t t = 0; t < out_len; ++t) {
        for (int32_t c = 0; c < out_ch; ++c) {
            out[static_cast<size_t>(t) * out_ch + c] = src[static_cast<size_t>(t) + static_cast<size_t>(c) * out_len];
        }
    }
    const auto t_copy_done = std::chrono::steady_clock::now();

    if (profile) {
        const auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_done - t0).count();
        const auto ms_init = std::chrono::duration_cast<std::chrono::milliseconds>(t_compute_init - t0).count();
        const auto ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_build - t_compute_init).count();
        const auto ms_graph = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_ready - t_graph_build).count();
        const auto ms_compute = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_done - t_graph_ready).count();
        const auto ms_copy = std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_done - t_graph_done).count();
        std::fprintf(
            stderr,
            "ace_ggml_vae_decode profile: backend=%s frames=%d out_len=%d threads=%d init_ms=%lld build_ms=%lld graph_ms=%lld compute_ms=%lld copy_ms=%lld total_ms=%lld\n",
            use_backend ? "metal" : "cpu",
            n_frames,
            out_len,
            ctx->n_threads > 0 ? ctx->n_threads : 1,
            static_cast<long long>(ms_init),
            static_cast<long long>(ms_build),
            static_cast<long long>(ms_graph),
            static_cast<long long>(ms_compute),
            static_cast<long long>(ms_copy),
            static_cast<long long>(ms_total));
    }

    if (free_temp_contexts) {
        if (graph_ctx) {
            ggml_free(graph_ctx);
        }
        if (compute_ctx) {
            ggml_free(compute_ctx);
        }
    }
    return ACE_GGML_OK;
}
ace_ggml_status ace_ggml_vae_encode(
    ace_ggml_context * ctx,
    const float * audio,
    int32_t n_samples,
    float * out,
    size_t out_size) {

    if (!ctx || !audio || !out || n_samples <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    const int32_t latent_channels = ctx->vae.cfg.decoder_input_channels;
    const int32_t hop_length = ctx->vae.cfg.hop_length;
    const size_t expected = static_cast<size_t>(n_samples / hop_length) *
                            static_cast<size_t>(latent_channels) * sizeof(float);
    if (out_size < expected) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * out_tensor = ace_vae::forward_encode(compute_ctx, ctx->vae, audio, n_samples);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "vae encode failed");
    }
    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);

    size_t graph_size = 262144;
    graph_size = ace_read_graph_size_env(graph_size, "ACE_GGML_VAE_GRAPH_SIZE", "ACE_GGML_GRAPH_SIZE");
    std::vector<uint8_t> graph_buffer(ggml_graph_overhead_custom(graph_size, false));
    ggml_init_params gparams{};
    gparams.mem_size = graph_buffer.size();
    gparams.mem_buffer = graph_buffer.data();
    gparams.no_alloc = true;
    ggml_context * graph_ctx = ggml_init(gparams);
    if (!graph_ctx) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init graph failed");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
    if (!graph) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_new_graph_custom failed");
    }
    ggml_build_forward_expand(graph, out_f32);

    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }
    if (!out_f32->data) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }

    const int32_t out_len = static_cast<int32_t>(out_f32->ne[0]);
    const int32_t out_ch = static_cast<int32_t>(out_f32->ne[1]);
    if (out_ch != latent_channels || out_len <= 0) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "unexpected vae output shape");
    }
    const size_t needed = static_cast<size_t>(out_len) * static_cast<size_t>(out_ch) * sizeof(float);
    if (out_size < needed) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small for actual output");
    }

    const float * src = static_cast<const float *>(out_f32->data);
    for (int32_t t = 0; t < out_len; ++t) {
        for (int32_t c = 0; c < out_ch; ++c) {
            out[static_cast<size_t>(t) * out_ch + c] = src[static_cast<size_t>(t) + static_cast<size_t>(c) * out_len];
        }
    }

    ggml_free(graph_ctx);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_text_encoder_forward(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    float * out,
    size_t out_size) {

    if (!ctx || !token_ids || !out || n_tokens <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }

    const int32_t hidden = ctx->text_encoder.cfg.hidden_size;
    size_t needed = static_cast<size_t>(hidden) * static_cast<size_t>(n_tokens) * sizeof(float);
    if (out_size < needed) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * out_tensor = ace_qwen::forward_text_encoder(compute_ctx, ctx->text_encoder, token_ids, nullptr, n_tokens, true);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "forward failed");
    }

    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, out_f32);

    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }

    if (!out_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }
    std::memcpy(out, out_f32->data, needed);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_text_encoder_forward_masked(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    float * out,
    size_t out_size) {

    if (!ctx || !token_ids || !attention_mask || !out || n_tokens <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }

    const int32_t hidden = ctx->text_encoder.cfg.hidden_size;
    size_t needed = static_cast<size_t>(hidden) * static_cast<size_t>(n_tokens) * sizeof(float);
    if (out_size < needed) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * out_tensor = ace_qwen::forward_text_encoder(compute_ctx, ctx->text_encoder, token_ids, attention_mask, n_tokens, true);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "forward failed");
    }

    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, out_f32);

    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }

    if (!out_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }
    std::memcpy(out, out_f32->data, needed);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_text_encoder_forward_embeddings(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    float * out,
    size_t out_size) {

    if (!ctx || !token_ids || !out || n_tokens <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }

    const int32_t hidden = ctx->text_encoder.cfg.hidden_size;
    size_t needed = static_cast<size_t>(hidden) * static_cast<size_t>(n_tokens) * sizeof(float);
    if (out_size < needed) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * out_tensor = ace_qwen::forward_text_encoder_embeddings(
        compute_ctx, ctx->text_encoder, token_ids, n_tokens);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "embedding forward failed");
    }

    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, out_f32);

    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }

    if (!out_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }
    std::memcpy(out, out_f32->data, needed);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_text_encoder_forward_layers(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t apply_final_norm,
    float * out,
    size_t out_size) {

    if (!ctx || !token_ids || !out || n_tokens <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }

    const int32_t hidden = ctx->text_encoder.cfg.hidden_size;
    size_t needed = static_cast<size_t>(hidden) * static_cast<size_t>(n_tokens) * sizeof(float);
    if (out_size < needed) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * out_tensor = ace_qwen::forward_text_encoder_layers(
        compute_ctx, ctx->text_encoder, token_ids, attention_mask, n_tokens, true, n_layers, apply_final_norm != 0);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "forward layers failed");
    }

    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
    ggml_cgraph * graph = ggml_new_graph(compute_ctx);
    ggml_build_forward_expand(graph, out_f32);

    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }

    if (!out_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
    }
    std::memcpy(out, out_f32->data, needed);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_dit_forward(
    ace_ggml_context * ctx,
    const float * hidden_states,
    const float * context_latents,
    const float * encoder_hidden_states,
    const int32_t * attention_mask,
    const int32_t * encoder_attention_mask,
    int32_t seq_len,
    int32_t enc_len,
    float timestep,
    float timestep_r,
    float * out,
    size_t out_size) {

    if (!ctx || !out || seq_len <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->dit.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "dit not loaded");
    }

    const int32_t out_channels = ctx->dit.cfg.audio_acoustic_hidden_dim;
    const size_t needed = static_cast<size_t>(out_channels) * static_cast<size_t>(seq_len) * sizeof(float);
    if (out_size < needed) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    size_t graph_size = 12000000;
    const size_t layer_hint = ctx->dit.layers.size() * 900000ull;
    if (layer_hint > graph_size) {
        graph_size = layer_hint;
    }
    graph_size = ace_read_graph_size_env(graph_size, "ACE_GGML_DIT_GRAPH_SIZE", "ACE_GGML_GRAPH_SIZE");
    const bool use_backend = ace_should_use_metal_dit(ctx);
    const bool profile = ace_env_enabled("ACE_GGML_DIT_PROFILE");

    if (std::getenv("ACE_GGML_DEBUG_GRAPH")) {
        std::fprintf(
            stderr,
            "ace_ggml_dit_forward graph_size=%zu compute_buffer=%zu backend=%s\n",
            graph_size,
            ctx->compute_buffer_bytes,
            use_backend ? "metal" : "cpu");
        std::fflush(stderr);
    }

    const auto t0 = std::chrono::steady_clock::now();
    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;

    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }
    const auto t_compute_init = std::chrono::steady_clock::now();

    ggml_tensor * out_tensor = ace_dit::forward_dit(
        compute_ctx,
        ctx->dit,
        hidden_states,
        context_latents,
        encoder_hidden_states,
        attention_mask,
        encoder_attention_mask,
        seq_len,
        enc_len,
        timestep,
        timestep_r);
    if (!out_tensor) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "dit forward failed");
    }
    const auto t_graph_build = std::chrono::steady_clock::now();

    ggml_tensor * out_f32 = ggml_cast(compute_ctx, out_tensor, GGML_TYPE_F32);
    if (!out_f32) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "dit cast failed");
    }
    if (use_backend) {
        ggml_set_output(out_f32);
    }

    std::vector<uint8_t> graph_buffer(ggml_graph_overhead_custom(graph_size, false));
    ggml_init_params gparams{};
    gparams.mem_size = graph_buffer.size();
    gparams.mem_buffer = graph_buffer.data();
    gparams.no_alloc = true;
    ggml_context * graph_ctx = ggml_init(gparams);
    if (!graph_ctx) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init graph failed");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
    if (!graph) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_new_graph_custom failed");
    }
    ggml_build_forward_expand(graph, out_f32);
    const auto t_graph_ready = std::chrono::steady_clock::now();

    std::vector<float> backend_output;
    ggml_status status = GGML_STATUS_FAILED;

    if (use_backend) {
        ggml_backend_sched_reset(ctx->dit_sched);
        ggml_backend_sched_set_tensor_backend(ctx->dit_sched, out_f32, ctx->vae_backend);
        if (!ggml_backend_sched_alloc_graph(ctx->dit_sched, graph)) {
            ggml_free(graph_ctx);
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "ggml_backend_sched_alloc_graph failed");
        }

        status = ggml_backend_sched_graph_compute(ctx->dit_sched, graph);
        if (status == GGML_STATUS_SUCCESS) {
            backend_output.resize(static_cast<size_t>(out_channels) * static_cast<size_t>(seq_len));
            ggml_backend_tensor_get(out_f32, backend_output.data(), 0, needed);
        }
    } else {
        status = ggml_graph_compute_with_ctx(compute_ctx, graph, ctx->n_threads > 0 ? ctx->n_threads : 1);
    }
    const auto t_graph_done = std::chrono::steady_clock::now();

    if (status != GGML_STATUS_SUCCESS) {
        ggml_free(graph_ctx);
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "graph compute failed");
    }

    const float * src = nullptr;
    if (use_backend) {
        if (backend_output.empty()) {
            ggml_free(graph_ctx);
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "backend output copy is empty");
        }
        src = backend_output.data();
    } else {
        if (!out_f32->data) {
            ggml_free(graph_ctx);
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "output tensor has no data");
        }
        src = static_cast<const float *>(out_f32->data);
    }
    std::memcpy(out, src, needed);
    const auto t_copy_done = std::chrono::steady_clock::now();

    if (profile) {
        const auto ms_total = std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_done - t0).count();
        const auto ms_init = std::chrono::duration_cast<std::chrono::milliseconds>(t_compute_init - t0).count();
        const auto ms_build = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_build - t_compute_init).count();
        const auto ms_graph = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_ready - t_graph_build).count();
        const auto ms_compute = std::chrono::duration_cast<std::chrono::milliseconds>(t_graph_done - t_graph_ready).count();
        const auto ms_copy = std::chrono::duration_cast<std::chrono::milliseconds>(t_copy_done - t_graph_done).count();
        std::fprintf(
            stderr,
            "ace_ggml_dit_forward profile: backend=%s seq=%d enc=%d threads=%d init_ms=%lld build_ms=%lld graph_ms=%lld compute_ms=%lld copy_ms=%lld total_ms=%lld\n",
            use_backend ? "metal" : "cpu",
            seq_len,
            enc_len,
            ctx->n_threads > 0 ? ctx->n_threads : 1,
            static_cast<long long>(ms_init),
            static_cast<long long>(ms_build),
            static_cast<long long>(ms_graph),
            static_cast<long long>(ms_compute),
            static_cast<long long>(ms_copy),
            static_cast<long long>(ms_total));
    }

    ggml_free(graph_ctx);
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

static void ace_get_shift_schedule(float shift, std::vector<float> & out_schedule) {
    static const float s1[] = {1.0f, 0.875f, 0.75f, 0.625f, 0.5f, 0.375f, 0.25f, 0.125f};
    static const float s2[] = {1.0f, 0.9333333333f, 0.8571428571f, 0.7692307692f, 0.6666666667f, 0.5454545455f, 0.4f, 0.2222222222f};
    static const float s3[] = {1.0f, 0.9545454545f, 0.9f, 0.8333333333f, 0.75f, 0.6428571429f, 0.5f, 0.3f};

    const float d1 = std::fabs(shift - 1.0f);
    const float d2 = std::fabs(shift - 2.0f);
    const float d3 = std::fabs(shift - 3.0f);

    if (d1 <= d2 && d1 <= d3) {
        out_schedule.assign(std::begin(s1), std::end(s1));
    } else if (d2 <= d1 && d2 <= d3) {
        out_schedule.assign(std::begin(s2), std::end(s2));
    } else {
        out_schedule.assign(std::begin(s3), std::end(s3));
    }
}

static int32_t ace_read_nonneg_env_int(const char * key, int32_t fallback) {
    if (const char * value = std::getenv(key)) {
        char * endp = nullptr;
        const long long v = std::strtoll(value, &endp, 10);
        if (endp && endp != value && v >= 0) {
            return static_cast<int32_t>(v);
        }
    }
    return fallback;
}

static bool ace_load_silence_latent_f32(
    const char * path,
    int32_t seq_len,
    int32_t dim,
    std::vector<float> & out,
    std::string & err) {

    if (!path || !path[0]) {
        err = "empty silence latent path";
        return false;
    }
    if (seq_len <= 0 || dim <= 0) {
        err = "invalid silence latent shape request";
        return false;
    }

    std::ifstream fin(path, std::ios::binary | std::ios::ate);
    if (!fin) {
        err = std::string("failed to open silence latent file: ") + path;
        return false;
    }

    const std::streamsize size = fin.tellg();
    if (size <= 0) {
        err = "silence latent file is empty";
        return false;
    }
    if ((size % static_cast<std::streamsize>(sizeof(float))) != 0) {
        err = "silence latent file has invalid byte size";
        return false;
    }

    const size_t n_floats = static_cast<size_t>(size / static_cast<std::streamsize>(sizeof(float)));
    if (n_floats % static_cast<size_t>(dim) != 0) {
        err = "silence latent float count is not divisible by latent dim";
        return false;
    }

    const size_t n_frames = n_floats / static_cast<size_t>(dim);
    if (n_frames == 0) {
        err = "silence latent has zero frames";
        return false;
    }

    std::vector<float> full(n_floats, 0.0f);
    fin.seekg(0, std::ios::beg);
    fin.read(reinterpret_cast<char *>(full.data()), size);
    if (!fin) {
        err = "failed to read silence latent file";
        return false;
    }

    out.assign(static_cast<size_t>(seq_len) * static_cast<size_t>(dim), 0.0f);
    for (int32_t t = 0; t < seq_len; ++t) {
        const size_t src_t = std::min<size_t>(static_cast<size_t>(t), n_frames - 1);
        const float * src = full.data() + src_t * static_cast<size_t>(dim);
        float * dst = out.data() + static_cast<size_t>(t) * static_cast<size_t>(dim);
        std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
    }
    return true;
}

static ggml_tensor * ace_cast_f32(ggml_context * ctx, ggml_tensor * t) {
    if (t->type == GGML_TYPE_F32) {
        return t;
    }
    return ggml_cast(ctx, t, GGML_TYPE_F32);
}

static ggml_tensor * ace_add_bias(ggml_context * ctx, ggml_tensor * x, ggml_tensor * bias) {
    ggml_tensor * b = ace_cast_f32(ctx, bias);
    ggml_tensor * rep = ggml_repeat(ctx, b, x);
    return ggml_add(ctx, x, rep);
}

static ace_ggml_status ace_run_graph_for_single_output(
    ace_ggml_context * ctx,
    ggml_context * compute_ctx,
    ggml_tensor * out_f32,
    const char * env_graph_key,
    const char * error_prefix) {

    size_t graph_size = 262144;
    graph_size = ace_read_graph_size_env(graph_size, env_graph_key, "ACE_GGML_GRAPH_SIZE");

    std::vector<uint8_t> graph_buffer(ggml_graph_overhead_custom(graph_size, false));
    ggml_init_params gparams{};
    gparams.mem_size = graph_buffer.size();
    gparams.mem_buffer = graph_buffer.data();
    gparams.no_alloc = true;

    ggml_context * graph_ctx = ggml_init(gparams);
    if (!graph_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init graph failed");
    }

    ggml_cgraph * graph = ggml_new_graph_custom(graph_ctx, graph_size, false);
    if (!graph) {
        ggml_free(graph_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_new_graph_custom failed");
    }
    ggml_build_forward_expand(graph, out_f32);

    const int32_t n_threads = ctx->n_threads > 0 ? ctx->n_threads : 1;
    ggml_status status = ggml_graph_compute_with_ctx(compute_ctx, graph, n_threads);
    ggml_free(graph_ctx);
    if (status != GGML_STATUS_SUCCESS) {
        return ace_set_error(ctx, ACE_GGML_ERR, error_prefix);
    }
    return ACE_GGML_OK;
}

static ace_ggml_status ace_project_tokens_linear(
    ace_ggml_context * ctx,
    ggml_tensor * weight,
    ggml_tensor * bias,
    const float * in_states,
    int32_t n_tokens,
    int32_t in_dim,
    std::vector<float> & out_states,
    int32_t & out_dim) {

    if (!weight || !in_states || n_tokens <= 0 || in_dim <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "invalid linear projection args");
    }
    const int32_t w_in = static_cast<int32_t>(weight->ne[0]);
    const int32_t w_out = static_cast<int32_t>(weight->ne[1]);
    if (w_in != in_dim || w_out <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR, "linear projection weight shape mismatch");
    }
    if (bias && static_cast<int32_t>(bias->ne[0]) != w_out) {
        return ace_set_error(ctx, ACE_GGML_ERR, "linear projection bias shape mismatch");
    }

    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;
    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * x = ggml_new_tensor_2d(compute_ctx, GGML_TYPE_F32, in_dim, n_tokens);
    if (!x || !x->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "failed to allocate projection input");
    }
    std::memcpy(x->data, in_states, static_cast<size_t>(in_dim) * static_cast<size_t>(n_tokens) * sizeof(float));

    ggml_tensor * y = ggml_mul_mat(compute_ctx, weight, x);
    if (bias) {
        y = ace_add_bias(compute_ctx, y, bias);
    }
    ggml_tensor * y_f32 = ace_cast_f32(compute_ctx, y);
    ace_ggml_status st_graph = ace_run_graph_for_single_output(
        ctx, compute_ctx, y_f32, "ACE_GGML_COND_GRAPH_SIZE", "linear projection compute failed");
    if (st_graph != ACE_GGML_OK) {
        ggml_free(compute_ctx);
        return st_graph;
    }
    if (!y_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "linear projection output missing");
    }

    out_dim = w_out;
    out_states.resize(static_cast<size_t>(out_dim) * static_cast<size_t>(n_tokens));
    std::memcpy(out_states.data(), y_f32->data, out_states.size() * sizeof(float));
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

static ace_ggml_status ace_encode_lyric_condition(
    ace_ggml_context * ctx,
    const float * lyric_embeds,
    int32_t n_lyric_tokens,
    std::vector<float> & out_states,
    int32_t & out_dim) {

    if (!ctx || !lyric_embeds || n_lyric_tokens <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }

    std::vector<int32_t> lyric_mask(static_cast<size_t>(n_lyric_tokens), 1);
    std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
    ggml_init_params params{};
    params.mem_size = compute_buffer.size();
    params.mem_buffer = compute_buffer.data();
    params.no_alloc = false;
    ggml_context * compute_ctx = ggml_init(params);
    if (!compute_ctx) {
        return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
    }

    ggml_tensor * y = ace_dit::forward_lyric_encoder(
        compute_ctx,
        ctx->dit,
        lyric_embeds,
        lyric_mask.data(),
        n_lyric_tokens);
    if (!y) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "forward_lyric_encoder failed");
    }

    ggml_tensor * y_f32 = ace_cast_f32(compute_ctx, y);
    ace_ggml_status st_graph = ace_run_graph_for_single_output(
        ctx, compute_ctx, y_f32, "ACE_GGML_COND_GRAPH_SIZE", "lyric encoder compute failed");
    if (st_graph != ACE_GGML_OK) {
        ggml_free(compute_ctx);
        return st_graph;
    }
    if (!y_f32->data) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "lyric encoder output missing");
    }

    out_dim = static_cast<int32_t>(y_f32->ne[0]);
    const int32_t out_tokens = static_cast<int32_t>(y_f32->ne[1]);
    if (out_tokens != n_lyric_tokens || out_dim <= 0) {
        ggml_free(compute_ctx);
        return ace_set_error(ctx, ACE_GGML_ERR, "lyric encoder output shape mismatch");
    }
    out_states.resize(static_cast<size_t>(out_dim) * static_cast<size_t>(out_tokens));
    std::memcpy(out_states.data(), y_f32->data, out_states.size() * sizeof(float));
    ggml_free(compute_ctx);
    return ACE_GGML_OK;
}

static void ace_pack_sequences_single_batch(
    const std::vector<float> & hidden1,
    const std::vector<int32_t> & mask1,
    int32_t len1,
    const std::vector<float> & hidden2,
    const std::vector<int32_t> & mask2,
    int32_t len2,
    int32_t dim,
    std::vector<float> & out_hidden,
    std::vector<int32_t> & out_mask) {

    if (len1 <= 0 && len2 <= 0) {
        out_hidden.clear();
        out_mask.clear();
        return;
    }
    if (len2 <= 0) {
        out_hidden = hidden1;
        out_mask = mask1;
        return;
    }
    if (len1 <= 0) {
        out_hidden = hidden2;
        out_mask = mask2;
        return;
    }

    const int32_t len = len1 + len2;
    out_hidden.resize(static_cast<size_t>(len) * static_cast<size_t>(dim), 0.0f);
    out_mask.assign(static_cast<size_t>(len), 0);

    std::vector<int32_t> idx(static_cast<size_t>(len), 0);
    std::iota(idx.begin(), idx.end(), 0);
    auto has_token = [&](int32_t i) -> bool {
        if (i < len1) {
            return mask1[static_cast<size_t>(i)] != 0;
        }
        return mask2[static_cast<size_t>(i - len1)] != 0;
    };
    std::stable_partition(idx.begin(), idx.end(), has_token);

    int32_t n_valid = 0;
    for (int32_t i = 0; i < len; ++i) {
        const int32_t src_idx = idx[static_cast<size_t>(i)];
        const float * src = nullptr;
        if (src_idx < len1) {
            src = hidden1.data() + static_cast<size_t>(src_idx) * static_cast<size_t>(dim);
        } else {
            src = hidden2.data() + static_cast<size_t>(src_idx - len1) * static_cast<size_t>(dim);
        }
        float * dst = out_hidden.data() + static_cast<size_t>(i) * static_cast<size_t>(dim);
        std::memcpy(dst, src, static_cast<size_t>(dim) * sizeof(float));
        if (has_token(src_idx)) {
            ++n_valid;
        }
    }
    std::fill(out_mask.begin(), out_mask.begin() + n_valid, 1);
}

static ace_ggml_status ace_encode_timbre_condition(
    ace_ggml_context * ctx,
    const float * refer_audio_acoustic_hidden_states,
    const int32_t * refer_audio_order_mask,
    int32_t n_refer_audio,
    int32_t refer_audio_len,
    std::vector<float> & out_states,
    std::vector<int32_t> & out_mask,
    int32_t & out_dim) {

    if (!ctx || !refer_audio_acoustic_hidden_states || n_refer_audio <= 0 || refer_audio_len <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->dit.timbre_embed_w) {
        return ace_set_error(ctx, ACE_GGML_ERR, "timbre encoder weights are not loaded");
    }

    const int32_t in_dim =
        ctx->dit.cfg.timbre_hidden_dim > 0 ? ctx->dit.cfg.timbre_hidden_dim :
        (ctx->dit.cfg.audio_acoustic_hidden_dim > 0 ? ctx->dit.cfg.audio_acoustic_hidden_dim : 64);
    if (in_dim <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR, "invalid timbre hidden dimension");
    }

    out_states.clear();
    out_mask.clear();
    out_dim = 0;

    for (int32_t i = 0; i < n_refer_audio; ++i) {
        if (refer_audio_order_mask && refer_audio_order_mask[i] != 0) {
            return ace_set_error(ctx, ACE_GGML_ERR_UNSUPPORTED, "multi-batch refer_audio_order_mask is not supported");
        }
        const float * ref_ptr = refer_audio_acoustic_hidden_states +
            static_cast<size_t>(i) * static_cast<size_t>(refer_audio_len) * static_cast<size_t>(in_dim);

        std::vector<int32_t> ref_mask(static_cast<size_t>(refer_audio_len), 1);
        std::vector<uint8_t> compute_buffer(ctx->compute_buffer_bytes);
        ggml_init_params params{};
        params.mem_size = compute_buffer.size();
        params.mem_buffer = compute_buffer.data();
        params.no_alloc = false;
        ggml_context * compute_ctx = ggml_init(params);
        if (!compute_ctx) {
            return ace_set_error(ctx, ACE_GGML_ERR, "ggml_init compute failed");
        }

        ggml_tensor * y = ace_dit::forward_timbre_encoder(
            compute_ctx,
            ctx->dit,
            ref_ptr,
            ref_mask.data(),
            refer_audio_len);
        if (!y) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "forward_timbre_encoder failed");
        }

        ggml_tensor * y_f32 = ace_cast_f32(compute_ctx, y);
        ace_ggml_status st_graph = ace_run_graph_for_single_output(
            ctx, compute_ctx, y_f32, "ACE_GGML_COND_GRAPH_SIZE", "timbre encoder compute failed");
        if (st_graph != ACE_GGML_OK) {
            ggml_free(compute_ctx);
            return st_graph;
        }
        if (!y_f32->data) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "timbre encoder output missing");
        }

        const int32_t cur_dim = static_cast<int32_t>(y_f32->ne[0]);
        const int32_t cur_tokens = static_cast<int32_t>(y_f32->ne[1]);
        if (cur_tokens != 1 || cur_dim <= 0) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "timbre encoder output shape mismatch");
        }
        if (out_dim == 0) {
            out_dim = cur_dim;
        } else if (out_dim != cur_dim) {
            ggml_free(compute_ctx);
            return ace_set_error(ctx, ACE_GGML_ERR, "timbre encoder output dim mismatch");
        }

        const size_t old_size = out_states.size();
        out_states.resize(old_size + static_cast<size_t>(cur_dim), 0.0f);
        std::memcpy(
            out_states.data() + old_size,
            y_f32->data,
            static_cast<size_t>(cur_dim) * sizeof(float));
        out_mask.push_back(1);
        ggml_free(compute_ctx);
    }

    if (out_states.empty() || out_dim <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR, "empty timbre condition");
    }
    return ACE_GGML_OK;
}

static ace_ggml_status ace_generate_audio_from_encoder(
    ace_ggml_context * ctx,
    const float * encoder_hidden_states,
    const int32_t * encoder_attention_mask,
    int32_t enc_len,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels) {

    if (!ctx || !encoder_hidden_states || enc_len <= 0 || seq_len <= 0 || !out_audio) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->dit.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "dit not loaded");
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    const int32_t audio_dim = ctx->dit.cfg.audio_acoustic_hidden_dim;
    const int32_t ctx_dim = ctx->dit.cfg.in_channels - audio_dim;
    if (ctx_dim <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR, "invalid dit context dimension");
    }

    int32_t latent_channels = 0;
    int32_t audio_channels = 0;
    int32_t hop_length = 0;
    ace_ggml_status st_info = ace_ggml_vae_get_info(ctx, &latent_channels, &audio_channels, &hop_length);
    if (st_info != ACE_GGML_OK) {
        return st_info;
    }
    if (latent_channels != audio_dim) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae latent channels mismatch with dit audio dim");
    }
    const size_t needed_audio = static_cast<size_t>(seq_len) * static_cast<size_t>(hop_length) *
                                static_cast<size_t>(audio_channels) * sizeof(float);
    if (out_size < needed_audio) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "output buffer too small");
    }

    std::vector<int32_t> attention_mask(static_cast<size_t>(seq_len), 1);
    std::vector<float> context_latents(static_cast<size_t>(seq_len) * ctx_dim, 0.0f);

    // Match the reference pipeline:
    // - context_latents = concat(src_latents, chunk_masks)
    // - text2music uses silence_latent as src_latents and all-ones chunk mask.
    const bool use_silence_context = ace_read_nonneg_env_int("ACE_GGML_USE_SILENCE_CONTEXT", 1) != 0;
    if (use_silence_context && ctx_dim > 0) {
        const int32_t src_dim = std::min<int32_t>(audio_dim, ctx_dim);
        std::vector<float> src_latents(static_cast<size_t>(seq_len) * static_cast<size_t>(src_dim), 0.0f);
        bool src_loaded = false;
        const int32_t debug_context = ace_read_nonneg_env_int("ACE_GGML_DEBUG_CONTEXT", 0);
        const float chunk_mask_fill = ace_read_nonneg_env_int("ACE_GGML_CHUNK_MASK_FILL", 1) != 0 ? 1.0f : 0.0f;

        if (src_dim > 0) {
            if (const char * f32_path = std::getenv("ACE_GGML_SILENCE_LATENT_F32")) {
                std::string load_err;
                if (ace_load_silence_latent_f32(f32_path, seq_len, src_dim, src_latents, load_err)) {
                    src_loaded = true;
                    if (debug_context) {
                        std::fprintf(stderr, "ace_ggml: silence latent loaded from %s\n", f32_path);
                    }
                } else if (debug_context) {
                    std::fprintf(stderr, "ace_ggml: failed to load silence latent %s (%s)\n",
                        f32_path, load_err.c_str());
                }
            }

            if (!src_loaded) {
                const int32_t encode_chunk_frames = ace_read_nonneg_env_int("ACE_GGML_VAE_ENCODE_CHUNK_FRAMES", 0);
                int32_t chunk_frames = seq_len;
                if (encode_chunk_frames > 0) {
                    chunk_frames = std::max<int32_t>(1, std::min<int32_t>(encode_chunk_frames, seq_len));
                } else {
                    // Keep silence-latent encode memory bounded by default.
                    // Environment overrides can still tune this value.
                    int32_t auto_chunk_frames = ace_read_nonneg_env_int("ACE_GGML_VAE_ENCODE_CHUNK_FRAMES_AUTO", 0);
                    if (auto_chunk_frames <= 0) {
                        auto_chunk_frames = (seq_len > 128) ? 64 : seq_len;
                    }
                    chunk_frames = std::max<int32_t>(1, std::min<int32_t>(auto_chunk_frames, seq_len));
                }

                bool encode_ok = true;
                for (int32_t frame0 = 0; frame0 < seq_len; frame0 += chunk_frames) {
                    const int32_t cur_frames = std::min<int32_t>(chunk_frames, seq_len - frame0);
                    const int32_t n_samples = cur_frames * hop_length;
                    std::vector<float> silence_audio(
                        static_cast<size_t>(n_samples) * static_cast<size_t>(audio_channels), 0.0f);
                    std::vector<float> chunk_latents(
                        static_cast<size_t>(cur_frames) * static_cast<size_t>(audio_dim), 0.0f);

                    const ace_ggml_status st_encode = ace_ggml_vae_encode(
                        ctx,
                        silence_audio.data(),
                        n_samples,
                        chunk_latents.data(),
                        chunk_latents.size() * sizeof(float));

                    if (st_encode != ACE_GGML_OK) {
                        encode_ok = false;
                        break;
                    }

                    for (int32_t t = 0; t < cur_frames; ++t) {
                        const float * src = chunk_latents.data() + static_cast<size_t>(t) * static_cast<size_t>(audio_dim);
                        float * dst = src_latents.data() +
                            static_cast<size_t>(frame0 + t) * static_cast<size_t>(src_dim);
                        std::memcpy(dst, src, static_cast<size_t>(src_dim) * sizeof(float));
                    }
                }

                if (encode_ok) {
                    src_loaded = true;
                } else {
                    // Keep zero src latents as fallback if VAE encode fails.
                    ctx->last_error.clear();
                    if (debug_context) {
                        std::fprintf(stderr, "ace_ggml: silence encode fallback failed, using zero src latents\n");
                    }
                }
            }
        }

        for (int32_t t = 0; t < seq_len; ++t) {
            float * ctx_row = context_latents.data() + static_cast<size_t>(t) * static_cast<size_t>(ctx_dim);
            if (src_dim > 0) {
                const float * src_row = src_latents.data() + static_cast<size_t>(t) * static_cast<size_t>(src_dim);
                std::memcpy(ctx_row, src_row, static_cast<size_t>(src_dim) * sizeof(float));
            }
            for (int32_t c = src_dim; c < ctx_dim; ++c) {
                ctx_row[c] = chunk_mask_fill;
            }
        }
    }

    std::mt19937 rng(static_cast<uint32_t>(seed));
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::vector<float> xt(static_cast<size_t>(seq_len) * audio_dim, 0.0f);
    for (float & v : xt) {
        v = dist(rng);
    }
    std::vector<float> vt(static_cast<size_t>(seq_len) * audio_dim, 0.0f);

    std::vector<float> schedule;
    ace_get_shift_schedule(shift, schedule);
    if (schedule.empty()) {
        return ace_set_error(ctx, ACE_GGML_ERR, "empty scheduler");
    }

    for (size_t i = 0; i < schedule.size(); ++i) {
        const float t = schedule[i];
        st_info = ace_ggml_dit_forward(
            ctx,
            xt.data(),
            context_latents.data(),
            encoder_hidden_states,
            attention_mask.data(),
            encoder_attention_mask,
            seq_len,
            enc_len,
            t,
            t,
            vt.data(),
            vt.size() * sizeof(float));
        if (st_info != ACE_GGML_OK) {
            return st_info;
        }

        if (i + 1 == schedule.size()) {
            for (size_t k = 0; k < xt.size(); ++k) {
                xt[k] = xt[k] - vt[k] * t;
            }
            break;
        }

        const float dt = t - schedule[i + 1];
        for (size_t k = 0; k < xt.size(); ++k) {
            xt[k] = xt[k] - vt[k] * dt;
        }
    }

    const int32_t debug_gen_stats = ace_read_nonneg_env_int("ACE_GGML_DEBUG_GENERATE_STATS", 0);
    if (debug_gen_stats) {
        auto frame_rms = [&](int32_t t) -> float {
            if (t < 0) {
                t = 0;
            } else if (t >= seq_len) {
                t = seq_len - 1;
            }
            const float * row = xt.data() + static_cast<size_t>(t) * static_cast<size_t>(audio_dim);
            double acc = 0.0;
            for (int32_t c = 0; c < audio_dim; ++c) {
                const double v = static_cast<double>(row[c]);
                acc += v * v;
            }
            return static_cast<float>(std::sqrt(acc / static_cast<double>(audio_dim)));
        };
        std::fprintf(stderr,
            "ace_ggml: xt_rms t0=%.6f t25%%=%.6f t50%%=%.6f t75%%=%.6f tEnd=%.6f\n",
            frame_rms(0),
            frame_rms(seq_len / 4),
            frame_rms(seq_len / 2),
            frame_rms((seq_len * 3) / 4),
            frame_rms(seq_len - 1));
        std::fflush(stderr);
    }

    const char * vae_chunk_env = std::getenv("ACE_GGML_VAE_CHUNK_FRAMES");
    int32_t vae_chunk_frames = ace_read_nonneg_env_int("ACE_GGML_VAE_CHUNK_FRAMES", -1);
    if (!vae_chunk_env || !vae_chunk_env[0]) {
        const int32_t auto_frames = ace_read_nonneg_env_int("ACE_GGML_VAE_CHUNK_FRAMES_AUTO", 0);
        if (auto_frames > 0) {
            vae_chunk_frames = auto_frames;
        } else if (seq_len > 128) {
            vae_chunk_frames = 128;
        } else {
            vae_chunk_frames = 0;
        }
    } else if (vae_chunk_frames < 0) {
        vae_chunk_frames = 0;
    }
    int32_t produced_samples = seq_len * hop_length;
    if (vae_chunk_frames > 0 && vae_chunk_frames < seq_len) {
        const size_t samples_per_frame = static_cast<size_t>(hop_length) * static_cast<size_t>(audio_channels);
        const size_t channel_stride = static_cast<size_t>(audio_channels);
        const size_t out_capacity_samples = out_size / (sizeof(float) * channel_stride);
        int32_t overlap = ace_read_nonneg_env_int("ACE_GGML_VAE_CHUNK_OVERLAP_FRAMES", -1);
        if (overlap < 0) {
            const int32_t auto_overlap = std::max<int32_t>(1, vae_chunk_frames / 4);
            overlap = std::min<int32_t>(64, auto_overlap);
        }
        if (overlap * 2 >= vae_chunk_frames) {
            overlap = std::max<int32_t>(0, (vae_chunk_frames / 2) - 1);
        }
        int32_t stride = vae_chunk_frames - 2 * overlap;
        if (stride <= 0) {
            overlap = 0;
            stride = vae_chunk_frames;
        }
        double upsample_factor = -1.0;
        size_t audio_write_pos = 0;  // In decoded samples (not floats).
        int32_t chunk_idx = 0;
        for (int32_t core_start = 0; core_start < seq_len; core_start += stride) {
            const int32_t core_end = std::min<int32_t>(core_start + stride, seq_len);
            const int32_t win_start = std::max<int32_t>(0, core_start - overlap);
            const int32_t win_end = std::min<int32_t>(seq_len, core_end + overlap);
            const int32_t win_frames = win_end - win_start;
            std::vector<float> chunk_audio(static_cast<size_t>(win_frames) * samples_per_frame, 0.0f);
            st_info = ace_ggml_vae_decode(
                ctx,
                xt.data() + static_cast<size_t>(win_start) * static_cast<size_t>(audio_dim),
                win_frames,
                chunk_audio.data(),
                chunk_audio.size() * sizeof(float));
            if (st_info != ACE_GGML_OK) {
                return st_info;
            }

            const int32_t added_start = core_start - win_start;
            const int32_t added_end = win_end - core_end;
            const size_t decoded_samples = chunk_audio.empty() ? 0 : (chunk_audio.size() / channel_stride);
            if (upsample_factor <= 0.0 && win_frames > 0) {
                upsample_factor = static_cast<double>(decoded_samples) / static_cast<double>(win_frames);
            }
            const double trim_factor = upsample_factor > 0.0 ? upsample_factor : static_cast<double>(hop_length);

            int32_t trim_start_samples = static_cast<int32_t>(std::llround(static_cast<double>(added_start) * trim_factor));
            int32_t trim_end_samples = static_cast<int32_t>(std::llround(static_cast<double>(added_end) * trim_factor));
            trim_start_samples = std::max<int32_t>(0, std::min<int32_t>(trim_start_samples, static_cast<int32_t>(decoded_samples)));
            trim_end_samples = std::max<int32_t>(0, std::min<int32_t>(trim_end_samples, static_cast<int32_t>(decoded_samples)));
            int32_t end_sample = static_cast<int32_t>(decoded_samples) - trim_end_samples;
            if (end_sample < trim_start_samples) {
                end_sample = trim_start_samples;
            }
            size_t core_samples = static_cast<size_t>(end_sample - trim_start_samples);
            if (core_samples > 0 && audio_write_pos < out_capacity_samples) {
                const size_t max_copy_samples = out_capacity_samples - audio_write_pos;
                if (core_samples > max_copy_samples) {
                    core_samples = max_copy_samples;
                }
                const size_t src_offset = static_cast<size_t>(trim_start_samples) * channel_stride;
                const size_t dst_offset = audio_write_pos * channel_stride;
                const size_t copy_floats = core_samples * channel_stride;
                std::memcpy(out_audio + dst_offset, chunk_audio.data() + src_offset, copy_floats * sizeof(float));
                audio_write_pos += core_samples;

                if (debug_gen_stats) {
                    double acc = 0.0;
                    double peak = 0.0;
                    const float * core_ptr = chunk_audio.data() + src_offset;
                    for (size_t i = 0; i < copy_floats; ++i) {
                        const double d = static_cast<double>(core_ptr[i]);
                        acc += d * d;
                        peak = std::max(peak, std::fabs(d));
                    }
                    const double rms = copy_floats == 0 ? 0.0 : std::sqrt(acc / static_cast<double>(copy_floats));
                    std::fprintf(stderr,
                        "ace_ggml: vae_chunk idx=%d frame=[%d,%d) rms=%.6f peak=%.6f\n",
                        chunk_idx,
                        core_start,
                        core_end,
                        rms,
                        peak);
                    std::fflush(stderr);
                }
            }
            ++chunk_idx;
        }
        produced_samples = static_cast<int32_t>(audio_write_pos);
        if (debug_gen_stats) {
            std::fprintf(stderr,
                "ace_ggml: vae_chunk_concat samples=%d expected=%d factor=%.6f\n",
                produced_samples,
                seq_len * hop_length,
                upsample_factor > 0.0 ? upsample_factor : static_cast<double>(hop_length));
            std::fflush(stderr);
        }
    } else {
        st_info = ace_ggml_vae_decode(ctx, xt.data(), seq_len, out_audio, out_size);
        if (st_info != ACE_GGML_OK) {
            return st_info;
        }
    }

    if (out_audio_samples) {
        *out_audio_samples = produced_samples;
    }
    if (out_audio_channels) {
        *out_audio_channels = audio_channels;
    }
    return ACE_GGML_OK;
}

ace_ggml_status ace_ggml_generate_audio_simple(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels) {

    if (!ctx || !token_ids || !out_audio || n_tokens <= 0 || seq_len <= 0) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }
    if (!ctx->dit.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "dit not loaded");
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    const int32_t dit_hidden = ctx->dit.cfg.hidden_size;
    const int32_t text_hidden = ctx->text_encoder.cfg.hidden_size;
    const bool allow_text_mismatch = std::getenv("ACE_GGML_ALLOW_TEXT_DIM_MISMATCH") != nullptr;
    if (text_hidden != dit_hidden && !allow_text_mismatch) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder hidden size mismatch with dit");
    }

    std::vector<float> text_states(static_cast<size_t>(n_tokens) * text_hidden, 0.0f);
    int32_t max_text_layers = ace_read_nonneg_env_int("ACE_GGML_TEXT_MAX_LAYERS", -1);
    ace_ggml_status st_info = ACE_GGML_OK;
    if (max_text_layers >= 0) {
        st_info = ace_ggml_text_encoder_forward_layers(
            ctx,
            token_ids,
            nullptr,
            n_tokens,
            max_text_layers,
            1,
            text_states.data(),
            text_states.size() * sizeof(float));
    } else {
        st_info = ace_ggml_text_encoder_forward(
            ctx,
            token_ids,
            n_tokens,
            text_states.data(),
            text_states.size() * sizeof(float));
    }
    if (st_info != ACE_GGML_OK) {
        return st_info;
    }

    std::vector<float> encoder_hidden_states(static_cast<size_t>(n_tokens) * dit_hidden, 0.0f);
    if (text_hidden == dit_hidden) {
        std::memcpy(encoder_hidden_states.data(), text_states.data(), text_states.size() * sizeof(float));
    } else {
        const int32_t cpy = std::min(text_hidden, dit_hidden);
        for (int32_t t = 0; t < n_tokens; ++t) {
            const float * src = text_states.data() + static_cast<size_t>(t) * text_hidden;
            float * dst = encoder_hidden_states.data() + static_cast<size_t>(t) * dit_hidden;
            std::memcpy(dst, src, static_cast<size_t>(cpy) * sizeof(float));
        }
    }

    std::vector<int32_t> encoder_attention_mask(static_cast<size_t>(n_tokens), 1);
    return ace_generate_audio_from_encoder(
        ctx,
        encoder_hidden_states.data(),
        encoder_attention_mask.data(),
        n_tokens,
        seq_len,
        shift,
        seed,
        out_audio,
        out_size,
        out_audio_samples,
        out_audio_channels);
}

static ace_ggml_status ace_generate_audio_style_lyric_timbre_impl(
    ace_ggml_context * ctx,
    const int32_t * style_token_ids,
    int32_t n_style_tokens,
    const int32_t * lyric_token_ids,
    int32_t n_lyric_tokens,
    const float * refer_audio_acoustic_hidden_states,
    const int32_t * refer_audio_order_mask,
    int32_t n_refer_audio,
    int32_t refer_audio_len,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels) {

    const bool has_style = n_style_tokens > 0;
    const bool has_lyric = n_lyric_tokens > 0;
    const bool has_timbre = n_refer_audio > 0;

    if (!ctx || !out_audio || seq_len <= 0 || (!has_style && !has_lyric && !has_timbre)) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (has_style && !style_token_ids) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (has_lyric && !lyric_token_ids) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (has_timbre && (!refer_audio_acoustic_hidden_states || refer_audio_len <= 0)) {
        return ACE_GGML_ERR_INVALID_ARG;
    }
    if (!ctx->text_encoder.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "text encoder not loaded");
    }
    if (!ctx->dit.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "dit not loaded");
    }
    if (!ctx->vae.loaded) {
        return ace_set_error(ctx, ACE_GGML_ERR, "vae not loaded");
    }

    const int32_t text_hidden = ctx->text_encoder.cfg.hidden_size;
    const int32_t dit_hidden = ctx->dit.cfg.hidden_size;
    const bool allow_text_mismatch = std::getenv("ACE_GGML_ALLOW_TEXT_DIM_MISMATCH") != nullptr;

    std::vector<float> style_states;
    std::vector<float> lyric_states;
    ace_ggml_status st = ACE_GGML_OK;

    if (has_style) {
        style_states.resize(static_cast<size_t>(n_style_tokens) * static_cast<size_t>(text_hidden), 0.0f);
        const int32_t max_text_layers = ace_read_nonneg_env_int("ACE_GGML_TEXT_MAX_LAYERS", -1);
        if (max_text_layers >= 0) {
            st = ace_ggml_text_encoder_forward_layers(
                ctx,
                style_token_ids,
                nullptr,
                n_style_tokens,
                max_text_layers,
                1,
                style_states.data(),
                style_states.size() * sizeof(float));
        } else {
            st = ace_ggml_text_encoder_forward(
                ctx,
                style_token_ids,
                n_style_tokens,
                style_states.data(),
                style_states.size() * sizeof(float));
        }
        if (st != ACE_GGML_OK) {
            return st;
        }
    }

    if (has_lyric) {
        lyric_states.resize(static_cast<size_t>(n_lyric_tokens) * static_cast<size_t>(text_hidden), 0.0f);
        st = ace_ggml_text_encoder_forward_embeddings(
            ctx,
            lyric_token_ids,
            n_lyric_tokens,
            lyric_states.data(),
            lyric_states.size() * sizeof(float));
        if (st != ACE_GGML_OK) {
            return st;
        }
    }

    std::vector<float> style_cond;
    std::vector<float> lyric_cond;
    std::vector<float> timbre_cond;
    std::vector<int32_t> timbre_mask;
    int32_t style_cond_dim = 0;
    int32_t lyric_cond_dim = 0;
    int32_t timbre_cond_dim = 0;
    bool style_encoded = false;
    bool lyric_encoded = false;

    if (has_style && ctx->dit.text_projector_w) {
        st = ace_project_tokens_linear(
            ctx,
            ctx->dit.text_projector_w,
            nullptr,
            style_states.data(),
            n_style_tokens,
            text_hidden,
            style_cond,
            style_cond_dim);
        if (st != ACE_GGML_OK) {
            return st;
        }
        style_encoded = (style_cond_dim == dit_hidden);
    }
    if (has_lyric && (ctx->dit.lyric_embed_w || ctx->dit.text_projector_w)) {
        st = ace_encode_lyric_condition(
            ctx,
            lyric_states.data(),
            n_lyric_tokens,
            lyric_cond,
            lyric_cond_dim);
        if (st == ACE_GGML_OK && lyric_cond_dim == dit_hidden) {
            lyric_encoded = true;
        } else if (st != ACE_GGML_OK) {
            // Keep fallback path when lyric encoder is unavailable/incomplete.
            ctx->last_error.clear();
        }
    }
    if (has_timbre) {
        st = ace_encode_timbre_condition(
            ctx,
            refer_audio_acoustic_hidden_states,
            refer_audio_order_mask,
            n_refer_audio,
            refer_audio_len,
            timbre_cond,
            timbre_mask,
            timbre_cond_dim);
        if (st != ACE_GGML_OK) {
            return st;
        }
        if (timbre_cond_dim != dit_hidden) {
            return ace_set_error(ctx, ACE_GGML_ERR, "timbre condition dim mismatch with dit");
        }
    }

    if ((has_style && !style_encoded) || (has_lyric && !lyric_encoded)) {
        if (text_hidden != dit_hidden && !allow_text_mismatch) {
            return ace_set_error(ctx, ACE_GGML_ERR, "text encoder hidden size mismatch with dit");
        }
    }

    const int32_t cpy = std::min(text_hidden, dit_hidden);

    std::vector<float> style_hidden;
    std::vector<int32_t> style_mask;
    if (has_style) {
        style_hidden.resize(static_cast<size_t>(n_style_tokens) * static_cast<size_t>(dit_hidden), 0.0f);
        style_mask.assign(static_cast<size_t>(n_style_tokens), 1);
        if (style_encoded) {
            std::memcpy(style_hidden.data(), style_cond.data(), style_cond.size() * sizeof(float));
        } else {
            for (int32_t t = 0; t < n_style_tokens; ++t) {
                const float * src = style_states.data() + static_cast<size_t>(t) * static_cast<size_t>(text_hidden);
                float * dst = style_hidden.data() + static_cast<size_t>(t) * static_cast<size_t>(dit_hidden);
                std::memcpy(dst, src, static_cast<size_t>(cpy) * sizeof(float));
            }
        }
    }

    std::vector<float> lyric_hidden;
    std::vector<int32_t> lyric_mask;
    if (has_lyric) {
        lyric_hidden.resize(static_cast<size_t>(n_lyric_tokens) * static_cast<size_t>(dit_hidden), 0.0f);
        lyric_mask.assign(static_cast<size_t>(n_lyric_tokens), 1);
        if (lyric_encoded) {
            std::memcpy(lyric_hidden.data(), lyric_cond.data(), lyric_cond.size() * sizeof(float));
        } else {
            for (int32_t t = 0; t < n_lyric_tokens; ++t) {
                const float * src = lyric_states.data() + static_cast<size_t>(t) * static_cast<size_t>(text_hidden);
                float * dst = lyric_hidden.data() + static_cast<size_t>(t) * static_cast<size_t>(dit_hidden);
                std::memcpy(dst, src, static_cast<size_t>(cpy) * sizeof(float));
            }
        }
    }

    int32_t cond_len = 0;
    std::vector<float> encoder_hidden_states;
    std::vector<int32_t> encoder_attention_mask;
    if (has_lyric) {
        encoder_hidden_states = lyric_hidden;
        encoder_attention_mask = lyric_mask;
        cond_len = n_lyric_tokens;
    }

    if (has_timbre) {
        std::vector<float> packed_hidden;
        std::vector<int32_t> packed_mask;
        const int32_t timbre_len = static_cast<int32_t>(timbre_mask.size());
        ace_pack_sequences_single_batch(
            encoder_hidden_states,
            encoder_attention_mask,
            cond_len,
            timbre_cond,
            timbre_mask,
            timbre_len,
            dit_hidden,
            packed_hidden,
            packed_mask);
        encoder_hidden_states.swap(packed_hidden);
        encoder_attention_mask.swap(packed_mask);
        cond_len += timbre_len;
    }

    if (has_style) {
        std::vector<float> packed_hidden;
        std::vector<int32_t> packed_mask;
        ace_pack_sequences_single_batch(
            encoder_hidden_states,
            encoder_attention_mask,
            cond_len,
            style_hidden,
            style_mask,
            n_style_tokens,
            dit_hidden,
            packed_hidden,
            packed_mask);
        encoder_hidden_states.swap(packed_hidden);
        encoder_attention_mask.swap(packed_mask);
        cond_len += n_style_tokens;
    }

    if (cond_len <= 0) {
        return ace_set_error(ctx, ACE_GGML_ERR_INVALID_ARG, "empty style/lyric/timbre inputs");
    }

    return ace_generate_audio_from_encoder(
        ctx,
        encoder_hidden_states.data(),
        encoder_attention_mask.data(),
        cond_len,
        seq_len,
        shift,
        seed,
        out_audio,
        out_size,
        out_audio_samples,
        out_audio_channels);
}

ace_ggml_status ace_ggml_generate_audio_style_lyric_simple(
    ace_ggml_context * ctx,
    const int32_t * style_token_ids,
    int32_t n_style_tokens,
    const int32_t * lyric_token_ids,
    int32_t n_lyric_tokens,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels) {

    return ace_generate_audio_style_lyric_timbre_impl(
        ctx,
        style_token_ids,
        n_style_tokens,
        lyric_token_ids,
        n_lyric_tokens,
        nullptr,
        nullptr,
        0,
        0,
        seq_len,
        shift,
        seed,
        out_audio,
        out_size,
        out_audio_samples,
        out_audio_channels);
}

ace_ggml_status ace_ggml_generate_audio_style_lyric_timbre_simple(
    ace_ggml_context * ctx,
    const int32_t * style_token_ids,
    int32_t n_style_tokens,
    const int32_t * lyric_token_ids,
    int32_t n_lyric_tokens,
    const float * refer_audio_acoustic_hidden_states,
    const int32_t * refer_audio_order_mask,
    int32_t n_refer_audio,
    int32_t refer_audio_len,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels) {

    return ace_generate_audio_style_lyric_timbre_impl(
        ctx,
        style_token_ids,
        n_style_tokens,
        lyric_token_ids,
        n_lyric_tokens,
        refer_audio_acoustic_hidden_states,
        refer_audio_order_mask,
        n_refer_audio,
        refer_audio_len,
        seq_len,
        shift,
        seed,
        out_audio,
        out_size,
        out_audio_samples,
        out_audio_channels);
}
