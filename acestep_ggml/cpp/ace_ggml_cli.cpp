#include "acestep_ggml.h"
#include "acestep_dit_config.h"
#include "qwen_config.h"

#include <cstdio>
#include <cstdlib>
#include <random>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

static void print_usage() {
    std::fprintf(stderr,
                 "Usage:\n"
                 "  ace_ggml_cli --text-encoder <model_dir> --tokens 1,2,3 [--mask 1,1,0] [--embed] [--layers N] [--final-norm] [--threads N] [--use-metal]\n"
                 "  ace_ggml_cli --dit <model_dir> --seq-len N [--enc-len M] [--timestep T] [--timestep-r R] [--mask 1,1,..] [--enc-mask 1,1,..] [--seed S] [--threads N] [--compute-buffer-mb MB] [--use-metal]\n"
                 "  ace_ggml_cli --vae <model_dir_or_vae_dir> --latent-len N [--encode --audio-len M] [--seed S] [--threads N] [--compute-buffer-mb MB] [--use-metal]\n"
                 "  ace_ggml_cli --pipeline --text-encoder <lm_dir> --dit <dit_dir> --vae <vae_dir_or_root> --tokens 1,2,3 (--seq-len N|--audio-seconds SEC) [--shift 1|2|3] [--seed S] [--threads N] [--compute-buffer-mb MB] [--use-metal]\n"
                 "  ace_ggml_cli --pipeline-style-lyric --text-encoder <lm_dir> --dit <dit_dir> --vae <vae_dir_or_root> --style-tokens 1,2 --lyric-tokens 3,4 (--seq-len N|--audio-seconds SEC) [--shift 1|2|3] [--seed S] [--threads N] [--compute-buffer-mb MB] [--use-metal]\n"
                 "  ace_ggml_cli --pipeline-style-lyric-timbre --text-encoder <lm_dir> --dit <dit_dir> --vae <vae_dir_or_root> --style-tokens 1,2 --lyric-tokens 3,4 --timbre-rand-n N --timbre-len L [--timbre-order 0,0] (--seq-len N|--audio-seconds SEC) [--shift 1|2|3] [--seed S] [--threads N] [--compute-buffer-mb MB] [--use-metal]\n");
}

int main(int argc, char ** argv) {
    std::string model_dir;
    std::string dit_dir;
    std::string vae_dir;
    std::string token_str;
    std::string style_token_str;
    std::string lyric_token_str;
    std::string timbre_order_str;
    std::string mask_str;
    std::string enc_mask_str;
    int threads = 1;
    int compute_buffer_mb = 512;
    bool embed_only = false;
    int layers = -1;
    bool apply_final_norm = false;
    int seq_len = 0;
    int enc_len = 0;
    float timestep = 0.5f;
    float timestep_r = 0.0f;
    int seed = 1234;
    int latent_len = 0;
    bool vae_encode = false;
    int audio_len = 0;
    bool pipeline = false;
    bool pipeline_style_lyric = false;
    bool pipeline_style_lyric_timbre = false;
    int timbre_rand_n = 0;
    int timbre_len = 0;
    float shift = 3.0f;
    float audio_seconds = -1.0f;
    bool use_metal = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--text-encoder" && i + 1 < argc) {
            model_dir = argv[++i];
        } else if (arg == "--dit" && i + 1 < argc) {
            dit_dir = argv[++i];
        } else if (arg == "--vae" && i + 1 < argc) {
            vae_dir = argv[++i];
        } else if (arg == "--tokens" && i + 1 < argc) {
            token_str = argv[++i];
        } else if (arg == "--style-tokens" && i + 1 < argc) {
            style_token_str = argv[++i];
        } else if (arg == "--lyric-tokens" && i + 1 < argc) {
            lyric_token_str = argv[++i];
        } else if (arg == "--mask" && i + 1 < argc) {
            mask_str = argv[++i];
        } else if (arg == "--enc-mask" && i + 1 < argc) {
            enc_mask_str = argv[++i];
        } else if (arg == "--embed") {
            embed_only = true;
        } else if (arg == "--layers" && i + 1 < argc) {
            layers = std::atoi(argv[++i]);
        } else if (arg == "--final-norm") {
            apply_final_norm = true;
        } else if (arg == "--seq-len" && i + 1 < argc) {
            seq_len = std::atoi(argv[++i]);
        } else if (arg == "--audio-seconds" && i + 1 < argc) {
            audio_seconds = std::strtof(argv[++i], nullptr);
        } else if (arg == "--enc-len" && i + 1 < argc) {
            enc_len = std::atoi(argv[++i]);
        } else if (arg == "--timestep" && i + 1 < argc) {
            timestep = std::strtof(argv[++i], nullptr);
        } else if (arg == "--timestep-r" && i + 1 < argc) {
            timestep_r = std::strtof(argv[++i], nullptr);
        } else if (arg == "--seed" && i + 1 < argc) {
            seed = std::atoi(argv[++i]);
        } else if (arg == "--latent-len" && i + 1 < argc) {
            latent_len = std::atoi(argv[++i]);
        } else if (arg == "--encode") {
            vae_encode = true;
        } else if (arg == "--audio-len" && i + 1 < argc) {
            audio_len = std::atoi(argv[++i]);
        } else if (arg == "--pipeline") {
            pipeline = true;
        } else if (arg == "--pipeline-style-lyric") {
            pipeline_style_lyric = true;
        } else if (arg == "--pipeline-style-lyric-timbre") {
            pipeline_style_lyric_timbre = true;
        } else if (arg == "--timbre-rand-n" && i + 1 < argc) {
            timbre_rand_n = std::atoi(argv[++i]);
        } else if (arg == "--timbre-len" && i + 1 < argc) {
            timbre_len = std::atoi(argv[++i]);
        } else if (arg == "--timbre-order" && i + 1 < argc) {
            timbre_order_str = argv[++i];
        } else if (arg == "--shift" && i + 1 < argc) {
            shift = std::strtof(argv[++i], nullptr);
        } else if (arg == "--threads" && i + 1 < argc) {
            threads = std::atoi(argv[++i]);
        } else if (arg == "--compute-buffer-mb" && i + 1 < argc) {
            compute_buffer_mb = std::atoi(argv[++i]);
        } else if (arg == "--use-metal") {
            use_metal = true;
        } else {
            print_usage();
            return 1;
        }
    }

    if (pipeline_style_lyric_timbre) {
        if (model_dir.empty() || dit_dir.empty() || vae_dir.empty() || (seq_len <= 0 && audio_seconds <= 0.0f)) {
            print_usage();
            return 1;
        }
        if (style_token_str.empty() && lyric_token_str.empty()) {
            std::fprintf(stderr, "style-tokens and lyric-tokens cannot both be empty\n");
            return 1;
        }
        if (timbre_rand_n <= 0 || timbre_len <= 0) {
            std::fprintf(stderr, "pipeline-style-lyric-timbre requires --timbre-rand-n > 0 and --timbre-len > 0\n");
            return 1;
        }
    } else if (pipeline_style_lyric) {
        if (model_dir.empty() || dit_dir.empty() || vae_dir.empty() || (seq_len <= 0 && audio_seconds <= 0.0f)) {
            print_usage();
            return 1;
        }
        if (style_token_str.empty() && lyric_token_str.empty()) {
            std::fprintf(stderr, "style-tokens and lyric-tokens cannot both be empty\n");
            return 1;
        }
    } else if (pipeline) {
        if (model_dir.empty() || dit_dir.empty() || vae_dir.empty() || token_str.empty() || (seq_len <= 0 && audio_seconds <= 0.0f)) {
            print_usage();
            return 1;
        }
    } else if (!vae_dir.empty()) {
        if (vae_encode) {
            if (audio_len <= 0) {
                print_usage();
                return 1;
            }
        } else if (latent_len <= 0) {
            print_usage();
            return 1;
        }
    } else if (!dit_dir.empty()) {
        if (seq_len <= 0) {
            print_usage();
            return 1;
        }
    } else {
        if (model_dir.empty() || token_str.empty()) {
            print_usage();
            return 1;
        }
    }

    std::vector<int32_t> tokens;
    auto parse_i32_csv = [](const std::string & s, std::vector<int32_t> & out) {
        size_t start = 0;
        while (start < s.size()) {
            size_t end = s.find(',', start);
            if (end == std::string::npos) end = s.size();
            std::string piece = s.substr(start, end - start);
            if (!piece.empty()) {
                out.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
            }
            start = end + 1;
        }
    };
    parse_i32_csv(token_str, tokens);

    std::vector<int32_t> style_tokens;
    std::vector<int32_t> lyric_tokens;
    std::vector<int32_t> timbre_order;
    parse_i32_csv(style_token_str, style_tokens);
    parse_i32_csv(lyric_token_str, lyric_tokens);
    parse_i32_csv(timbre_order_str, timbre_order);

    std::vector<int32_t> mask;
    if (!mask_str.empty()) {
        size_t start = 0;
        while (start < mask_str.size()) {
            size_t end = mask_str.find(',', start);
            if (end == std::string::npos) end = mask_str.size();
            std::string piece = mask_str.substr(start, end - start);
            if (!piece.empty()) {
                mask.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
            }
            start = end + 1;
        }
        if (!dit_dir.empty()) {
            if (seq_len <= 0) {
                std::fprintf(stderr, "mask requires explicit --seq-len when using auto duration\n");
                return 1;
            }
            if (mask.size() != static_cast<size_t>(seq_len)) {
                std::fprintf(stderr, "mask length must match seq_len\n");
                return 1;
            }
        } else {
            if (mask.size() != tokens.size()) {
                std::fprintf(stderr, "mask length must match tokens\n");
                return 1;
            }
        }
    }

    std::vector<int32_t> enc_mask;
    if (!enc_mask_str.empty()) {
        size_t start = 0;
        while (start < enc_mask_str.size()) {
            size_t end = enc_mask_str.find(',', start);
            if (end == std::string::npos) end = enc_mask_str.size();
            std::string piece = enc_mask_str.substr(start, end - start);
            if (!piece.empty()) {
                enc_mask.push_back(static_cast<int32_t>(std::strtol(piece.c_str(), nullptr, 10)));
            }
            start = end + 1;
        }
        if (enc_len > 0 && enc_mask.size() != static_cast<size_t>(enc_len)) {
            std::fprintf(stderr, "enc-mask length must match enc-len\n");
            return 1;
        }
    }

    if (pipeline_style_lyric_timbre) {
        if (!timbre_order.empty() && timbre_order.size() != static_cast<size_t>(timbre_rand_n)) {
            std::fprintf(stderr, "timbre-order length must match timbre-rand-n\n");
            return 1;
        }
        if (timbre_order.empty()) {
            timbre_order.assign(static_cast<size_t>(timbre_rand_n), 0);
        }
    }

    ace_ggml_context * ctx = nullptr;
    ace_ggml_init_params params{};
    params.n_threads = threads;
    params.use_metal = use_metal ? 1 : 0;
    params.compute_buffer_bytes = static_cast<size_t>(std::max(64, compute_buffer_mb)) * 1024ULL * 1024ULL;

    if (ace_ggml_create(&params, &ctx) != ACE_GGML_OK) {
        std::fprintf(stderr, "failed to create context\n");
        return 1;
    }

    if (pipeline || pipeline_style_lyric || pipeline_style_lyric_timbre) {
        ace_ggml_status st = ace_ggml_load_text_encoder(ctx, model_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "text encoder load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }
        st = ace_ggml_load_dit(ctx, dit_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "dit load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }
        st = ace_ggml_load_vae(ctx, vae_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "vae load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        int32_t latent_channels = 0;
        int32_t audio_channels = 0;
        int32_t hop_length = 0;
        st = ace_ggml_vae_get_info(ctx, &latent_channels, &audio_channels, &hop_length);
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "vae info failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        int32_t resolved_seq_len = seq_len;
        if (resolved_seq_len <= 0) {
            if (audio_seconds <= 0.0f) {
                std::fprintf(stderr, "either --seq-len or --audio-seconds must be provided\n");
                ace_ggml_destroy(ctx);
                return 1;
            }
            resolved_seq_len = std::max<int32_t>(1, static_cast<int32_t>(
                std::lround(static_cast<double>(audio_seconds) * 48000.0 / static_cast<double>(hop_length))));
            std::fprintf(stderr,
                "info: resolved seq_len=%d from audio_seconds=%.3f (hop=%d, sample_rate=48000)\n",
                resolved_seq_len, audio_seconds, hop_length);
        }

        const int32_t out_samples = resolved_seq_len * hop_length;
        std::vector<float> out(static_cast<size_t>(out_samples) * audio_channels);
        if (pipeline_style_lyric_timbre) {
            ace_dit::Config dit_cfg{};
            std::string dit_cfg_err;
            int32_t timbre_hidden = 64;
            if (ace_dit::load_config(dit_dir + "/config.json", dit_cfg, dit_cfg_err) && dit_cfg.timbre_hidden_dim > 0) {
                timbre_hidden = dit_cfg.timbre_hidden_dim;
            }

            std::mt19937 rng(static_cast<uint32_t>(seed));
            std::normal_distribution<float> dist(0.0f, 1.0f);
            std::vector<float> timbre_feats(
                static_cast<size_t>(timbre_rand_n) * static_cast<size_t>(timbre_len) * static_cast<size_t>(timbre_hidden));
            for (float & v : timbre_feats) {
                v = dist(rng);
            }

            st = ace_ggml_generate_audio_style_lyric_timbre_simple(
                ctx,
                style_tokens.empty() ? nullptr : style_tokens.data(),
                static_cast<int32_t>(style_tokens.size()),
                lyric_tokens.empty() ? nullptr : lyric_tokens.data(),
                static_cast<int32_t>(lyric_tokens.size()),
                timbre_feats.data(),
                timbre_order.data(),
                timbre_rand_n,
                timbre_len,
                resolved_seq_len,
                shift,
                seed,
                out.data(),
                out.size() * sizeof(float),
                nullptr,
                nullptr);
        } else if (pipeline_style_lyric) {
            st = ace_ggml_generate_audio_style_lyric_simple(
                ctx,
                style_tokens.empty() ? nullptr : style_tokens.data(),
                static_cast<int32_t>(style_tokens.size()),
                lyric_tokens.empty() ? nullptr : lyric_tokens.data(),
                static_cast<int32_t>(lyric_tokens.size()),
                resolved_seq_len,
                shift,
                seed,
                out.data(),
                out.size() * sizeof(float),
                nullptr,
                nullptr);
        } else {
            st = ace_ggml_generate_audio_simple(
                ctx,
                tokens.data(),
                static_cast<int32_t>(tokens.size()),
                resolved_seq_len,
                shift,
                seed,
                out.data(),
                out.size() * sizeof(float),
                nullptr,
                nullptr);
        }
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "pipeline failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        if (pipeline_style_lyric_timbre) {
            std::printf(
                "ok pipeline-style-lyric-timbre seq_len=%d style_tokens=%d lyric_tokens=%d timbre_refs=%d timbre_len=%d out_samples=%d out_channels=%d shift=%.3f\n",
                resolved_seq_len,
                static_cast<int32_t>(style_tokens.size()),
                static_cast<int32_t>(lyric_tokens.size()),
                timbre_rand_n,
                timbre_len,
                out_samples,
                audio_channels,
                shift);
        } else if (pipeline_style_lyric) {
            std::printf("ok pipeline-style-lyric seq_len=%d style_tokens=%d lyric_tokens=%d out_samples=%d out_channels=%d shift=%.3f\n",
                        resolved_seq_len, static_cast<int32_t>(style_tokens.size()), static_cast<int32_t>(lyric_tokens.size()),
                        out_samples, audio_channels, shift);
        } else {
            std::printf("ok pipeline seq_len=%d out_samples=%d out_channels=%d shift=%.3f\n",
                        resolved_seq_len, out_samples, audio_channels, shift);
        }
        for (int i = 0; i < 8 && i < audio_channels; ++i) {
            std::printf("%d: %.6f\n", i, out[i]);
        }
    } else if (!vae_dir.empty()) {
        ace_ggml_status st = ace_ggml_load_vae(ctx, vae_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "vae load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        int32_t latent_channels = 0;
        int32_t audio_channels = 0;
        int32_t hop_length = 0;
        st = ace_ggml_vae_get_info(ctx, &latent_channels, &audio_channels, &hop_length);
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "vae info failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::normal_distribution<float> dist(0.0f, 1.0f);
        if (vae_encode) {
            std::vector<float> audio(static_cast<size_t>(audio_len) * audio_channels);
            for (auto & v : audio) {
                v = dist(rng);
            }
            const int32_t latent_cap = (audio_len / hop_length + 4);
            std::vector<float> out(static_cast<size_t>(latent_cap) * latent_channels);
            st = ace_ggml_vae_encode(
                ctx,
                audio.data(),
                audio_len,
                out.data(),
                out.size() * sizeof(float));
            if (st != ACE_GGML_OK) {
                std::fprintf(stderr, "vae encode failed: %s\n", ace_ggml_last_error(ctx));
                ace_ggml_destroy(ctx);
                return 1;
            }
            std::printf("ok vae encode audio_len=%d audio_ch=%d latent_ch=%d hop=%d est_latent_len=%d\n",
                        audio_len, audio_channels, latent_channels, hop_length, audio_len / hop_length);
            for (int i = 0; i < 8 && i < latent_channels; ++i) {
                std::printf("%d: %.6f\n", i, out[i]);
            }
        } else {
            std::vector<float> latents(static_cast<size_t>(latent_len) * latent_channels);
            for (auto & v : latents) {
                v = dist(rng);
            }

            const int32_t audio_len_dec = latent_len * hop_length;
            std::vector<float> out(static_cast<size_t>(audio_len_dec) * audio_channels);
            st = ace_ggml_vae_decode(
                ctx,
                latents.data(),
                latent_len,
                out.data(),
                out.size() * sizeof(float));
            if (st != ACE_GGML_OK) {
                std::fprintf(stderr, "vae decode failed: %s\n", ace_ggml_last_error(ctx));
                ace_ggml_destroy(ctx);
                return 1;
            }

            std::printf("ok vae decode latent_len=%d latent_ch=%d audio_len=%d audio_ch=%d hop=%d\n",
                        latent_len, latent_channels, audio_len_dec, audio_channels, hop_length);
            for (int i = 0; i < 8 && i < audio_channels; ++i) {
                std::printf("%d: %.6f\n", i, out[i]);
            }
        }
    } else if (!dit_dir.empty()) {
        ace_dit::Config cfg{};
        std::string cfg_err;
        if (!ace_dit::load_config(dit_dir + "/config.json", cfg, cfg_err)) {
            std::fprintf(stderr, "failed to load dit config: %s\n", cfg_err.c_str());
            ace_ggml_destroy(ctx);
            return 1;
        }

        ace_ggml_status st = ace_ggml_load_dit(ctx, dit_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "dit load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        const int32_t audio_dim = cfg.audio_acoustic_hidden_dim;
        const int32_t ctx_dim = cfg.in_channels - audio_dim;
        if (ctx_dim <= 0) {
            std::fprintf(stderr, "invalid in_channels/audio_acoustic_hidden_dim\n");
            ace_ggml_destroy(ctx);
            return 1;
        }
        const int32_t hidden = cfg.hidden_size;
        if (enc_len <= 0) {
            enc_len = 1;
        }

        std::mt19937 rng(static_cast<uint32_t>(seed));
        std::normal_distribution<float> dist(0.0f, 1.0f);

        std::vector<float> hidden_states(static_cast<size_t>(seq_len) * audio_dim);
        std::vector<float> context_latents(static_cast<size_t>(seq_len) * ctx_dim);
        std::vector<float> encoder_hidden_states(static_cast<size_t>(enc_len) * hidden);

        for (auto & v : hidden_states) v = dist(rng);
        for (auto & v : context_latents) v = dist(rng);
        for (auto & v : encoder_hidden_states) v = dist(rng);

        if (mask.empty()) {
            mask.assign(seq_len, 1);
        }
        if (enc_mask.empty()) {
            enc_mask.assign(enc_len, 1);
        }

        std::vector<float> out(static_cast<size_t>(seq_len) * audio_dim);
        st = ace_ggml_dit_forward(
            ctx,
            hidden_states.data(),
            context_latents.data(),
            encoder_hidden_states.data(),
            mask.empty() ? nullptr : mask.data(),
            enc_mask.empty() ? nullptr : enc_mask.data(),
            static_cast<int32_t>(seq_len),
            static_cast<int32_t>(enc_len),
            timestep,
            timestep_r,
            out.data(),
            out.size() * sizeof(float));
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "dit forward failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        std::printf("ok dit seq_len=%d enc_len=%d hidden=%d\n", seq_len, enc_len, hidden);
        for (int i = 0; i < 8 && i < audio_dim; ++i) {
            std::printf("%d: %.6f\n", i, out[i]);
        }
    } else {
        ace_ggml_status st = ace_ggml_load_text_encoder(ctx, model_dir.c_str());
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "load failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        ace_qwen::Config qcfg{};
        std::string qerr;
        if (!ace_qwen::load_config(model_dir + "/config.json", qcfg, qerr) || qcfg.hidden_size <= 0) {
            std::fprintf(stderr, "failed to read qwen config: %s\n", qerr.c_str());
            ace_ggml_destroy(ctx);
            return 1;
        }
        const int32_t hidden = qcfg.hidden_size;
        std::vector<float> out(static_cast<size_t>(hidden) * tokens.size());

        if (embed_only) {
            layers = 0;
        }

        if (layers >= 0) {
            st = ace_ggml_text_encoder_forward_layers(
                ctx,
                tokens.data(),
                mask.empty() ? nullptr : mask.data(),
                static_cast<int32_t>(tokens.size()),
                static_cast<int32_t>(layers),
                apply_final_norm ? 1 : 0,
                out.data(),
                out.size() * sizeof(float));
        } else if (embed_only) {
            st = ace_ggml_text_encoder_forward_embeddings(
                ctx, tokens.data(), static_cast<int32_t>(tokens.size()), out.data(), out.size() * sizeof(float));
        } else if (!mask.empty()) {
            st = ace_ggml_text_encoder_forward_masked(
                ctx, tokens.data(), mask.data(), static_cast<int32_t>(tokens.size()), out.data(), out.size() * sizeof(float));
        } else {
            st = ace_ggml_text_encoder_forward(
                ctx, tokens.data(), static_cast<int32_t>(tokens.size()), out.data(), out.size() * sizeof(float));
        }
        if (st != ACE_GGML_OK) {
            std::fprintf(stderr, "forward failed: %s\n", ace_ggml_last_error(ctx));
            ace_ggml_destroy(ctx);
            return 1;
        }

        std::printf("ok n_tokens=%zu hidden=%d\n", tokens.size(), hidden);
        for (int i = 0; i < 8 && i < hidden; ++i) {
            std::printf("%d: %.6f\n", i, out[i]);
        }
    }

    ace_ggml_destroy(ctx);
    return 0;
}
