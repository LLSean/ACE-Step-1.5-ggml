#ifndef ACESTEP_GGML_H
#define ACESTEP_GGML_H

#include <stdint.h>
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

#if defined(_WIN32)
  #if defined(ACE_GGML_BUILD)
    #define ACE_GGML_API __declspec(dllexport)
  #else
    #define ACE_GGML_API __declspec(dllimport)
  #endif
#else
  #define ACE_GGML_API __attribute__((visibility("default")))
#endif

typedef struct ace_ggml_context ace_ggml_context;

typedef enum ace_ggml_status {
    ACE_GGML_OK = 0,
    ACE_GGML_ERR = 1,
    ACE_GGML_ERR_INVALID_ARG = 2,
    ACE_GGML_ERR_IO = 3,
    ACE_GGML_ERR_UNSUPPORTED = 4
} ace_ggml_status;

typedef struct ace_ggml_init_params {
    int32_t n_threads;
    int32_t use_metal;
    size_t compute_buffer_bytes;
} ace_ggml_init_params;

ACE_GGML_API ace_ggml_status ace_ggml_create(const ace_ggml_init_params * params, ace_ggml_context ** out_ctx);
ACE_GGML_API void            ace_ggml_destroy(ace_ggml_context * ctx);
ACE_GGML_API const char *    ace_ggml_last_error(const ace_ggml_context * ctx);

ACE_GGML_API ace_ggml_status ace_ggml_load_lm(ace_ggml_context * ctx, const char * model_dir);
ACE_GGML_API ace_ggml_status ace_ggml_load_text_encoder(ace_ggml_context * ctx, const char * model_dir);
ACE_GGML_API ace_ggml_status ace_ggml_load_dit(ace_ggml_context * ctx, const char * model_dir);
ACE_GGML_API ace_ggml_status ace_ggml_load_vae(ace_ggml_context * ctx, const char * model_dir);
ACE_GGML_API ace_ggml_status ace_ggml_vae_get_info(
    ace_ggml_context * ctx,
    int32_t * latent_channels,
    int32_t * audio_channels,
    int32_t * hop_length);
ACE_GGML_API ace_ggml_status ace_ggml_vae_decode(
    ace_ggml_context * ctx,
    const float * latents,
    int32_t n_frames,
    float * out,
    size_t out_size);
ACE_GGML_API ace_ggml_status ace_ggml_vae_encode(
    ace_ggml_context * ctx,
    const float * audio,
    int32_t n_samples,
    float * out,
    size_t out_size);

ACE_GGML_API ace_ggml_status ace_ggml_text_encoder_forward(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    float * out,
    size_t out_size);

ACE_GGML_API ace_ggml_status ace_ggml_text_encoder_forward_masked(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    float * out,
    size_t out_size);

ACE_GGML_API ace_ggml_status ace_ggml_text_encoder_forward_embeddings(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    float * out,
    size_t out_size);


ACE_GGML_API ace_ggml_status ace_ggml_text_encoder_forward_layers(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    const int32_t * attention_mask,
    int32_t n_tokens,
    int32_t n_layers,
    int32_t apply_final_norm,
    float * out,
    size_t out_size);

ACE_GGML_API ace_ggml_status ace_ggml_dit_forward(
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
    size_t out_size);

ACE_GGML_API ace_ggml_status ace_ggml_generate_audio_simple(
    ace_ggml_context * ctx,
    const int32_t * token_ids,
    int32_t n_tokens,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels);

ACE_GGML_API ace_ggml_status ace_ggml_generate_audio_style_lyric_simple(
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
    int32_t * out_audio_channels);

ACE_GGML_API ace_ggml_status ace_ggml_generate_audio_style_lyric_timbre_simple(
    ace_ggml_context * ctx,
    const int32_t * style_token_ids,
    int32_t n_style_tokens,
    const int32_t * lyric_token_ids,
    int32_t n_lyric_tokens,
    const float * refer_audio_acoustic_hidden_states, // [n_refer_audio, refer_audio_len, timbre_hidden_dim]
    const int32_t * refer_audio_order_mask,           // [n_refer_audio], optional, single-batch only
    int32_t n_refer_audio,
    int32_t refer_audio_len,
    int32_t seq_len,
    float shift,
    int32_t seed,
    float * out_audio,
    size_t out_size,
    int32_t * out_audio_samples,
    int32_t * out_audio_channels);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // ACESTEP_GGML_H
