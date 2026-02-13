#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="/Users/fmh/project/ACE-Step-1.5"
PYTHON_BIN="/opt/miniconda3/envs/ace-ggml-py311/bin/python"

if [[ -z "${GGML_LIB:-}" ]]; then
  if [[ -f "${PROJECT_ROOT}/acestep_ggml/build_metal/libacestep_ggml.dylib" ]]; then
    GGML_LIB="${PROJECT_ROOT}/acestep_ggml/build_metal/libacestep_ggml.dylib"
  else
    GGML_LIB="${PROJECT_ROOT}/acestep_ggml/build/libacestep_ggml.dylib"
  fi
fi

TEXT_ENCODER_GGUF="${TEXT_ENCODER_GGUF:-${PROJECT_ROOT}/checkpoints/Qwen3-Embedding-0.6B/model.q4.gguf}"
TOKENIZER_DIR="${TOKENIZER_DIR:-${PROJECT_ROOT}/checkpoints/Qwen3-Embedding-0.6B}"
TEXT_ENCODER_DIR="${TEXT_ENCODER_DIR:-${PROJECT_ROOT}/checkpoints/Qwen3-Embedding-0.6B/model.q8.gguf}"
DIT_DIR="${DIT_DIR:-${PROJECT_ROOT}/checkpoints/acestep-v15-turbo}"
# Optional pre-quantized gguf path for DiT (if empty, loader uses safetensors/default files in DIT_DIR).
DIT_GGUF="${DIT_GGUF:-}"
# Optional gguf path for VAE (if empty, loader uses safetensors/default files in VAE_DIR).
VAE_GGUF="${VAE_GGUF:-}"
VAE_DIR="${VAE_DIR:-${PROJECT_ROOT}/checkpoints}"
OUT_DIR="${OUT_DIR:-${PROJECT_ROOT}/outputs/step_textenc_gguf_q8_dit_ggml_vae_ggml}"
RUN_MODE="${RUN_MODE:-python}" # python | cli

# Default to ggml-capi VAE, but keep VAE Metal decode disabled for stability.
VAE_BACKEND="${VAE_BACKEND:-ggml-capi}"
# VAE_BACKEND="${VAE_BACKEND:-python}"
GGML_THREADS="${GGML_THREADS:-4}"
GGML_COMPUTE_BUFFER_MB="${GGML_COMPUTE_BUFFER_MB:-10240}"
GGML_USE_METAL="${GGML_USE_METAL:-1}"
GGML_DIT_METAL_FORWARD="${GGML_DIT_METAL_FORWARD:-off}"
GGML_VAE_METAL_DECODE="${GGML_VAE_METAL_DECODE:-off}"
GGML_VAE_METAL_WEIGHT_MAP="${GGML_VAE_METAL_WEIGHT_MAP:-auto}"
GGML_VAE_METAL_MIN_FREE_MB="${GGML_VAE_METAL_MIN_FREE_MB:-2048}"
GGML_VAE_METAL_REQUIRE_WEIGHT_MAP="${GGML_VAE_METAL_REQUIRE_WEIGHT_MAP:-1}"
# Safer kernel path when manually enabling VAE Metal decode.
GGML_VAE_TRANSPOSE_CONV_F32="${GGML_VAE_TRANSPOSE_CONV_F32:-1}"
GGML_ALLOW_UNSAFE_VAE_METAL="${GGML_ALLOW_UNSAFE_VAE_METAL:-0}"
GGML_VAE_CHUNK_SIZE="${GGML_VAE_CHUNK_SIZE:-64}"
GGML_VAE_OVERLAP="${GGML_VAE_OVERLAP:-8}"
GGML_VAE_ENCODE_CHUNK_FRAMES="${GGML_VAE_ENCODE_CHUNK_FRAMES:-64}"
GGML_VAE_PROFILE="${GGML_VAE_PROFILE:-0}"
GGML_VAE_CHUNK_PROFILE="${GGML_VAE_CHUNK_PROFILE:-0}"

# CLI mode args (RUN_MODE=cli)
CLI_BIN="${CLI_BIN:-${PROJECT_ROOT}/acestep_ggml/build_metal/ace_ggml_cli}"
CLI_MODE="${CLI_MODE:-pipeline-style-lyric}" # pipeline | pipeline-style-lyric
CLI_SEQ_LEN="${CLI_SEQ_LEN:-}"
CLI_DURATION="${CLI_DURATION:-30}"
CLI_SHIFT="${CLI_SHIFT:-3}"
CLI_SEED="${CLI_SEED:-1234}"
CLI_OUT_WAV="${CLI_OUT_WAV:-${OUT_DIR}/cli_output.wav}"
TOKENS="${TOKENS:-}"
TOKENS_FILE="${TOKENS_FILE:-}"
STYLE_TOKENS="${STYLE_TOKENS:-}"
STYLE_TOKENS_FILE="${STYLE_TOKENS_FILE:-}"
LYRIC_TOKENS="${LYRIC_TOKENS:-}"
LYRIC_TOKENS_FILE="${LYRIC_TOKENS_FILE:-}"
STYLE_FILE="${STYLE_FILE:-${PROJECT_ROOT}/acestep_ggml/reports/real_case/style_neo_soul.txt}"
LYRIC_FILE="${LYRIC_FILE:-${PROJECT_ROOT}/acestep_ggml/reports/real_case/lyric_neo_soul.txt}"
CAPTION="${CAPTION:-}"
LYRICS="${LYRICS:-}"
LANGUAGE="${LANGUAGE:-zh}"
BPM="${BPM:-}"
KEYSCALE="${KEYSCALE:-}"
TIMESIGNATURE="${TIMESIGNATURE:-}"
INSTRUCTION="${INSTRUCTION:-Fill the audio semantic mask based on the given conditions:}"
STYLE_MAX_LEN="${STYLE_MAX_LEN:-256}"
LYRIC_MAX_LEN="${LYRIC_MAX_LEN:-2048}"
CLI_TOKEN_PREPARE_SCRIPT="${CLI_TOKEN_PREPARE_SCRIPT:-${PROJECT_ROOT}/scripts/build_cli_token_files.py}"

if [[ "${VAE_BACKEND}" != "ggml-capi" ]]; then
  GGML_VAE_METAL_DECODE="off"
fi
if [[ "${GGML_ALLOW_UNSAFE_VAE_METAL}" != "1" && "${GGML_VAE_METAL_DECODE}" != "off" ]]; then
  echo "[run.sh] forcing GGML_VAE_METAL_DECODE=off for stability (set GGML_ALLOW_UNSAFE_VAE_METAL=1 to override)"
  GGML_VAE_METAL_DECODE="off"
fi

echo "[run.sh] backends: text=ggml-capi dit=ggml-capi vae=${VAE_BACKEND}"
echo "[run.sh] metal flags: use=${GGML_USE_METAL} dit_forward=${GGML_DIT_METAL_FORWARD} vae_decode=${GGML_VAE_METAL_DECODE} vae_weight_map=${GGML_VAE_METAL_WEIGHT_MAP}"
echo "[run.sh] vae tiling: chunk_size=${GGML_VAE_CHUNK_SIZE} overlap=${GGML_VAE_OVERLAP}"
echo "[run.sh] vae encode chunk_frames=${GGML_VAE_ENCODE_CHUNK_FRAMES}"
echo "[run.sh] run mode: ${RUN_MODE}"
echo "[run.sh] text gguf: ${TEXT_ENCODER_GGUF}"
if [[ -n "${DIT_GGUF}" ]]; then
  echo "[run.sh] dit gguf: ${DIT_GGUF}"
else
  echo "[run.sh] dit gguf: <disabled> (using DIT_DIR files)"
fi
if [[ -n "${VAE_GGUF}" ]]; then
  echo "[run.sh] vae gguf: ${VAE_GGUF}"
else
  echo "[run.sh] vae gguf: <disabled> (using VAE_DIR files)"
fi
echo "[run.sh] ggml lib: ${GGML_LIB}"

EXTRA_METAL_FLAG=""
if [[ "${GGML_USE_METAL}" == "1" ]]; then
  EXTRA_METAL_FLAG="--ggml-use-metal"
fi

EXTRA_VAE_PROFILE_FLAGS=""
if [[ "${GGML_VAE_PROFILE}" == "1" ]]; then
  EXTRA_VAE_PROFILE_FLAGS="${EXTRA_VAE_PROFILE_FLAGS} --ggml-vae-profile"
fi
if [[ "${GGML_VAE_CHUNK_PROFILE}" == "1" ]]; then
  EXTRA_VAE_PROFILE_FLAGS="${EXTRA_VAE_PROFILE_FLAGS} --ggml-vae-chunk-profile"
fi

run_with_common_env() {
  env OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 NUMEXPR_NUM_THREADS=1 \
  ACE_GGML_TEXT_ENCODER_GGUF="${TEXT_ENCODER_GGUF}" \
  ACE_GGML_QWEN_GGUF="${TEXT_ENCODER_GGUF}" \
  ACE_GGML_DIT_GGUF="${DIT_GGUF}" \
  ACE_GGML_DIT_GGUF_PATH="${DIT_GGUF}" \
  ACE_GGML_VAE_GGUF="${VAE_GGUF}" \
  ACE_GGML_VAE_GGUF_PATH="${VAE_GGUF}" \
  ACE_GGML_DIT_METAL_FORWARD="${GGML_DIT_METAL_FORWARD}" \
  ACE_GGML_VAE_METAL_DECODE="${GGML_VAE_METAL_DECODE}" \
  ACE_GGML_VAE_METAL_WEIGHT_MAP="${GGML_VAE_METAL_WEIGHT_MAP}" \
  ACE_GGML_VAE_METAL_MIN_FREE_MB="${GGML_VAE_METAL_MIN_FREE_MB}" \
  ACE_GGML_VAE_METAL_REQUIRE_WEIGHT_MAP="${GGML_VAE_METAL_REQUIRE_WEIGHT_MAP}" \
  ACE_GGML_ALLOW_UNSAFE_VAE_METAL="${GGML_ALLOW_UNSAFE_VAE_METAL}" \
  ACE_GGML_VAE_TRANSPOSE_CONV_F32="${GGML_VAE_TRANSPOSE_CONV_F32}" \
  ACE_GGML_VAE_CHUNK_FRAMES="${GGML_VAE_CHUNK_SIZE}" \
  ACE_GGML_VAE_CHUNK_OVERLAP_FRAMES="${GGML_VAE_OVERLAP}" \
  ACE_GGML_VAE_ENCODE_CHUNK_FRAMES="${GGML_VAE_ENCODE_CHUNK_FRAMES}" \
  "$@"
}

CLI_TOKEN_TMP_DIR=""
cleanup_cli_tokens() {
  if [[ -n "${CLI_TOKEN_TMP_DIR}" && -d "${CLI_TOKEN_TMP_DIR}" ]]; then
    rm -rf "${CLI_TOKEN_TMP_DIR}"
  fi
}
trap cleanup_cli_tokens EXIT

if [[ "${RUN_MODE}" == "cli" ]]; then
  if [[ ! -x "${CLI_BIN}" ]]; then
    echo "[run.sh] error: cli binary not found or not executable: ${CLI_BIN}"
    exit 1
  fi
  mkdir -p "${OUT_DIR}"
  mkdir -p "$(dirname "${CLI_OUT_WAV}")"

  CLI_METAL_ARG=""
  if [[ "${GGML_USE_METAL}" == "1" ]]; then
    CLI_METAL_ARG="--use-metal"
  fi

  if [[ "${CLI_MODE}" == "pipeline" ]]; then
    CLI_TOKEN_ARG=""
    if [[ -n "${TOKENS_FILE}" ]]; then
      CLI_TOKEN_ARG="--tokens-file ${TOKENS_FILE}"
    elif [[ -n "${TOKENS}" ]]; then
      CLI_TOKEN_ARG="--tokens ${TOKENS}"
    else
      echo "[run.sh] error: CLI_MODE=pipeline needs TOKENS or TOKENS_FILE"
      exit 1
    fi
    CLI_LEN_ARG="--audio-seconds ${CLI_DURATION}"
    if [[ -n "${CLI_SEQ_LEN}" ]]; then
      CLI_LEN_ARG="--seq-len ${CLI_SEQ_LEN}"
    fi
    # shellcheck disable=SC2086
    run_with_common_env "${CLI_BIN}" \
      --pipeline \
      --text-encoder "${TEXT_ENCODER_DIR}" \
      --dit "${DIT_DIR}" \
      --vae "${VAE_DIR}" \
      ${CLI_TOKEN_ARG} \
      ${CLI_LEN_ARG} \
      --shift "${CLI_SHIFT}" \
      --seed "${CLI_SEED}" \
      --threads "${GGML_THREADS}" \
      --compute-buffer-mb "${GGML_COMPUTE_BUFFER_MB}" \
      ${CLI_METAL_ARG} \
      --out-wav "${CLI_OUT_WAV}"
  elif [[ "${CLI_MODE}" == "pipeline-style-lyric" ]]; then
    CLI_STYLE_ARG=""
    CLI_LYRIC_ARG=""
    if [[ -n "${STYLE_TOKENS_FILE}" || -n "${STYLE_TOKENS}" || -n "${LYRIC_TOKENS_FILE}" || -n "${LYRIC_TOKENS}" ]]; then
      if [[ -n "${STYLE_TOKENS_FILE}" ]]; then
        CLI_STYLE_ARG="--style-tokens-file ${STYLE_TOKENS_FILE}"
      elif [[ -n "${STYLE_TOKENS}" ]]; then
        CLI_STYLE_ARG="--style-tokens ${STYLE_TOKENS}"
      else
        echo "[run.sh] error: style tokens missing"
        exit 1
      fi
      if [[ -n "${LYRIC_TOKENS_FILE}" ]]; then
        CLI_LYRIC_ARG="--lyric-tokens-file ${LYRIC_TOKENS_FILE}"
      elif [[ -n "${LYRIC_TOKENS}" ]]; then
        CLI_LYRIC_ARG="--lyric-tokens ${LYRIC_TOKENS}"
      else
        echo "[run.sh] error: lyric tokens missing"
        exit 1
      fi
    else
      if [[ ! -f "${CLI_TOKEN_PREPARE_SCRIPT}" ]]; then
        echo "[run.sh] error: token prepare script not found: ${CLI_TOKEN_PREPARE_SCRIPT}"
        exit 1
      fi
      CLI_TOKEN_TMP_DIR="$(mktemp -d "${TMPDIR:-/tmp}/ace_cli_tokens.XXXXXX")"
      AUTO_STYLE_TOKENS_FILE="${CLI_TOKEN_TMP_DIR}/style_tokens.txt"
      AUTO_LYRIC_TOKENS_FILE="${CLI_TOKEN_TMP_DIR}/lyric_tokens.txt"
      echo "[run.sh] auto-tokenizing style/lyric with tokenizer.json..."
      "${PYTHON_BIN}" "${CLI_TOKEN_PREPARE_SCRIPT}" \
        --project-root "${PROJECT_ROOT}" \
        --tokenizer-dir "${TOKENIZER_DIR}" \
        --style-file "${STYLE_FILE}" \
        --lyric-file "${LYRIC_FILE}" \
        --caption "${CAPTION}" \
        --lyrics "${LYRICS}" \
        --language "${LANGUAGE}" \
        --instruction "${INSTRUCTION}" \
        --bpm "${BPM}" \
        --keyscale "${KEYSCALE}" \
        --timesignature "${TIMESIGNATURE}" \
        --duration "${CLI_DURATION}" \
        --style-max-len "${STYLE_MAX_LEN}" \
        --lyric-max-len "${LYRIC_MAX_LEN}" \
        --out-style-tokens-file "${AUTO_STYLE_TOKENS_FILE}" \
        --out-lyric-tokens-file "${AUTO_LYRIC_TOKENS_FILE}" \
        --vae-dir "${VAE_DIR}" \
        --sample-rate 48000
      CLI_STYLE_ARG="--style-tokens-file ${AUTO_STYLE_TOKENS_FILE}"
      CLI_LYRIC_ARG="--lyric-tokens-file ${AUTO_LYRIC_TOKENS_FILE}"
    fi
    CLI_LEN_ARG="--audio-seconds ${CLI_DURATION}"
    if [[ -n "${CLI_SEQ_LEN}" ]]; then
      CLI_LEN_ARG="--seq-len ${CLI_SEQ_LEN}"
    fi
    # shellcheck disable=SC2086
    run_with_common_env "${CLI_BIN}" \
      --pipeline-style-lyric \
      --text-encoder "${TEXT_ENCODER_DIR}" \
      --dit "${DIT_DIR}" \
      --vae "${VAE_DIR}" \
      ${CLI_STYLE_ARG} \
      ${CLI_LYRIC_ARG} \
      ${CLI_LEN_ARG} \
      --shift "${CLI_SHIFT}" \
      --seed "${CLI_SEED}" \
      --threads "${GGML_THREADS}" \
      --compute-buffer-mb "${GGML_COMPUTE_BUFFER_MB}" \
      ${CLI_METAL_ARG} \
      --out-wav "${CLI_OUT_WAV}"
  else
    echo "[run.sh] error: unsupported CLI_MODE=${CLI_MODE} (use pipeline or pipeline-style-lyric)"
    exit 1
  fi
else
  # shellcheck disable=SC2086
  run_with_common_env "${PYTHON_BIN}" "${PROJECT_ROOT}/scripts/run_non_ggml_real_case.py" \
    --device auto --config-path acestep-v15-turbo \
    --duration 30 --inference-steps 8 --shift 3 --seed 1234 \
    --tokenizer-backend ggml-json \
    --tokenizer-dir "${TOKENIZER_DIR}" \
    --text-encoder-backend ggml-capi \
    --ggml-text-encoder-dir "${TEXT_ENCODER_DIR}" \
    --dit-backend ggml-capi \
    --ggml-dit-dir "${DIT_DIR}" \
    --vae-backend "${VAE_BACKEND}" \
    --ggml-vae-dir "${VAE_DIR}" \
    --ggml-lib "${GGML_LIB}" \
    --ggml-threads "${GGML_THREADS}" \
    --ggml-compute-buffer-mb "${GGML_COMPUTE_BUFFER_MB}" \
    --ggml-vae-chunk-size "${GGML_VAE_CHUNK_SIZE}" \
    --ggml-vae-overlap "${GGML_VAE_OVERLAP}" \
    ${EXTRA_VAE_PROFILE_FLAGS} \
    ${EXTRA_METAL_FLAG} \
    --audio-format wav --out-dir "${OUT_DIR}"
fi
