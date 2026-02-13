#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
GGML_SUBMODULE_DIR="${ROOT_DIR}/acestep_ggml/third_party/ggml"
BUILD_DIR="${BUILD_DIR:-${ROOT_DIR}/acestep_ggml/build}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
ENABLE_METAL="${ENABLE_METAL:-OFF}"
SKIP_BUILD="${SKIP_BUILD:-0}"

MODEL_ROOT="${MODEL_ROOT:-${ROOT_DIR}/Ace-Step1.5}"
TEXT_ENCODER_GGUF="${TEXT_ENCODER_GGUF:-}"
DIT_GGUF="${DIT_GGUF:-}"
TOKENIZER_DIR="${TOKENIZER_DIR:-}"
LIB_EXT="${LIB_EXT:-}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

STYLE_TEXT="${STYLE_TEXT:-$(cat <<'EOF'
Dreamy synth-pop with warm pads, soft bass, and a gentle kick.
Airy female vocal, intimate and calm, like neon lights at night.
EOF
)}"
LYRIC_TEXT="${LYRIC_TEXT:-$(cat <<'EOF'
Neon rain on empty streets,
I follow the light where the silence meets.
EOF
)}"
LANGUAGE="${LANGUAGE:-en}"
AUDIO_SECONDS="${AUDIO_SECONDS:-10}"
DURATION="${DURATION:-${AUDIO_SECONDS} seconds}"
SHIFT="${SHIFT:-3}"
SEED="${SEED:-1234}"
THREADS="${THREADS:-8}"
COMPUTE_BUFFER_MB="${COMPUTE_BUFFER_MB:-12288}"

VAE_CHUNK_FRAMES="${VAE_CHUNK_FRAMES:-72}"
VAE_CHUNK_FRAMES_AUTO="${VAE_CHUNK_FRAMES_AUTO:-72}"
VAE_CHUNK_OVERLAP_FRAMES="${VAE_CHUNK_OVERLAP_FRAMES:-18}"
DEBUG_GENERATE_STATS="${DEBUG_GENERATE_STATS:-0}"
DEBUG_CONTEXT="${DEBUG_CONTEXT:-0}"
SILENCE_LATENT_F32="${SILENCE_LATENT_F32:-}"
AUTO_SILENCE_TIMBRE="${AUTO_SILENCE_TIMBRE:-on}"

OUT_DIR="${OUT_DIR:-${ROOT_DIR}/acestep_ggml/reports/outputs}"
OUT_NAME="${OUT_NAME:-ggml_10s_style_lyric_timbre}"

if [[ ! -f "${GGML_SUBMODULE_DIR}/CMakeLists.txt" ]]; then
  echo "missing ggml submodule: ${GGML_SUBMODULE_DIR}" >&2
  echo "run: git submodule update --init --recursive -- acestep_ggml/third_party/ggml" >&2
  exit 1
fi

if [[ "${SKIP_BUILD}" != "1" ]]; then
  echo "[stage] configure+build start $(date +%H:%M:%S)"
  cmake -S "${ROOT_DIR}/acestep_ggml" \
    -B "${BUILD_DIR}" \
    -DCMAKE_BUILD_TYPE="${BUILD_TYPE}" \
    -DACE_GGML_ENABLE_METAL="${ENABLE_METAL}"
  cmake --build "${BUILD_DIR}" --config "${BUILD_TYPE}" --target acestep_ggml
  echo "[stage] configure+build done  $(date +%H:%M:%S)"
else
  echo "[stage] skip build enabled      $(date +%H:%M:%S)"
fi

if [[ -z "${LIB_EXT}" ]]; then
  case "$(uname -s)" in
    Darwin) LIB_EXT="dylib" ;;
    MINGW*|MSYS*|CYGWIN*) LIB_EXT="dll" ;;
    *) LIB_EXT="so" ;;
  esac
fi
LIB_PATH="${BUILD_DIR}/libacestep_ggml.${LIB_EXT}"
if [[ ! -f "${LIB_PATH}" ]]; then
  echo "lib not found: ${LIB_PATH}" >&2
  exit 1
fi

export PYTHONUNBUFFERED=1
export ACE_GGML_DEBUG_GENERATE_STATS="${DEBUG_GENERATE_STATS}"
export ACE_GGML_DEBUG_CONTEXT="${DEBUG_CONTEXT}"
if [[ -n "${VAE_CHUNK_FRAMES}" ]]; then
  export ACE_GGML_VAE_CHUNK_FRAMES="${VAE_CHUNK_FRAMES}"
else
  export ACE_GGML_VAE_CHUNK_FRAMES_AUTO="${VAE_CHUNK_FRAMES_AUTO}"
fi
export ACE_GGML_VAE_CHUNK_OVERLAP_FRAMES="${VAE_CHUNK_OVERLAP_FRAMES}"
if [[ -n "${SILENCE_LATENT_F32}" ]]; then
  export ACE_GGML_SILENCE_LATENT_F32="${SILENCE_LATENT_F32}"
fi

mkdir -p "${OUT_DIR}"
OUT_WAV="${OUT_DIR}/${OUT_NAME}.wav"
OUT_JSON="${OUT_DIR}/${OUT_NAME}.json"
OUT_LOG="${OUT_DIR}/${OUT_NAME}.log"

echo "[stage] generation start        $(date +%H:%M:%S)"
"${PYTHON_BIN}" "${ROOT_DIR}/acestep_ggml/tools/run_unified_prompt_style_lyric_timbre.py" \
  --model-root "${MODEL_ROOT}" \
  --lib "${LIB_PATH}" \
  --tokenizer-dir "${TOKENIZER_DIR}" \
  --text-encoder-gguf "${TEXT_ENCODER_GGUF}" \
  --dit-gguf "${DIT_GGUF}" \
  --style-text "${STYLE_TEXT}" \
  --lyric-text "${LYRIC_TEXT}" \
  --language "${LANGUAGE}" \
  --duration "${DURATION}" \
  --seq-len 0 \
  --audio-seconds "${AUDIO_SECONDS}" \
  --shift "${SHIFT}" \
  --seed "${SEED}" \
  --threads "${THREADS}" \
  --compute-buffer-mb "${COMPUTE_BUFFER_MB}" \
  --auto-silence-timbre "${AUTO_SILENCE_TIMBRE}" \
  --out-wav "${OUT_WAV}" \
  --dump-json "${OUT_JSON}" \
  2>&1 | tee "${OUT_LOG}"
echo "[stage] generation done         $(date +%H:%M:%S)"
