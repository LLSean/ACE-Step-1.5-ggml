#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import os
import pathlib
import re
import wave

import numpy as np


ACE_GGML_OK = 0
DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"


class AceInitParams(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int32),
        ("use_metal", ctypes.c_int32),
        ("compute_buffer_bytes", ctypes.c_size_t),
    ]


AceCtxPtr = ctypes.c_void_p


def bind_lib(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    lib.ace_ggml_create.argtypes = [ctypes.POINTER(AceInitParams), ctypes.POINTER(AceCtxPtr)]
    lib.ace_ggml_create.restype = ctypes.c_int
    lib.ace_ggml_destroy.argtypes = [AceCtxPtr]
    lib.ace_ggml_destroy.restype = None
    lib.ace_ggml_last_error.argtypes = [AceCtxPtr]
    lib.ace_ggml_last_error.restype = ctypes.c_char_p

    lib.ace_ggml_load_text_encoder.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_text_encoder.restype = ctypes.c_int
    lib.ace_ggml_load_dit.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_dit.restype = ctypes.c_int
    lib.ace_ggml_load_vae.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_vae.restype = ctypes.c_int
    lib.ace_ggml_vae_get_info.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ace_ggml_vae_get_info.restype = ctypes.c_int
    lib.ace_ggml_generate_audio_style_lyric_simple.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ace_ggml_generate_audio_style_lyric_simple.restype = ctypes.c_int
    lib.ace_ggml_generate_audio_style_lyric_timbre_simple.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ace_ggml_generate_audio_style_lyric_timbre_simple.restype = ctypes.c_int
    return lib


def lib_error(lib, ctx: AceCtxPtr) -> str:
    msg = lib.ace_ggml_last_error(ctx)
    return msg.decode("utf-8", errors="replace") if msg else "unknown error"


def ensure_ok(lib, ctx: AceCtxPtr, status: int, step: str) -> None:
    if status != ACE_GGML_OK:
        raise RuntimeError(f"{step} failed: {lib_error(lib, ctx)} (status={status})")


def write_wav(path: pathlib.Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.clip(audio, -1.0, 1.0)
    pcm = (pcm * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(audio.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def format_style_prompt(
    style_text: str,
    instruction: str,
    bpm: str,
    timesignature: str,
    keyscale: str,
    duration: str,
) -> str:
    metas = (
        f"- bpm: {bpm}\n"
        f"- timesignature: {timesignature}\n"
        f"- keyscale: {keyscale}\n"
        f"- duration: {duration}\n"
    )
    return (
        "# Instruction\n"
        f"{instruction}\n\n"
        "# Caption\n"
        f"{style_text}\n\n"
        "# Metas\n"
        f"{metas}<|endoftext|>\n"
    )


def format_lyric_prompt(lyric_text: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyric_text}<|endoftext|>"


def parse_duration_seconds(value: str) -> float | None:
    text = (value or "").strip()
    if not text:
        return None
    m = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*([a-zA-Z]+)?", text)
    if not m:
        return None
    num = float(m.group(1))
    if num <= 0:
        return None
    unit = (m.group(2) or "s").lower()
    if unit.startswith("m"):
        return num * 60.0
    return num


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run style+lyric dual-input pipeline with tokenizer + ggml C API.")
    p.add_argument("--model-root", default="/Users/fmh/project/ACE-Step-1.5/Ace-Step1.5")
    p.add_argument("--text-encoder-dir", default="")
    p.add_argument("--dit-dir", default="")
    p.add_argument("--vae-dir", default="")
    p.add_argument("--tokenizer-dir", default="", help="Tokenizer dir for style/lyric text (defaults to text-encoder dir for parity).")
    p.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")

    p.add_argument("--style-text", required=True)
    p.add_argument("--lyric-text", required=True)
    p.add_argument("--instruction", default=DEFAULT_DIT_INSTRUCTION)
    p.add_argument("--language", default="en")
    p.add_argument("--bpm", default="N/A")
    p.add_argument("--timesignature", default="N/A")
    p.add_argument("--keyscale", default="N/A")
    p.add_argument("--duration", default="30 seconds")
    p.add_argument("--audio-seconds", type=float, default=None, help="target output length in seconds")

    p.add_argument("--style-max-len", type=int, default=256)
    p.add_argument("--lyric-max-len", type=int, default=2048)
    p.add_argument("--seq-len", type=int, default=0, help="latent sequence length; <=0 enables auto conversion from duration/seconds")
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--compute-buffer-mb", type=int, default=1024)
    p.add_argument("--graph-size", type=int, default=0, help="set ACE_GGML_DIT_GRAPH_SIZE/ACE_GGML_GRAPH_SIZE")
    p.add_argument("--debug-graph", action="store_true", help="enable ACE_GGML_DEBUG_GRAPH=1")
    p.add_argument("--timbre-npy", default="")
    p.add_argument("--timbre-order", default="")
    p.add_argument("--timbre-rand-n", type=int, default=0)
    p.add_argument("--timbre-len", type=int, default=0)
    p.add_argument("--timbre-hidden", type=int, default=64)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--out-wav", default="/tmp/ace_style_lyric.wav")
    return p.parse_args()


def encode_with_local_tokenizer(tokenizer_dir: pathlib.Path, text: str, max_len: int) -> list[int]:
    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.exists():
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(tok_json))
        ids = tok.encode(text).ids
        if max_len > 0 and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    from transformers import AutoTokenizer  # Lazy import to avoid torch path when unnecessary

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)


def main() -> int:
    args = parse_args()
    if args.debug_graph:
        os.environ["ACE_GGML_DEBUG_GRAPH"] = "1"
    if args.graph_size > 0:
        os.environ["ACE_GGML_DIT_GRAPH_SIZE"] = str(args.graph_size)
        os.environ["ACE_GGML_GRAPH_SIZE"] = str(args.graph_size)

    model_root = pathlib.Path(args.model_root)
    text_encoder_dir = pathlib.Path(args.text_encoder_dir) if args.text_encoder_dir else model_root / "Qwen3-Embedding-0.6B"
    dit_dir = pathlib.Path(args.dit_dir) if args.dit_dir else model_root / "acestep-v15-turbo"
    vae_dir = pathlib.Path(args.vae_dir) if args.vae_dir else model_root
    tokenizer_dir = pathlib.Path(args.tokenizer_dir) if args.tokenizer_dir else text_encoder_dir

    style_prompt = format_style_prompt(
        style_text=args.style_text,
        instruction=args.instruction,
        bpm=args.bpm,
        timesignature=args.timesignature,
        keyscale=args.keyscale,
        duration=args.duration,
    )
    lyric_prompt = format_lyric_prompt(args.lyric_text, args.language)

    style_ids = encode_with_local_tokenizer(tokenizer_dir, style_prompt, args.style_max_len)
    lyric_ids = encode_with_local_tokenizer(tokenizer_dir, lyric_prompt, args.lyric_max_len)
    if len(style_ids) == 0 or len(lyric_ids) == 0:
        raise RuntimeError("style or lyric tokenized to empty sequence")

    style_arr = np.asarray(style_ids, dtype=np.int32)
    lyric_arr = np.asarray(lyric_ids, dtype=np.int32)
    print(f"[tokenizer] style_tokens={style_arr.size} lyric_tokens={lyric_arr.size}")
    print(f"[tokenizer] dir={tokenizer_dir}")
    print(f"[tokenizer] style_first_32={style_arr[:32].tolist()}")
    print(f"[tokenizer] lyric_first_32={lyric_arr[:32].tolist()}")

    timbre_feat = None
    timbre_order = None
    if args.timbre_npy:
        timbre_feat = np.load(args.timbre_npy)
        if timbre_feat.ndim == 2:
            timbre_feat = timbre_feat[None, ...]
        if timbre_feat.ndim != 3:
            raise RuntimeError("timbre_npy must have shape [T,D] or [N,T,D]")
        timbre_feat = np.asarray(timbre_feat, dtype=np.float32)
    elif args.timbre_rand_n > 0 and args.timbre_len > 0:
        rng = np.random.default_rng(args.seed)
        timbre_feat = rng.standard_normal(
            (int(args.timbre_rand_n), int(args.timbre_len), int(args.timbre_hidden)),
            dtype=np.float32,
        )

    if timbre_feat is not None:
        n_refer = int(timbre_feat.shape[0])
        if args.timbre_order:
            order = [int(x) for x in args.timbre_order.split(",") if x.strip()]
            if len(order) != n_refer:
                raise RuntimeError("timbre-order length must match timbre references")
            timbre_order = np.asarray(order, dtype=np.int32)
        else:
            timbre_order = np.zeros((n_refer,), dtype=np.int32)
        print(
            f"[timbre] refs={timbre_feat.shape[0]} len={timbre_feat.shape[1]} dim={timbre_feat.shape[2]} "
            f"order={timbre_order.tolist()}"
        )

    lib = bind_lib(args.lib)
    ctx = AceCtxPtr()
    params = AceInitParams(
        n_threads=args.threads,
        use_metal=0,
        compute_buffer_bytes=args.compute_buffer_mb * 1024 * 1024,
    )
    ensure_ok(lib, ctx, lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx)), "ace_ggml_create")
    try:
        ensure_ok(lib, ctx, lib.ace_ggml_load_text_encoder(ctx, str(text_encoder_dir).encode("utf-8")), "load_text_encoder")
        ensure_ok(lib, ctx, lib.ace_ggml_load_dit(ctx, str(dit_dir).encode("utf-8")), "load_dit")
        ensure_ok(lib, ctx, lib.ace_ggml_load_vae(ctx, str(vae_dir).encode("utf-8")), "load_vae")

        latent_ch = ctypes.c_int32(0)
        audio_ch = ctypes.c_int32(0)
        hop = ctypes.c_int32(0)
        ensure_ok(lib, ctx, lib.ace_ggml_vae_get_info(ctx, ctypes.byref(latent_ch), ctypes.byref(audio_ch), ctypes.byref(hop)), "vae_get_info")

        resolved_seq_len = int(args.seq_len)
        resolved_from = "seq-len"
        if resolved_seq_len <= 0:
            target_seconds: float | None = None
            if args.audio_seconds is not None and args.audio_seconds > 0:
                target_seconds = float(args.audio_seconds)
                resolved_from = "audio-seconds"
            elif args.duration.strip().lower() != "30 seconds":
                parsed = parse_duration_seconds(args.duration)
                if parsed is not None:
                    target_seconds = parsed
                    resolved_from = "duration"
            if target_seconds is None:
                resolved_seq_len = 4
                resolved_from = "fallback"
            else:
                resolved_seq_len = max(1, int(round(target_seconds * args.sample_rate / float(hop.value))))
        resolved_seconds = float(resolved_seq_len * hop.value) / float(args.sample_rate)
        print(
            f"[length] source={resolved_from} seq_len={resolved_seq_len} seconds≈{resolved_seconds:.2f} "
            f"(hop={hop.value}, sample_rate={args.sample_rate})"
        )

        out_samples = resolved_seq_len * hop.value
        out_channels = audio_ch.value
        out_audio = np.zeros(out_samples * out_channels, dtype=np.float32)
        ret_samples = ctypes.c_int32(0)
        ret_channels = ctypes.c_int32(0)

        if timbre_feat is not None:
            ensure_ok(
                lib,
                ctx,
                lib.ace_ggml_generate_audio_style_lyric_timbre_simple(
                    ctx,
                    style_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_int32(style_arr.size),
                    lyric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_int32(lyric_arr.size),
                    timbre_feat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    timbre_order.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_int32(timbre_feat.shape[0]),
                    ctypes.c_int32(timbre_feat.shape[1]),
                    ctypes.c_int32(resolved_seq_len),
                    ctypes.c_float(args.shift),
                    ctypes.c_int32(args.seed),
                    out_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_size_t(out_audio.nbytes),
                    ctypes.byref(ret_samples),
                    ctypes.byref(ret_channels),
                ),
                "generate_audio_style_lyric_timbre_simple",
            )
        else:
            ensure_ok(
                lib,
                ctx,
                lib.ace_ggml_generate_audio_style_lyric_simple(
                    ctx,
                    style_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_int32(style_arr.size),
                    lyric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    ctypes.c_int32(lyric_arr.size),
                    ctypes.c_int32(resolved_seq_len),
                    ctypes.c_float(args.shift),
                    ctypes.c_int32(args.seed),
                    out_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    ctypes.c_size_t(out_audio.nbytes),
                    ctypes.byref(ret_samples),
                    ctypes.byref(ret_channels),
                ),
                "generate_audio_style_lyric_simple",
            )

        if ret_samples.value > 0 and ret_channels.value > 0:
            out_samples = ret_samples.value
            out_channels = ret_channels.value
        out_audio = out_audio[: out_samples * out_channels].reshape(out_samples, out_channels)
        out_wav = pathlib.Path(args.out_wav)
        write_wav(out_wav, out_audio, args.sample_rate)

        print(
            f"[ok] seq_len={resolved_seq_len} out_samples={out_samples} out_channels={out_channels} "
            f"shift={args.shift} wav={out_wav}"
        )
        print(f"[audio] first_8={out_audio.reshape(-1)[:8].tolist()}")
        return 0
    finally:
        lib.ace_ggml_destroy(ctx)


if __name__ == "__main__":
    raise SystemExit(main())
