#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import os
import pathlib
import re
import wave
import zipfile

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


def encode_with_local_tokenizer(tokenizer_dir: pathlib.Path, text: str, max_len: int) -> list[int]:
    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.exists():
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(tok_json))
        ids = tok.encode(text).ids
        if max_len > 0 and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    from transformers import AutoTokenizer  # Lazy import

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)


def parse_prompt_tags(prompt: str) -> tuple[str, str]:
    if not prompt.strip():
        return "", ""

    def _find(pattern: str) -> str:
        m = re.search(pattern, prompt, flags=re.IGNORECASE | re.DOTALL)
        return m.group(1).strip() if m else ""

    style = _find(r"<(?:prompt|style|caption)>\s*(.*?)\s*</(?:prompt|style|caption)>")
    lyric = _find(r"<(?:lyrics|lyric)>\s*(.*?)\s*</(?:lyrics|lyric)>")
    return style, lyric


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


def resolve_text_inputs(args: argparse.Namespace) -> tuple[str, str]:
    style_text = args.style_text
    lyric_text = args.lyric_text

    if args.style_file:
        style_text = pathlib.Path(args.style_file).read_text(encoding="utf-8")
    if args.lyric_file:
        lyric_text = pathlib.Path(args.lyric_file).read_text(encoding="utf-8")

    prompt_style, prompt_lyric = parse_prompt_tags(args.prompt)
    if not style_text.strip():
        if prompt_style:
            style_text = prompt_style
        elif args.prompt.strip():
            style_text = args.prompt.strip()
    if not lyric_text.strip() and prompt_lyric:
        lyric_text = prompt_lyric
    return style_text.strip(), lyric_text.strip()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Unified prompt->tokenizer->style/lyric/timbre CLI (ComfyUI-like conditioning flow)."
    )
    p.add_argument("--model-root", default="/Users/fmh/project/ACE-Step-1.5/Ace-Step1.5")
    p.add_argument("--text-encoder-dir", default="")
    p.add_argument("--dit-dir", default="")
    p.add_argument("--vae-dir", default="")
    p.add_argument("--tokenizer-dir", default="", help="Tokenizer dir for style/lyric text (defaults to text-encoder dir for parity).")
    p.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")

    p.add_argument("--prompt", default="", help="free-form prompt; also supports <prompt>/<lyrics> tags")
    p.add_argument("--style-text", "--style", dest="style_text", default="")
    p.add_argument("--lyric-text", "--lyrics", dest="lyric_text", default="")
    p.add_argument("--style-file", default="")
    p.add_argument("--lyric-file", default="")

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
    p.add_argument("--graph-size", type=int, default=0)
    p.add_argument("--debug-graph", action="store_true")

    p.add_argument("--text-encoder-gguf", default="", help="optional pre-quantized text encoder gguf path")
    p.add_argument("--dit-gguf", default="", help="optional pre-quantized DiT gguf path")

    p.add_argument("--timbre-npy", default="")
    p.add_argument("--timbre-order", default="")
    p.add_argument("--timbre-rand-n", type=int, default=0)
    p.add_argument("--timbre-len", type=int, default=0)
    p.add_argument("--timbre-hidden", type=int, default=64)
    p.add_argument(
        "--auto-silence-timbre",
        choices=("on", "off"),
        default="on",
        help="When no timbre input is provided, use silence latent as synthetic timbre conditioning.",
    )

    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--out-wav", default="/tmp/ace_unified_style_lyric_timbre.wav")
    p.add_argument("--dump-json", default="", help="optional json dump for prompts/token ids")
    return p.parse_args()


def ensure_silence_latent_f32(dit_dir: pathlib.Path, latent_dim: int) -> pathlib.Path | None:
    pt_path = dit_dir / "silence_latent.pt"
    if not pt_path.exists():
        return None

    out_path = dit_dir / f"silence_latent_tfirst_{latent_dim}.f32"
    expected_row_bytes = int(latent_dim) * 4
    if out_path.exists() and out_path.stat().st_size >= expected_row_bytes and (out_path.stat().st_size % expected_row_bytes) == 0:
        return out_path

    with zipfile.ZipFile(pt_path, "r") as zf:
        members = set(zf.namelist())
        data_member = "silence_latent/data/0"
        if data_member not in members:
            raise RuntimeError(f"{pt_path} missing member {data_member}")
        raw = zf.read(data_member)

    buf = np.frombuffer(raw, dtype=np.float32)
    if buf.size == 0:
        raise RuntimeError(f"{pt_path} contains empty silence latent buffer")
    if (buf.size % latent_dim) != 0:
        raise RuntimeError(
            f"{pt_path} float count {buf.size} not divisible by latent_dim={latent_dim}"
        )

    # Torch stores this tensor as [1, C, T] contiguous; convert to [T, C] row-major.
    n_frames = buf.size // latent_dim
    t_first = buf.reshape(latent_dim, n_frames).transpose(1, 0).copy()
    t_first.tofile(out_path)
    return out_path


def main() -> int:
    args = parse_args()
    if args.debug_graph:
        os.environ["ACE_GGML_DEBUG_GRAPH"] = "1"
    if args.graph_size > 0:
        os.environ["ACE_GGML_DIT_GRAPH_SIZE"] = str(args.graph_size)
        os.environ["ACE_GGML_GRAPH_SIZE"] = str(args.graph_size)

    if args.text_encoder_gguf:
        os.environ["ACE_GGML_TEXT_ENCODER_GGUF"] = args.text_encoder_gguf
        os.environ["ACE_GGML_QWEN_GGUF"] = args.text_encoder_gguf
    if args.dit_gguf:
        os.environ["ACE_GGML_DIT_GGUF"] = args.dit_gguf
        os.environ["ACE_GGML_DIT_GGUF_PATH"] = args.dit_gguf

    model_root = pathlib.Path(args.model_root)
    text_encoder_dir = pathlib.Path(args.text_encoder_dir) if args.text_encoder_dir else model_root / "Qwen3-Embedding-0.6B"
    dit_dir = pathlib.Path(args.dit_dir) if args.dit_dir else model_root / "acestep-v15-turbo"
    vae_dir = pathlib.Path(args.vae_dir) if args.vae_dir else model_root
    tokenizer_dir = pathlib.Path(args.tokenizer_dir) if args.tokenizer_dir else text_encoder_dir

    style_text, lyric_text = resolve_text_inputs(args)
    if not style_text and not lyric_text:
        raise SystemExit("style and lyric are both empty; provide --prompt or --style/--lyrics")

    style_prompt = ""
    lyric_prompt = ""
    style_ids: list[int] = []
    lyric_ids: list[int] = []

    if style_text:
        style_prompt = format_style_prompt(
            style_text=style_text,
            instruction=args.instruction,
            bpm=args.bpm,
            timesignature=args.timesignature,
            keyscale=args.keyscale,
            duration=args.duration,
        )
        style_ids = encode_with_local_tokenizer(tokenizer_dir, style_prompt, args.style_max_len)
    if lyric_text:
        lyric_prompt = format_lyric_prompt(lyric_text, args.language)
        lyric_ids = encode_with_local_tokenizer(tokenizer_dir, lyric_prompt, args.lyric_max_len)
    if len(style_ids) == 0 and len(lyric_ids) == 0:
        raise RuntimeError("style and lyric tokenized to empty sequences")

    style_arr = np.asarray(style_ids, dtype=np.int32)
    lyric_arr = np.asarray(lyric_ids, dtype=np.int32)
    print(f"[tokenizer] style_tokens={style_arr.size} lyric_tokens={lyric_arr.size}")
    print(f"[tokenizer] dir={tokenizer_dir}")
    print(f"[tokenizer] style_first_32={style_arr[:32].tolist() if style_arr.size else []}")
    print(f"[tokenizer] lyric_first_32={lyric_arr[:32].tolist() if lyric_arr.size else []}")

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

    if args.dump_json:
        payload = {
            "prompt": args.prompt,
            "style_text": style_text,
            "lyric_text": lyric_text,
            "style_prompt": style_prompt,
            "lyric_prompt": lyric_prompt,
            "style_tokens": style_arr.tolist(),
            "lyric_tokens": lyric_arr.tolist(),
            "tokenizer_dir": str(tokenizer_dir),
            "text_encoder_dir": str(text_encoder_dir),
            "text_encoder_gguf": args.text_encoder_gguf,
            "dit_gguf": args.dit_gguf,
        }
        pathlib.Path(args.dump_json).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[saved] {args.dump_json}")

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

        os.environ.setdefault("ACE_GGML_USE_SILENCE_CONTEXT", "1")
        if not os.environ.get("ACE_GGML_SILENCE_LATENT_F32"):
            try:
                silence_f32 = ensure_silence_latent_f32(dit_dir, latent_ch.value)
            except Exception as exc:
                silence_f32 = None
                print(f"[warn] failed to prepare silence latent f32: {exc}")
            if silence_f32 is not None:
                os.environ["ACE_GGML_SILENCE_LATENT_F32"] = str(silence_f32)
                print(f"[context] ACE_GGML_SILENCE_LATENT_F32={silence_f32}")

        auto_silence_timbre = args.auto_silence_timbre == "on"
        if timbre_feat is None and auto_silence_timbre:
            silence_path = os.environ.get("ACE_GGML_SILENCE_LATENT_F32", "")
            if silence_path:
                try:
                    silence_lat = np.fromfile(silence_path, dtype=np.float32).reshape(-1, latent_ch.value)
                    refer_len = min(750, silence_lat.shape[0])
                    timbre_feat = silence_lat[:refer_len][None, :, :].astype(np.float32, copy=False)
                    timbre_order = np.zeros((1,), dtype=np.int32)
                    print(
                        f"[timbre] auto_silence refs=1 len={timbre_feat.shape[1]} dim={timbre_feat.shape[2]} order=[0]"
                    )
                except Exception as exc:
                    print(f"[warn] failed to create auto silence timbre: {exc}")
        elif timbre_feat is None:
            print("[timbre] auto_silence disabled; using style+lyric only")

        if timbre_feat is not None and timbre_order is not None:
            print(
                f"[timbre] refs={timbre_feat.shape[0]} len={timbre_feat.shape[1]} dim={timbre_feat.shape[2]} "
                f"order={timbre_order.tolist()}"
            )

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

        style_ptr = style_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if style_arr.size else None
        lyric_ptr = lyric_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if lyric_arr.size else None

        if timbre_feat is not None:
            ensure_ok(
                lib,
                ctx,
                lib.ace_ggml_generate_audio_style_lyric_timbre_simple(
                    ctx,
                    style_ptr,
                    ctypes.c_int32(style_arr.size),
                    lyric_ptr,
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
                    style_ptr,
                    ctypes.c_int32(style_arr.size),
                    lyric_ptr,
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
