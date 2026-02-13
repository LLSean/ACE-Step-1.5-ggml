#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import datetime as dt
import json
import math
import os
import pathlib
import time
import wave
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

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


@dataclass
class RunResult:
    name: str
    quant: str
    ok: bool
    load_s: float
    infer_s: float
    total_s: float
    out_samples: int
    out_channels: int
    wav_path: str
    text_encoder_gguf: str
    dit_gguf: str
    error: str
    metrics: Dict[str, float]


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
    return lib


def ace_error(lib, ctx: AceCtxPtr) -> str:
    if not ctx:
        return "unknown error"
    msg = lib.ace_ggml_last_error(ctx)
    if not msg:
        return "unknown error"
    return msg.decode("utf-8", errors="replace")


def ensure_ok(lib, ctx: AceCtxPtr, status: int, step: str) -> None:
    if status != ACE_GGML_OK:
        raise RuntimeError(f"{step} failed: {ace_error(lib, ctx)} (status={status})")


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


def encode_with_local_tokenizer(tokenizer_dir: pathlib.Path, text: str, max_len: int) -> List[int]:
    tok_json = tokenizer_dir / "tokenizer.json"
    if tok_json.exists():
        from tokenizers import Tokenizer  # type: ignore

        tok = Tokenizer.from_file(str(tok_json))
        ids = tok.encode(text).ids
        if max_len > 0 and len(ids) > max_len:
            ids = ids[:max_len]
        return ids

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    return tokenizer.encode(text, add_special_tokens=True, truncation=True, max_length=max_len)


def set_env_for_run(
    quant: Optional[str],
    text_max_layers: int,
    text_encoder_gguf: str = "",
    dit_gguf: str = "",
) -> Dict[str, Optional[str]]:
    keys = [
        "ACE_GGML_WEIGHT_QTYPE",
        "ACE_GGML_QWEN_WEIGHT_QTYPE",
        "ACE_GGML_DIT_WEIGHT_QTYPE",
        "ACE_GGML_TEXT_MAX_LAYERS",
        "ACE_GGML_TEXT_ENCODER_GGUF",
        "ACE_GGML_QWEN_GGUF",
        "ACE_GGML_DIT_GGUF",
        "ACE_GGML_DIT_GGUF_PATH",
    ]
    old = {k: os.environ.get(k) for k in keys}
    for k in ("ACE_GGML_WEIGHT_QTYPE", "ACE_GGML_QWEN_WEIGHT_QTYPE", "ACE_GGML_DIT_WEIGHT_QTYPE"):
        os.environ.pop(k, None)
    for k in ("ACE_GGML_TEXT_ENCODER_GGUF", "ACE_GGML_QWEN_GGUF", "ACE_GGML_DIT_GGUF", "ACE_GGML_DIT_GGUF_PATH"):
        os.environ.pop(k, None)
    if quant:
        os.environ["ACE_GGML_WEIGHT_QTYPE"] = quant
    if text_encoder_gguf:
        os.environ["ACE_GGML_TEXT_ENCODER_GGUF"] = text_encoder_gguf
        os.environ["ACE_GGML_QWEN_GGUF"] = text_encoder_gguf
    if dit_gguf:
        os.environ["ACE_GGML_DIT_GGUF"] = dit_gguf
        os.environ["ACE_GGML_DIT_GGUF_PATH"] = dit_gguf
    if text_max_layers >= 0:
        os.environ["ACE_GGML_TEXT_MAX_LAYERS"] = str(text_max_layers)
    else:
        os.environ.pop("ACE_GGML_TEXT_MAX_LAYERS", None)
    return old


def restore_env(old: Dict[str, Optional[str]]) -> None:
    for k, v in old.items():
        if v is None:
            os.environ.pop(k, None)
        else:
            os.environ[k] = v


def write_wav(path: pathlib.Path, audio: np.ndarray, sample_rate: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    a = np.asarray(audio, dtype=np.float32)
    a = np.clip(a, -1.0, 1.0)
    pcm = (a * 32767.0).astype(np.int16)
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(int(a.shape[1]))
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm.tobytes())


def _mono(audio: np.ndarray) -> np.ndarray:
    if audio.ndim == 1:
        return audio.astype(np.float32)
    return np.mean(audio.astype(np.float32), axis=1)


def _stft_logmag(x: np.ndarray, n_fft: int = 1024, hop: int = 256, eps: float = 1e-8) -> np.ndarray:
    x = x.astype(np.float32, copy=False)
    if x.size < n_fft:
        x = np.pad(x, (0, n_fft - x.size), mode="constant")
    n_frames = 1 + (x.size - n_fft) // hop
    if n_frames <= 0:
        n_frames = 1
        x = np.pad(x, (0, n_fft - x.size), mode="constant")
    w = np.hanning(n_fft).astype(np.float32)
    feats = []
    for i in range(n_frames):
        s = i * hop
        frame = x[s:s + n_fft]
        if frame.size < n_fft:
            frame = np.pad(frame, (0, n_fft - frame.size), mode="constant")
        spec = np.fft.rfft(frame * w)
        mag = np.abs(spec).astype(np.float32)
        feats.append(np.log10(mag + eps))
    return np.stack(feats, axis=0)


def calc_metrics(ref: np.ndarray, cur: np.ndarray) -> Dict[str, float]:
    a = ref.astype(np.float32).reshape(-1)
    b = cur.astype(np.float32).reshape(-1)
    n = min(a.size, b.size)
    a = a[:n]
    b = b[:n]
    diff = b - a
    mae = float(np.mean(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff * diff)))
    peak = float(np.max(np.abs(diff)))
    denom = float(np.linalg.norm(a) * np.linalg.norm(b) + 1e-12)
    cosine = float(np.dot(a, b) / denom)
    ref_pow = float(np.mean(a * a) + 1e-12)
    err_pow = float(np.mean(diff * diff) + 1e-12)
    snr_db = float(10.0 * math.log10(ref_pow / err_pow))
    la = _stft_logmag(_mono(ref))
    lb = _stft_logmag(_mono(cur))
    m = min(la.shape[0], lb.shape[0])
    la = la[:m]
    lb = lb[:m]
    lsd = float(np.mean(np.sqrt(np.mean((la - lb) ** 2, axis=1))))
    return {
        "mae": mae,
        "rmse": rmse,
        "peak_abs_diff": peak,
        "cosine": cosine,
        "snr_db": snr_db,
        "lsd": lsd,
    }


def markdown_table(results: List[RunResult]) -> str:
    lines = []
    lines.append("| variant | ok | load_s | infer_s | total_s | mae | rmse | cosine | snr_db | lsd | wav |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|")
    for r in results:
        m = r.metrics
        lines.append(
            f"| {r.quant} | {int(r.ok)} | {r.load_s:.3f} | {r.infer_s:.3f} | {r.total_s:.3f} | "
            f"{m.get('mae', float('nan')):.6f} | {m.get('rmse', float('nan')):.6f} | "
            f"{m.get('cosine', float('nan')):.6f} | {m.get('snr_db', float('nan')):.3f} | "
            f"{m.get('lsd', float('nan')):.6f} | {r.wav_path} |"
        )
    return "\n".join(lines)


def run_one(
    lib,
    name: str,
    quant: Optional[str],
    quant_label: str,
    text_encoder_dir: pathlib.Path,
    dit_dir: pathlib.Path,
    vae_dir: pathlib.Path,
    style_ids: np.ndarray,
    lyric_ids: np.ndarray,
    seq_len: int,
    shift: float,
    seed: int,
    threads: int,
    compute_buffer_mb: int,
    text_max_layers: int,
    text_encoder_gguf: str,
    dit_gguf: str,
    out_dir: pathlib.Path,
    sample_rate: int,
) -> Tuple[RunResult, Optional[np.ndarray]]:
    old = set_env_for_run(
        quant=quant,
        text_max_layers=text_max_layers,
        text_encoder_gguf=text_encoder_gguf,
        dit_gguf=dit_gguf,
    )
    ctx = AceCtxPtr()
    t_all0 = time.perf_counter()
    load_s = 0.0
    infer_s = 0.0
    out_audio = None
    out_samples = 0
    out_channels = 0
    error = ""
    ok = False
    wav_path = out_dir / f"{name}.wav"
    try:
        params = AceInitParams(
            n_threads=threads,
            use_metal=0,
            compute_buffer_bytes=int(compute_buffer_mb) * 1024 * 1024,
        )
        st = lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx))
        ensure_ok(lib, ctx, st, "ace_ggml_create")

        t0 = time.perf_counter()
        ensure_ok(lib, ctx, lib.ace_ggml_load_text_encoder(ctx, str(text_encoder_dir).encode("utf-8")), "load_text_encoder")
        ensure_ok(lib, ctx, lib.ace_ggml_load_dit(ctx, str(dit_dir).encode("utf-8")), "load_dit")
        ensure_ok(lib, ctx, lib.ace_ggml_load_vae(ctx, str(vae_dir).encode("utf-8")), "load_vae")
        load_s = time.perf_counter() - t0

        latent_ch = ctypes.c_int32(0)
        audio_ch = ctypes.c_int32(0)
        hop = ctypes.c_int32(0)
        ensure_ok(lib, ctx, lib.ace_ggml_vae_get_info(ctx, ctypes.byref(latent_ch), ctypes.byref(audio_ch), ctypes.byref(hop)), "vae_get_info")

        out_samples = int(seq_len * hop.value)
        out_channels = int(audio_ch.value)
        out_audio = np.zeros(out_samples * out_channels, dtype=np.float32)
        ret_samples = ctypes.c_int32(0)
        ret_channels = ctypes.c_int32(0)

        t1 = time.perf_counter()
        ensure_ok(
            lib,
            ctx,
            lib.ace_ggml_generate_audio_style_lyric_simple(
                ctx,
                style_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(style_ids.size),
                lyric_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(lyric_ids.size),
                ctypes.c_int32(seq_len),
                ctypes.c_float(shift),
                ctypes.c_int32(seed),
                out_audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_size_t(out_audio.nbytes),
                ctypes.byref(ret_samples),
                ctypes.byref(ret_channels),
            ),
            "generate_audio_style_lyric_simple",
        )
        infer_s = time.perf_counter() - t1

        if ret_samples.value > 0 and ret_channels.value > 0:
            out_samples = int(ret_samples.value)
            out_channels = int(ret_channels.value)
        out_audio = out_audio[: out_samples * out_channels].reshape(out_samples, out_channels)
        write_wav(wav_path, out_audio, sample_rate=sample_rate)
        ok = True
    except Exception as e:  # noqa: BLE001
        error = str(e)
    finally:
        if ctx:
            lib.ace_ggml_destroy(ctx)
        restore_env(old)

    total_s = time.perf_counter() - t_all0
    rr = RunResult(
        name=name,
        quant=quant_label,
        ok=ok,
        load_s=load_s,
        infer_s=infer_s,
        total_s=total_s,
        out_samples=out_samples,
        out_channels=out_channels,
        wav_path=str(wav_path),
        text_encoder_gguf=text_encoder_gguf,
        dit_gguf=dit_gguf,
        error=error,
        metrics={},
    )
    return rr, out_audio


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Style+Lyric tokenizer->ACE ggml quantized end-to-end eval")
    p.add_argument("--model-root", default="/Users/fmh/project/ACE-Step-1.5/Ace-Step1.5")
    p.add_argument("--text-encoder-dir", default="")
    p.add_argument("--dit-dir", default="")
    p.add_argument("--vae-dir", default="")
    p.add_argument("--tokenizer-dir", default="", help="Tokenizer dir for style/lyric text (defaults to text-encoder dir for parity).")
    p.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")
    p.add_argument("--style-text", default="")
    p.add_argument("--lyric-text", default="")
    p.add_argument("--style-file", default="")
    p.add_argument("--lyric-file", default="")
    p.add_argument("--instruction", default=DEFAULT_DIT_INSTRUCTION)
    p.add_argument("--language", default="en")
    p.add_argument("--bpm", default="N/A")
    p.add_argument("--timesignature", default="N/A")
    p.add_argument("--keyscale", default="N/A")
    p.add_argument("--duration", default="30 seconds")
    p.add_argument("--style-max-len", type=int, default=256)
    p.add_argument("--lyric-max-len", type=int, default=2048)
    p.add_argument("--sample-rate", type=int, default=48000)
    p.add_argument("--seq-len", type=int, default=4)
    p.add_argument("--shift", type=float, default=3.0)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--threads", type=int, default=8)
    p.add_argument("--compute-buffer-mb", type=int, default=1024)
    p.add_argument("--quant-list", default="Q8,Q6,Q4")
    p.add_argument("--text-max-layers", type=int, default=4)
    p.add_argument("--graph-size", type=int, default=0, help="set ACE_GGML_DIT_GRAPH_SIZE/ACE_GGML_GRAPH_SIZE")
    p.add_argument("--debug-graph", action="store_true", help="enable ACE_GGML_DEBUG_GRAPH=1")
    p.add_argument("--output-dir", default="")
    p.add_argument(
        "--text-encoder-gguf-map",
        default="",
        help='quant->gguf map, e.g. "Q8=/path/qwen_q8.gguf,Q6=/path/qwen_q6.gguf,Q4=/path/qwen_q4.gguf"',
    )
    p.add_argument(
        "--dit-gguf-map",
        default="",
        help='quant->gguf map, e.g. "Q8=/path/dit_q8.gguf,Q6=/path/dit_q6.gguf,Q4=/path/dit_q4.gguf"',
    )
    return p.parse_args()


def parse_quant_path_map(spec: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in spec.split(","):
        part = item.strip()
        if not part:
            continue
        if "=" not in part:
            raise ValueError(f"invalid map item: {part}")
        k, v = part.split("=", 1)
        q = k.strip().upper()
        p = v.strip()
        if not q or not p:
            raise ValueError(f"invalid map item: {part}")
        out[q] = p
    return out


def main() -> int:
    args = parse_args()
    style_text = args.style_text
    lyric_text = args.lyric_text
    if args.style_file:
        style_text = pathlib.Path(args.style_file).read_text(encoding="utf-8")
    if args.lyric_file:
        lyric_text = pathlib.Path(args.lyric_file).read_text(encoding="utf-8")
    if not style_text.strip() or not lyric_text.strip():
        raise SystemExit("style-text/lyric-text is empty")

    model_root = pathlib.Path(args.model_root)
    text_encoder_dir = pathlib.Path(args.text_encoder_dir) if args.text_encoder_dir else model_root / "Qwen3-Embedding-0.6B"
    dit_dir = pathlib.Path(args.dit_dir) if args.dit_dir else model_root / "acestep-v15-turbo"
    vae_dir = pathlib.Path(args.vae_dir) if args.vae_dir else model_root
    tokenizer_dir = pathlib.Path(args.tokenizer_dir) if args.tokenizer_dir else text_encoder_dir
    lib_path = pathlib.Path(args.lib)
    text_gguf_map = parse_quant_path_map(args.text_encoder_gguf_map)
    dit_gguf_map = parse_quant_path_map(args.dit_gguf_map)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = pathlib.Path(args.output_dir) if args.output_dir else pathlib.Path("/tmp") / f"ace_quant_style_lyric_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.debug_graph:
        os.environ["ACE_GGML_DEBUG_GRAPH"] = "1"
    if args.graph_size > 0:
        os.environ["ACE_GGML_DIT_GRAPH_SIZE"] = str(args.graph_size)
        os.environ["ACE_GGML_GRAPH_SIZE"] = str(args.graph_size)

    style_prompt = format_style_prompt(
        style_text=style_text,
        instruction=args.instruction,
        bpm=args.bpm,
        timesignature=args.timesignature,
        keyscale=args.keyscale,
        duration=args.duration,
    )
    lyric_prompt = format_lyric_prompt(lyric_text, args.language)

    style_ids = encode_with_local_tokenizer(tokenizer_dir, style_prompt, args.style_max_len)
    lyric_ids = encode_with_local_tokenizer(tokenizer_dir, lyric_prompt, args.lyric_max_len)
    style_arr = np.asarray(style_ids, dtype=np.int32)
    lyric_arr = np.asarray(lyric_ids, dtype=np.int32)
    if style_arr.size == 0 or lyric_arr.size == 0:
        raise RuntimeError("style/lyric tokenized to empty sequence")

    print(f"[info] output_dir={out_dir}")
    print(f"[tokenizer] style_tokens={style_arr.size} lyric_tokens={lyric_arr.size}")
    print(f"[tokenizer] dir={tokenizer_dir}")
    print(f"[tokenizer] style_first_64={style_arr[:64].tolist()}")
    print(f"[tokenizer] lyric_first_64={lyric_arr[:64].tolist()}")

    lib = bind_lib(str(lib_path))
    runs: List[RunResult] = []

    baseline, base_audio = run_one(
        lib=lib,
        name="fp",
        quant=None,
        quant_label="FP",
        text_encoder_dir=text_encoder_dir,
        dit_dir=dit_dir,
        vae_dir=vae_dir,
        style_ids=style_arr,
        lyric_ids=lyric_arr,
        seq_len=args.seq_len,
        shift=args.shift,
        seed=args.seed,
        threads=args.threads,
        compute_buffer_mb=args.compute_buffer_mb,
        text_max_layers=args.text_max_layers,
        text_encoder_gguf="",
        dit_gguf="",
        out_dir=out_dir,
        sample_rate=args.sample_rate,
    )
    runs.append(baseline)
    print(f"[run] FP ok={baseline.ok} load_s={baseline.load_s:.3f} infer_s={baseline.infer_s:.3f} total_s={baseline.total_s:.3f}")
    if not baseline.ok or base_audio is None:
        report = {
            "ok": False,
            "error": baseline.error,
            "runs": [baseline.__dict__],
            "style_text": style_text,
            "lyric_text": lyric_text,
            "style_tokens": style_arr.tolist(),
            "lyric_tokens": lyric_arr.tolist(),
        }
        (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[error] FP run failed: {baseline.error}")
        return 1

    quant_list = [q.strip().upper() for q in args.quant_list.split(",") if q.strip()]
    for q in quant_list:
        text_gguf = text_gguf_map.get(q, "")
        dit_gguf = dit_gguf_map.get(q, "")
        use_prequant_gguf = bool(text_gguf and dit_gguf)
        quant_mode = None if use_prequant_gguf else q
        print(f"[run] start {q} source={'GGUF' if use_prequant_gguf else 'ONLINE'}")
        rr, cur_audio = run_one(
            lib=lib,
            name=q.lower(),
            quant=quant_mode,
            quant_label=q,
            text_encoder_dir=text_encoder_dir,
            dit_dir=dit_dir,
            vae_dir=vae_dir,
            style_ids=style_arr,
            lyric_ids=lyric_arr,
            seq_len=args.seq_len,
            shift=args.shift,
            seed=args.seed,
            threads=args.threads,
            compute_buffer_mb=args.compute_buffer_mb,
            text_max_layers=args.text_max_layers,
            text_encoder_gguf=text_gguf,
            dit_gguf=dit_gguf,
            out_dir=out_dir,
            sample_rate=args.sample_rate,
        )
        if rr.ok and cur_audio is not None:
            rr.metrics = calc_metrics(base_audio, cur_audio)
        runs.append(rr)
        print(f"[run] {q} ok={rr.ok} load_s={rr.load_s:.3f} infer_s={rr.infer_s:.3f} total_s={rr.total_s:.3f}")
        if rr.ok:
            print(
                f"[metric] {q} mae={rr.metrics['mae']:.6f} rmse={rr.metrics['rmse']:.6f} "
                f"cos={rr.metrics['cosine']:.6f} snr_db={rr.metrics['snr_db']:.3f} lsd={rr.metrics['lsd']:.6f}"
            )
        else:
            print(f"[error] {q} failed: {rr.error}")

    report = {
        "ok": all(r.ok for r in runs),
        "style_text": style_text,
        "lyric_text": lyric_text,
        "style_token_count": int(style_arr.size),
        "lyric_token_count": int(lyric_arr.size),
        "style_tokens": style_arr.tolist(),
        "lyric_tokens": lyric_arr.tolist(),
        "model_root": str(model_root),
        "text_encoder_dir": str(text_encoder_dir),
        "dit_dir": str(dit_dir),
        "vae_dir": str(vae_dir),
        "lib": str(lib_path),
        "seq_len": args.seq_len,
        "shift": args.shift,
        "seed": args.seed,
        "threads": args.threads,
        "compute_buffer_mb": args.compute_buffer_mb,
        "text_max_layers": args.text_max_layers,
        "text_encoder_gguf_map": text_gguf_map,
        "dit_gguf_map": dit_gguf_map,
        "runs": [r.__dict__ for r in runs],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    md = markdown_table(runs)
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")

    print("[summary]")
    print(md)
    print(f"[saved] {out_dir / 'summary.json'}")
    print(f"[saved] {out_dir / 'summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
