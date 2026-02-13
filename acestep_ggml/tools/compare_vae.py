#!/usr/bin/env python3
import argparse
import ctypes
import datetime as dt
import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple


def _set_runtime_env_defaults() -> None:
    defaults = {
        "OMP_NUM_THREADS": "1",
        "MKL_NUM_THREADS": "1",
        "OPENBLAS_NUM_THREADS": "1",
        "VECLIB_MAXIMUM_THREADS": "1",
        "NUMEXPR_NUM_THREADS": "1",
        "KMP_DUPLICATE_LIB_OK": "TRUE",
        "KMP_INIT_AT_FORK": "FALSE",
        "KMP_USE_SHM": "0",
        "KMP_DISABLE_SHM": "1",
        "KMP_SHM": "0",
    }
    for k, v in defaults.items():
        os.environ.setdefault(k, v)


_set_runtime_env_defaults()

import numpy as np


ACE_GGML_OK = 0


def resolve_vae_dir(model_dir: str) -> str:
    p = Path(model_dir)
    if (p / "config.json").exists() and (p / "diffusion_pytorch_model.safetensors").exists():
        return str(p)
    p2 = p / "vae"
    if (p2 / "config.json").exists() and (p2 / "diffusion_pytorch_model.safetensors").exists():
        return str(p2)
    raise RuntimeError(f"cannot resolve vae dir from: {model_dir}")


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-12
    return float(np.dot(a_flat, b_flat) / denom)


@dataclass
class ModeResult:
    mode: str
    ok: bool
    ggml_s: float
    pt_s: float
    shape: str
    mae: float
    max_err: float
    cosine: float
    info: Optional[Tuple[int, int, int]]
    error: str


def markdown_table(results: List[ModeResult]) -> str:
    lines = []
    lines.append("| mode | ok | ggml_s | pt_s | shape | mae | max_err | cosine |")
    lines.append("|---|---:|---:|---:|---|---:|---:|---:|")
    for r in results:
        lines.append(
            f"| {r.mode} | {int(r.ok)} | {r.ggml_s:.3f} | {r.pt_s:.3f} | {r.shape} | "
            f"{r.mae:.6f} | {r.max_err:.6f} | {r.cosine:.6f} |"
        )
    return "\n".join(lines)


def load_ggml_vae_decode(
    lib_path: str,
    model_dir: str,
    latents: np.ndarray,
    threads: int,
    compute_mb: int,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    class AceCtx(ctypes.Structure):
        pass

    AceCtxPtr = ctypes.POINTER(AceCtx)

    class AceInitParams(ctypes.Structure):
        _fields_ = [
            ("n_threads", ctypes.c_int32),
            ("use_metal", ctypes.c_int32),
            ("compute_buffer_bytes", ctypes.c_size_t),
        ]

    lib = ctypes.CDLL(lib_path)
    lib.ace_ggml_create.argtypes = [ctypes.POINTER(AceInitParams), ctypes.POINTER(AceCtxPtr)]
    lib.ace_ggml_create.restype = ctypes.c_int
    lib.ace_ggml_destroy.argtypes = [AceCtxPtr]
    lib.ace_ggml_destroy.restype = None
    lib.ace_ggml_last_error.argtypes = [AceCtxPtr]
    lib.ace_ggml_last_error.restype = ctypes.c_char_p
    lib.ace_ggml_load_vae.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_vae.restype = ctypes.c_int
    lib.ace_ggml_vae_get_info.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ace_ggml_vae_get_info.restype = ctypes.c_int
    lib.ace_ggml_vae_decode.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.ace_ggml_vae_decode.restype = ctypes.c_int
    lib.ace_ggml_vae_encode.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.ace_ggml_vae_encode.restype = ctypes.c_int

    ctx = AceCtxPtr()
    params = AceInitParams(n_threads=threads, use_metal=0, compute_buffer_bytes=compute_mb * 1024 * 1024)
    status = lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx))
    if status != ACE_GGML_OK:
        raise RuntimeError("ace_ggml_create failed")

    try:
        status = lib.ace_ggml_load_vae(ctx, model_dir.encode("utf-8"))
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"load_vae failed: {err.decode('utf-8') if err else ''}")

        latent_channels = ctypes.c_int32()
        audio_channels = ctypes.c_int32()
        hop_length = ctypes.c_int32()
        status = lib.ace_ggml_vae_get_info(
            ctx,
            ctypes.byref(latent_channels),
            ctypes.byref(audio_channels),
            ctypes.byref(hop_length),
        )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"vae_get_info failed: {err.decode('utf-8') if err else ''}")

        if latents.shape[1] != latent_channels.value:
            raise RuntimeError(
                f"latent channel mismatch: got {latents.shape[1]}, expected {latent_channels.value}"
            )

        n_frames = latents.shape[0]
        n_samples = n_frames * hop_length.value
        out = np.empty((n_samples, audio_channels.value), dtype=np.float32)
        status = lib.ace_ggml_vae_decode(
            ctx,
            latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n_frames),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.nbytes,
        )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"vae_decode failed: {err.decode('utf-8') if err else ''}")
        return out, (latent_channels.value, audio_channels.value, hop_length.value)
    finally:
        lib.ace_ggml_destroy(ctx)


def load_ggml_vae_encode(
    lib_path: str,
    model_dir: str,
    audio: np.ndarray,
    threads: int,
    compute_mb: int,
) -> Tuple[np.ndarray, Tuple[int, int, int]]:
    class AceCtx(ctypes.Structure):
        pass

    AceCtxPtr = ctypes.POINTER(AceCtx)

    class AceInitParams(ctypes.Structure):
        _fields_ = [
            ("n_threads", ctypes.c_int32),
            ("use_metal", ctypes.c_int32),
            ("compute_buffer_bytes", ctypes.c_size_t),
        ]

    lib = ctypes.CDLL(lib_path)
    lib.ace_ggml_create.argtypes = [ctypes.POINTER(AceInitParams), ctypes.POINTER(AceCtxPtr)]
    lib.ace_ggml_create.restype = ctypes.c_int
    lib.ace_ggml_destroy.argtypes = [AceCtxPtr]
    lib.ace_ggml_destroy.restype = None
    lib.ace_ggml_last_error.argtypes = [AceCtxPtr]
    lib.ace_ggml_last_error.restype = ctypes.c_char_p
    lib.ace_ggml_load_vae.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_vae.restype = ctypes.c_int
    lib.ace_ggml_vae_get_info.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.ace_ggml_vae_get_info.restype = ctypes.c_int
    lib.ace_ggml_vae_encode.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.ace_ggml_vae_encode.restype = ctypes.c_int

    ctx = AceCtxPtr()
    params = AceInitParams(n_threads=threads, use_metal=0, compute_buffer_bytes=compute_mb * 1024 * 1024)
    status = lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx))
    if status != ACE_GGML_OK:
        raise RuntimeError("ace_ggml_create failed")

    try:
        status = lib.ace_ggml_load_vae(ctx, model_dir.encode("utf-8"))
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"load_vae failed: {err.decode('utf-8') if err else ''}")

        latent_channels = ctypes.c_int32()
        audio_channels = ctypes.c_int32()
        hop_length = ctypes.c_int32()
        status = lib.ace_ggml_vae_get_info(
            ctx,
            ctypes.byref(latent_channels),
            ctypes.byref(audio_channels),
            ctypes.byref(hop_length),
        )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"vae_get_info failed: {err.decode('utf-8') if err else ''}")

        if audio.shape[1] != audio_channels.value:
            raise RuntimeError(
                f"audio channel mismatch: got {audio.shape[1]}, expected {audio_channels.value}"
            )

        n_samples = audio.shape[0]
        n_frames = n_samples // hop_length.value
        out = np.empty((n_frames, latent_channels.value), dtype=np.float32)
        status = lib.ace_ggml_vae_encode(
            ctx,
            audio.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n_samples),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.nbytes,
        )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"vae_encode failed: {err.decode('utf-8') if err else ''}")
        return out, (latent_channels.value, audio_channels.value, hop_length.value)
    finally:
        lib.ace_ggml_destroy(ctx)


def load_pt_vae_decode(vae_dir: str, latents: np.ndarray, pt_dtype: str) -> np.ndarray:
    import torch
    from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
    from safetensors.torch import load_file

    dtype = torch.float32 if pt_dtype == "f32" else torch.bfloat16

    with open(os.path.join(vae_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = AutoencoderOobleck.from_config(cfg)
    state = load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(dtype=dtype)

    # latents [T, C] -> [B, C, T]
    z = torch.from_numpy(latents).t().unsqueeze(0).to(dtype=dtype)
    with torch.no_grad():
        dec = model.decode(z)
        x = dec.sample if hasattr(dec, "sample") else dec[0]
    # [B, C, T] -> [T, C]
    out = x[0].transpose(0, 1).contiguous().float().cpu().numpy().astype(np.float32)
    return out


def load_pt_vae_encode(vae_dir: str, audio: np.ndarray, pt_dtype: str) -> np.ndarray:
    import torch
    from diffusers.models.autoencoders.autoencoder_oobleck import AutoencoderOobleck
    from safetensors.torch import load_file

    dtype = torch.float32 if pt_dtype == "f32" else torch.bfloat16

    with open(os.path.join(vae_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    model = AutoencoderOobleck.from_config(cfg)
    state = load_file(os.path.join(vae_dir, "diffusion_pytorch_model.safetensors"), device="cpu")
    model.load_state_dict(state, strict=True)
    model.eval()
    model.to(dtype=dtype)

    # audio [T, C] -> [B, C, T]
    x = torch.from_numpy(audio).t().unsqueeze(0).to(dtype=dtype)
    with torch.no_grad():
        post = model.encode(x).latent_dist
        z = post.mode()
    # [B, C, T] -> [T, C]
    out = z[0].transpose(0, 1).contiguous().float().cpu().numpy().astype(np.float32)
    return out


def run_mode(
    mode: str,
    args: argparse.Namespace,
    cfg: Dict[str, object],
    vae_dir: str,
    rng: np.random.Generator,
) -> ModeResult:
    out_pt: Optional[np.ndarray] = None
    out_ggml: Optional[np.ndarray] = None
    info: Optional[Tuple[int, int, int]] = None
    ggml_s = 0.0
    pt_s = 0.0

    try:
        if mode == "decode":
            latent_channels = int(cfg["decoder_input_channels"])
            latents = rng.standard_normal((args.latent_len, latent_channels), dtype=np.float32)
            if not args.pt_only:
                print(">>> ggml vae decode...", file=sys.stderr, flush=True)
                t0 = time.perf_counter()
                out_ggml, info = load_ggml_vae_decode(args.lib, args.model_dir, latents, args.threads, args.compute_mb)
                ggml_s = time.perf_counter() - t0
                print(">>> ggml vae decode done", file=sys.stderr, flush=True)
            if not args.ggml_only:
                print(">>> pt vae decode...", file=sys.stderr, flush=True)
                t1 = time.perf_counter()
                out_pt = load_pt_vae_decode(vae_dir, latents, args.pt_dtype)
                pt_s = time.perf_counter() - t1
                print(">>> pt vae decode done", file=sys.stderr, flush=True)
        else:
            audio_channels = int(cfg["audio_channels"])
            hop_length = 1
            for r in cfg.get("downsampling_ratios", []):
                hop_length *= int(r)
            aligned_audio_len = max(int(args.audio_len), int(hop_length))
            aligned_audio_len = ((aligned_audio_len + hop_length - 1) // hop_length) * hop_length
            audio = rng.standard_normal((aligned_audio_len, audio_channels), dtype=np.float32)
            if not args.pt_only:
                print(">>> ggml vae encode...", file=sys.stderr, flush=True)
                t0 = time.perf_counter()
                out_ggml, info = load_ggml_vae_encode(args.lib, args.model_dir, audio, args.threads, args.compute_mb)
                ggml_s = time.perf_counter() - t0
                print(">>> ggml vae encode done", file=sys.stderr, flush=True)
            if not args.ggml_only:
                print(">>> pt vae encode...", file=sys.stderr, flush=True)
                t1 = time.perf_counter()
                out_pt = load_pt_vae_encode(vae_dir, audio, args.pt_dtype)
                pt_s = time.perf_counter() - t1
                print(">>> pt vae encode done", file=sys.stderr, flush=True)
    except Exception as e:  # noqa: BLE001
        return ModeResult(
            mode=mode,
            ok=False,
            ggml_s=ggml_s,
            pt_s=pt_s,
            shape="-",
            mae=float("nan"),
            max_err=float("nan"),
            cosine=float("nan"),
            info=info,
            error=str(e),
        )

    if out_pt is not None and out_ggml is not None:
        if out_pt.shape != out_ggml.shape:
            return ModeResult(
                mode=mode,
                ok=False,
                ggml_s=ggml_s,
                pt_s=pt_s,
                shape=f"pt={out_pt.shape}, ggml={out_ggml.shape}",
                mae=float("nan"),
                max_err=float("nan"),
                cosine=float("nan"),
                info=info,
                error="shape mismatch",
            )
        diff = out_pt - out_ggml
        mae = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))
        cos = cosine_similarity(out_pt, out_ggml)
        return ModeResult(
            mode=mode,
            ok=True,
            ggml_s=ggml_s,
            pt_s=pt_s,
            shape=str(tuple(out_pt.shape)),
            mae=mae,
            max_err=max_err,
            cosine=cos,
            info=info,
            error="",
        )

    if out_ggml is not None:
        return ModeResult(
            mode=mode,
            ok=True,
            ggml_s=ggml_s,
            pt_s=pt_s,
            shape=str(tuple(out_ggml.shape)),
            mae=float("nan"),
            max_err=float("nan"),
            cosine=float("nan"),
            info=info,
            error="",
        )
    if out_pt is not None:
        return ModeResult(
            mode=mode,
            ok=True,
            ggml_s=ggml_s,
            pt_s=pt_s,
            shape=str(tuple(out_pt.shape)),
            mae=float("nan"),
            max_err=float("nan"),
            cosine=float("nan"),
            info=info,
            error="",
        )

    return ModeResult(
        mode=mode,
        ok=False,
        ggml_s=ggml_s,
        pt_s=pt_s,
        shape="-",
        mae=float("nan"),
        max_err=float("nan"),
        cosine=float("nan"),
        info=info,
        error="no output generated",
    )


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True, help="Ace-Step root dir or vae dir")
    ap.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")
    ap.add_argument("--mode", default="decode", choices=["decode", "encode", "both"])
    ap.add_argument("--latent-len", type=int, default=8)
    ap.add_argument("--audio-len", type=int, default=7680)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--compute-mb", type=int, default=768)
    ap.add_argument("--pt-dtype", default="bf16", choices=["f32", "bf16"])
    ap.add_argument("--ggml-only", action="store_true")
    ap.add_argument("--pt-only", action="store_true")
    ap.add_argument("--output-dir", default="")
    args = ap.parse_args()

    if args.ggml_only and args.pt_only:
        raise ValueError("--ggml-only and --pt-only cannot both be set")
    if args.mode in ("decode", "both") and args.latent_len <= 0:
        raise ValueError("--latent-len must be > 0")
    if args.mode in ("encode", "both") and args.audio_len <= 0:
        raise ValueError("--audio-len must be > 0")

    vae_dir = resolve_vae_dir(args.model_dir)
    with open(os.path.join(vae_dir, "config.json"), "r", encoding="utf-8") as f:
        cfg = json.load(f)

    modes = ["decode", "encode"] if args.mode == "both" else [args.mode]
    results: List[ModeResult] = []
    for idx, mode in enumerate(modes):
        rng = np.random.default_rng(args.seed + idx)
        rr = run_mode(mode, args, cfg, vae_dir, rng)
        results.append(rr)
        print(f"[mode] {mode} ok={rr.ok} ggml_s={rr.ggml_s:.3f} pt_s={rr.pt_s:.3f} shape={rr.shape}")
        if rr.info is not None:
            print(f"[info] latent_channels={rr.info[0]} audio_channels={rr.info[1]} hop_length={rr.info[2]}")
        if rr.error:
            print(f"[error] {mode}: {rr.error}")
        elif not (args.ggml_only or args.pt_only):
            print(f"[metric] {mode} mae={rr.mae:.6f} max={rr.max_err:.6f} cos={rr.cosine:.6f}")

    md = markdown_table(results)
    print("[summary]")
    print(md)

    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.output_dir) if args.output_dir else Path("/tmp") / f"ace_vae_align_{ts}"
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "ok": all(r.ok for r in results),
        "model_dir": args.model_dir,
        "vae_dir": vae_dir,
        "lib": args.lib,
        "pt_dtype": args.pt_dtype,
        "threads": args.threads,
        "compute_mb": args.compute_mb,
        "results": [r.__dict__ for r in results],
    }
    (out_dir / "summary.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    (out_dir / "summary.md").write_text(md + "\n", encoding="utf-8")
    print(f"[saved] {out_dir / 'summary.json'}")
    print(f"[saved] {out_dir / 'summary.md'}")

    if not all(r.ok for r in results):
        sys.exit(2)


if __name__ == "__main__":
    main()
