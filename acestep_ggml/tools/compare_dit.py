#!/usr/bin/env python3
import argparse
import ctypes
import os
import sys
from typing import List, Optional


def _set_runtime_env_defaults() -> None:
    # Must be set before importing numpy/torch to avoid OpenMP SHM init failures.
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


def parse_int_list(s: Optional[str], expected: int) -> Optional[np.ndarray]:
    if not s:
        return None
    parts = [p.strip() for p in s.split(",") if p.strip()]
    vals = [int(p) for p in parts]
    if expected > 0 and len(vals) != expected:
        raise ValueError(f"length mismatch: expected {expected}, got {len(vals)}")
    return np.asarray(vals, dtype=np.int32)


def pool_mask(mask: np.ndarray, patch: int) -> np.ndarray:
    if patch <= 1:
        return mask.astype(np.int32)
    seq_len = mask.shape[0]
    seq_len_p = (seq_len + patch - 1) // patch
    pooled = np.zeros((seq_len_p,), dtype=np.int32)
    for p in range(seq_len_p):
        start = p * patch
        end = min(seq_len, start + patch)
        pooled[p] = 1 if np.any(mask[start:end] != 0) else 0
    return pooled


def load_ggml_dit(
    lib_path: str,
    model_dir: str,
    hidden_states: np.ndarray,
    context_latents: np.ndarray,
    encoder_hidden_states: np.ndarray,
    attention_mask: Optional[np.ndarray],
    encoder_attention_mask: Optional[np.ndarray],
    timestep: float,
    timestep_r: float,
    threads: int,
    compute_mb: int,
) -> np.ndarray:
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
    lib.ace_ggml_load_dit.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_dit.restype = ctypes.c_int
    lib.ace_ggml_dit_forward.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_float),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_float,
        ctypes.c_float,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.ace_ggml_dit_forward.restype = ctypes.c_int

    ctx = AceCtxPtr()
    params = AceInitParams(n_threads=threads, use_metal=0, compute_buffer_bytes=compute_mb * 1024 * 1024)
    status = lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx))
    if status != ACE_GGML_OK:
        raise RuntimeError("ace_ggml_create failed")

    try:
        status = lib.ace_ggml_load_dit(ctx, model_dir.encode("utf-8"))
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"load_dit failed: {err.decode('utf-8') if err else ''}")

        seq_len = hidden_states.shape[0]
        audio_dim = hidden_states.shape[1]
        enc_len = encoder_hidden_states.shape[0]

        out = np.empty((seq_len, audio_dim), dtype=np.float32)

        status = lib.ace_ggml_dit_forward(
            ctx,
            hidden_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            context_latents.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            encoder_hidden_states.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            attention_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if attention_mask is not None else None,
            encoder_attention_mask.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)) if encoder_attention_mask is not None else None,
            ctypes.c_int32(seq_len),
            ctypes.c_int32(enc_len),
            ctypes.c_float(timestep),
            ctypes.c_float(timestep_r),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.nbytes,
        )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"dit_forward failed: {err.decode('utf-8') if err else ''}")

        return out
    finally:
        lib.ace_ggml_destroy(ctx)


def load_pt_dit(
    model_dir: str,
    hidden_states: np.ndarray,
    context_latents: np.ndarray,
    encoder_hidden_states: np.ndarray,
    attention_mask: Optional[np.ndarray],
    encoder_attention_mask: Optional[np.ndarray],
    timestep: float,
    timestep_r: float,
    pt_dtype: str,
    patch_size: int,
) -> np.ndarray:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    import types
    import importlib.machinery

    def ensure_stub(name: str, attrs: Optional[dict] = None, is_pkg: bool = False):
        mod = types.ModuleType(name)
        mod.__spec__ = importlib.machinery.ModuleSpec(name, None)
        if is_pkg:
            mod.__path__ = []
        mod.__file__ = f"<stub {name}>"
        if attrs:
            for k, v in attrs.items():
                setattr(mod, k, v)
        sys.modules[name] = mod
        return mod

    def stub_hqq() -> None:
        class _HQQStub:
            pass

        def _hqq_getattr(name: str):
            return _HQQStub

        hqq_mod = ensure_stub("hqq", is_pkg=True)
        hqq_mod.__version__ = "0.0.0"
        hqq_core = ensure_stub("hqq.core", is_pkg=True)
        hqq_quant = ensure_stub(
            "hqq.core.quantize",
            attrs={
                "Quantizer": _HQQStub,
                "HQQLinear": _HQQStub,
                "__getattr__": _hqq_getattr,
            },
        )
        hqq_core.quantize = hqq_quant
        hqq_mod.core = hqq_core

    def stub_triton() -> None:
        triton_mod = ensure_stub("triton", is_pkg=True)
        ensure_stub("triton.language", is_pkg=True)
        ensure_stub("triton.runtime", is_pkg=True)
        ensure_stub("triton.compiler", attrs={"CompiledKernel": object})

        class _Config:  # minimal placeholder
            def __init__(self, *args, **kwargs) -> None:
                pass

        triton_mod.Config = _Config

    def stub_torch_internals() -> None:
        # Work around torch._meta_registrations importing prims (can trigger duplicate TORCH_LIBRARY)
        ensure_stub("torch._meta_registrations")
        # Stub the specific decomp/refs used by torch.backends.mps._init to avoid prims import
        ensure_stub("torch._decomp.decompositions", attrs={"native_group_norm_backward": (lambda *a, **k: None)})
        ensure_stub("torch._refs", attrs={"native_group_norm": (lambda *a, **k: None)})
        # Avoid torch.distributed / dtensor imports triggered by transformers
        ensure_stub(
            "torch.distributed",
            attrs={"is_available": (lambda: False), "is_initialized": (lambda: False)},
            is_pkg=True,
        )
        ensure_stub("torch.distributed.tensor", is_pkg=True)
        ensure_stub("torch.distributed.rpc", is_pkg=True)
        # Stub torch._jit_internal to avoid importing distributed during torch init
        def _overload(*args, **kwargs):
            def deco(fn):
                return fn
            return deco

        def boolean_dispatch(*args, **kwargs):
            def deco(fn):
                return fn
            return deco

        class BroadcastingList1(list):
            pass

        class BroadcastingList2(list):
            pass

        class BroadcastingList3(list):
            pass

        ensure_stub(
            "torch._jit_internal",
            attrs={
                "_overload": _overload,
                "boolean_dispatch": boolean_dispatch,
                "BroadcastingList1": BroadcastingList1,
                "BroadcastingList2": BroadcastingList2,
                "BroadcastingList3": BroadcastingList3,
            },
        )

    def stub_torchao() -> None:
        # Stub torchao to avoid triton-dependent quantization path
        class _TorchAoStub:
            def __init__(self, *args, **kwargs) -> None:
                pass

        torchao_mod = ensure_stub("torchao", is_pkg=True)
        torchao_mod.__version__ = "0.0.0"

        def _torchao_getattr(name: str):
            return _TorchAoStub

        torchao_quant = ensure_stub(
            "torchao.quantization",
            attrs={
                "Float8WeightOnlyConfig": _TorchAoStub,
                "Float8DynamicActivationConfig": _TorchAoStub,
                "Float8DynamicActivationFloat8WeightConfig": _TorchAoStub,
                "Int8DynamicActivationConfig": _TorchAoStub,
                "Int8WeightOnlyConfig": _TorchAoStub,
                "Int4WeightOnlyConfig": _TorchAoStub,
                "Int4WeightOnlyConfigPerGroup": _TorchAoStub,
                "__getattr__": _torchao_getattr,
            },
            is_pkg=True,
        )
        torchao_kernel = ensure_stub("torchao.kernel", is_pkg=True)
        torchao_mod.quantization = torchao_quant
        torchao_mod.kernel = torchao_kernel

    # Default to stubbing optional quantization stacks; this script only needs eager CPU forward.
    force_stubs = os.environ.get("ACE_GGML_FORCE_STUBS", "1") == "1"

    if force_stubs:
        stub_hqq()
        stub_triton()
        stub_torchao()
    else:
        try:
            from hqq.core.quantize import HQQLinear as _HQQCheck  # noqa: F401
        except Exception:
            stub_hqq()

        torchao_need_stub = False
        try:
            import torchao  # noqa: F401
            tq = getattr(torchao, "quantization", None)
            required = [
                "Float8WeightOnlyConfig",
                "Float8DynamicActivationConfig",
                "Float8DynamicActivationFloat8WeightConfig",
                "Int8DynamicActivationConfig",
                "Int8WeightOnlyConfig",
                "Int4WeightOnlyConfig",
                "Int4WeightOnlyConfigPerGroup",
            ]
            if tq is None or any(not hasattr(tq, attr) for attr in required):
                torchao_need_stub = True
        except Exception:
            torchao_need_stub = True
        if torchao_need_stub:
            stub_torchao()

        try:
            import triton  # noqa: F401
        except Exception:
            stub_triton()

    try:
        import vector_quantize_pytorch  # noqa: F401
    except Exception:
        dummy = types.ModuleType("vector_quantize_pytorch")

        class ResidualFSQ:  # type: ignore
            def __init__(self, *args, **kwargs) -> None:
                raise RuntimeError("ResidualFSQ is not available in this environment")

        dummy.ResidualFSQ = ResidualFSQ
        sys.modules["vector_quantize_pytorch"] = dummy

    try:
        import einops  # noqa: F401
    except Exception:
        import types

        dummy = types.ModuleType("einops")

        def rearrange(*args, **kwargs):
            raise RuntimeError("einops is not available in this environment")

        dummy.rearrange = rearrange
        sys.modules["einops"] = dummy

    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    sys.path.insert(0, model_dir)
    from configuration_acestep_v15 import AceStepConfig  # type: ignore
    from modeling_acestep_v15_turbo import AceStepDiTModel, create_4d_mask  # type: ignore

    dtype = torch.float32 if pt_dtype == "f32" else torch.bfloat16

    cfg = AceStepConfig.from_pretrained(model_dir)
    model = AceStepDiTModel(cfg)
    state = load_file(os.path.join(model_dir, "model.safetensors"), device="cpu")
    decoder_state = {k[len("decoder."):]: v for k, v in state.items() if k.startswith("decoder.")}
    model.load_state_dict(decoder_state, strict=True)
    model.eval()
    model.to(dtype=dtype)
    model.config._attn_implementation = "eager"

    hs = torch.from_numpy(hidden_states).unsqueeze(0).to(dtype=dtype)
    ctx = torch.from_numpy(context_latents).unsqueeze(0).to(dtype=dtype)
    enc = torch.from_numpy(encoder_hidden_states).unsqueeze(0).to(dtype=dtype)

    attn_mask = None
    if attention_mask is not None:
        attn_mask = torch.from_numpy(attention_mask).unsqueeze(0)
    enc_mask = None
    if encoder_attention_mask is not None:
        enc_mask = torch.from_numpy(encoder_attention_mask).unsqueeze(0)

    t = torch.tensor([timestep], dtype=dtype)
    tr = torch.tensor([timestep_r], dtype=dtype)

    def forward_manual():
        attn_mask_local = attn_mask
        enc_mask_local = enc_mask
        temb_t, timestep_proj_t = model.time_embed(t)
        temb_r, timestep_proj_r = model.time_embed_r(t - tr)
        temb = temb_t + temb_r
        timestep_proj = timestep_proj_t + timestep_proj_r

        hidden = torch.cat([ctx, hs], dim=-1)
        orig_len = hidden.shape[1]
        pad_len = 0
        if hidden.shape[1] % patch_size != 0:
            pad_len = patch_size - (hidden.shape[1] % patch_size)
            hidden = F.pad(hidden, (0, 0, 0, pad_len), mode="constant", value=0)
            if attn_mask_local is not None:
                attn_mask_local = F.pad(attn_mask_local, (0, pad_len), mode="constant", value=0)

        hidden = model.proj_in(hidden)
        enc_hid = model.condition_embedder(enc)

        seq_len_p = hidden.shape[1]
        cache_position = torch.arange(0, seq_len_p, device=hidden.device)
        position_ids = cache_position.unsqueeze(0)

        full_attn_mask = None
        sliding_attn_mask = None
        enc_attn_mask = None

        attn_patch = None
        if attn_mask_local is not None:
            total = seq_len_p * patch_size
            if attn_mask_local.shape[1] != total:
                raise RuntimeError(f"attention_mask length {attn_mask_local.shape[1]} != {total}")
            attn_patch = attn_mask_local.view(attn_mask_local.shape[0], seq_len_p, patch_size)
            attn_patch = (attn_patch.sum(dim=2) > 0)

        if attn_patch is not None:
            full_attn_mask = create_4d_mask(
                seq_len=seq_len_p,
                dtype=hidden.dtype,
                device=hidden.device,
                attention_mask=attn_patch,
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,
            )
            if cfg.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len_p,
                    dtype=hidden.dtype,
                    device=hidden.device,
                    attention_mask=attn_patch,
                    sliding_window=cfg.sliding_window,
                    is_sliding_window=True,
                    is_causal=False,
                )
        else:
            full_attn_mask = create_4d_mask(
                seq_len=seq_len_p,
                dtype=hidden.dtype,
                device=hidden.device,
                attention_mask=None,
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,
            )
            if cfg.use_sliding_window:
                sliding_attn_mask = create_4d_mask(
                    seq_len=seq_len_p,
                    dtype=hidden.dtype,
                    device=hidden.device,
                    attention_mask=None,
                    sliding_window=cfg.sliding_window,
                    is_sliding_window=True,
                    is_causal=False,
                )

        if enc_mask_local is not None:
            max_len = max(seq_len_p, enc_hid.shape[1])
            enc_attn_mask = create_4d_mask(
                seq_len=max_len,
                dtype=hidden.dtype,
                device=hidden.device,
                attention_mask=enc_mask_local,
                sliding_window=None,
                is_sliding_window=False,
                is_causal=False,
            )
            enc_attn_mask = enc_attn_mask[:, :, :seq_len_p, :enc_hid.shape[1]]

        self_attn_mask_mapping = {
            "full_attention": full_attn_mask,
            "sliding_attention": sliding_attn_mask,
        }

        position_embeddings = model.rotary_emb(hidden, position_ids)

        for layer in model.layers:
            layer_outputs = layer(
                hidden,
                position_embeddings,
                timestep_proj,
                self_attn_mask_mapping[layer.attention_type],
                position_ids,
                past_key_value=None,
                output_attentions=False,
                use_cache=False,
                cache_position=cache_position,
                encoder_hidden_states=enc_hid,
                encoder_attention_mask=enc_attn_mask,
            )
            hidden = layer_outputs[0]

        shift, scale = (model.scale_shift_table + temb.unsqueeze(1)).chunk(2, dim=1)
        hidden = (model.norm_out(hidden) * (1 + scale) + shift).type_as(hidden)
        hidden = model.proj_out(hidden)
        hidden = hidden[:, :orig_len, :]
        return hidden

    with torch.no_grad():
        out = forward_manual()
    return out[0].float().cpu().numpy().astype(np.float32)


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a_flat = a.reshape(-1)
    b_flat = b.reshape(-1)
    denom = (np.linalg.norm(a_flat) * np.linalg.norm(b_flat)) + 1e-12
    return float(np.dot(a_flat, b_flat) / denom)


def main() -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")
    os.environ.setdefault("KMP_USE_SHM", "0")
    os.environ.setdefault("KMP_DISABLE_SHM", "1")
    os.environ.setdefault("KMP_SHM", "0")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")
    ap.add_argument("--seq-len", type=int, default=8)
    ap.add_argument("--enc-len", type=int, default=8)
    ap.add_argument("--timestep", type=float, default=0.5)
    ap.add_argument("--timestep-r", type=float, default=0.0)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--mask", default="")
    ap.add_argument("--enc-mask", default="")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--compute-mb", type=int, default=768)
    ap.add_argument("--pt-dtype", default="bf16", choices=["f32", "bf16"])
    ap.add_argument("--debug-graph", action="store_true")
    ap.add_argument("--ggml-only", action="store_true")
    ap.add_argument("--pt-only", action="store_true")
    ap.add_argument("--max-layers", type=int, default=0)
    ap.add_argument("--graph-size", type=int, default=0)
    args = ap.parse_args()

    if args.ggml_only and args.pt_only:
        raise ValueError("--ggml-only and --pt-only cannot both be set")

    if args.enc_len <= 0:
        raise ValueError("--enc-len must be > 0 (cross-attn requires encoder states)")

    if args.debug_graph:
        os.environ["ACE_GGML_DEBUG_GRAPH"] = "1"
    if args.max_layers > 0:
        os.environ["ACE_GGML_DIT_MAX_LAYERS"] = str(args.max_layers)
    if args.graph_size > 0:
        os.environ["ACE_GGML_GRAPH_SIZE"] = str(args.graph_size)

    cfg_path = os.path.join(args.model_dir, "config.json")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = f.read()
    cfg_data = __import__("json").loads(cfg)
    audio_dim = int(cfg_data["audio_acoustic_hidden_dim"])
    in_channels = int(cfg_data["in_channels"])
    hidden_size = int(cfg_data["hidden_size"])
    patch_size = int(cfg_data["patch_size"])
    ctx_dim = in_channels - audio_dim

    if ctx_dim <= 0:
        raise RuntimeError("invalid in_channels/audio_acoustic_hidden_dim in config")

    rng = np.random.default_rng(args.seed)
    hidden_states = rng.standard_normal((args.seq_len, audio_dim), dtype=np.float32)
    context_latents = rng.standard_normal((args.seq_len, ctx_dim), dtype=np.float32)
    encoder_hidden_states = rng.standard_normal((args.enc_len, hidden_size), dtype=np.float32)

    attention_mask = parse_int_list(args.mask, args.seq_len)
    encoder_attention_mask = parse_int_list(args.enc_mask, args.enc_len)

    out_pt = None
    out_ggml = None

    if not args.pt_only:
        print(">>> ggml forward...", file=sys.stderr, flush=True)
        out_ggml = load_ggml_dit(
            args.lib,
            args.model_dir,
            hidden_states,
            context_latents,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            args.timestep,
            args.timestep_r,
            args.threads,
            args.compute_mb,
        )
        print(">>> ggml forward done", file=sys.stderr, flush=True)

    if not args.ggml_only:
        print(">>> pt forward...", file=sys.stderr, flush=True)
        out_pt = load_pt_dit(
            args.model_dir,
            hidden_states,
            context_latents,
            encoder_hidden_states,
            attention_mask,
            encoder_attention_mask,
            args.timestep,
            args.timestep_r,
            args.pt_dtype,
            patch_size,
        )
        print(">>> pt forward done", file=sys.stderr, flush=True)

    if out_pt is not None and out_ggml is not None:
        diff = out_pt - out_ggml
        mae = float(np.mean(np.abs(diff)))
        max_err = float(np.max(np.abs(diff)))
        cos = cosine_similarity(out_pt, out_ggml)

        print("shape", out_pt.shape)
        print(f"mae={mae:.6f} max={max_err:.6f} cos={cos:.6f}")
        print("pt_head", out_pt.reshape(-1)[:8])
        print("ggml_head", out_ggml.reshape(-1)[:8])
    elif out_ggml is not None:
        print("ggml only: ok")
        print("ggml_head", out_ggml.reshape(-1)[:8])
    elif out_pt is not None:
        print("pt only: ok")
        print("pt_head", out_pt.reshape(-1)[:8])


if __name__ == "__main__":
    main()
