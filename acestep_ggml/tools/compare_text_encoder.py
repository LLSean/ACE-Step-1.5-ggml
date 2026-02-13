#!/usr/bin/env python3
import argparse
import ctypes
import os
import sys
from typing import List

import numpy as np


ACE_GGML_OK = 0


def parse_tokens(token_str: str) -> List[int]:
    tokens = []
    for part in token_str.split(','):
        part = part.strip()
        if not part:
            continue
        tokens.append(int(part))
    return tokens


def load_ggml_embeddings(
    lib_path: str,
    model_dir: str,
    tokens: List[int],
    threads: int,
    compute_mb: int,
    mode: str,
    layers: int,
    apply_final_norm: bool,
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
    lib.ace_ggml_load_text_encoder.argtypes = [AceCtxPtr, ctypes.c_char_p]
    lib.ace_ggml_load_text_encoder.restype = ctypes.c_int
    lib.ace_ggml_text_encoder_forward.argtypes = [AceCtxPtr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.ace_ggml_text_encoder_forward.restype = ctypes.c_int
    lib.ace_ggml_text_encoder_forward_embeddings.argtypes = [AceCtxPtr, ctypes.POINTER(ctypes.c_int32), ctypes.c_int32, ctypes.POINTER(ctypes.c_float), ctypes.c_size_t]
    lib.ace_ggml_text_encoder_forward_embeddings.restype = ctypes.c_int
    lib.ace_ggml_text_encoder_forward_layers.argtypes = [
        AceCtxPtr,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.c_int32,
        ctypes.POINTER(ctypes.c_float),
        ctypes.c_size_t,
    ]
    lib.ace_ggml_text_encoder_forward_layers.restype = ctypes.c_int

    ctx = AceCtxPtr()
    params = AceInitParams(n_threads=threads, use_metal=0, compute_buffer_bytes=compute_mb * 1024 * 1024)
    status = lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(ctx))
    if status != ACE_GGML_OK:
        raise RuntimeError("ace_ggml_create failed")

    try:
        status = lib.ace_ggml_load_text_encoder(ctx, model_dir.encode("utf-8"))
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"load_text_encoder failed: {err.decode('utf-8') if err else ''}")

        token_arr = np.asarray(tokens, dtype=np.int32)
        hidden = 1024
        out = np.empty((hidden * len(tokens),), dtype=np.float32)

        if layers >= 0:
            status = lib.ace_ggml_text_encoder_forward_layers(
                ctx,
                token_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                None,
                ctypes.c_int32(len(tokens)),
                ctypes.c_int32(layers),
                ctypes.c_int32(1 if apply_final_norm else 0),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.nbytes,
            )
        elif mode == "embed":
            status = lib.ace_ggml_text_encoder_forward_embeddings(
                ctx,
                token_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(len(tokens)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.nbytes,
            )
        else:
            status = lib.ace_ggml_text_encoder_forward(
                ctx,
                token_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                ctypes.c_int32(len(tokens)),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                out.nbytes,
            )
        if status != ACE_GGML_OK:
            err = lib.ace_ggml_last_error(ctx)
            raise RuntimeError(f"text_encoder_forward failed: {err.decode('utf-8') if err else ''}")

        return out.reshape(len(tokens), hidden)
    finally:
        lib.ace_ggml_destroy(ctx)


def load_pt_embeddings(model_dir: str, tokens: List[int], pt_dtype: str, mode: str, layers: int, apply_final_norm: bool) -> np.ndarray:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
    os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    os.environ.setdefault("KMP_INIT_AT_FORK", "FALSE")

    import torch
    from transformers import AutoModel
    from transformers.masking_utils import create_causal_mask

    dtype = torch.float32 if pt_dtype == "f32" else torch.bfloat16
    model = AutoModel.from_pretrained(model_dir, torch_dtype=dtype)
    model.eval()
    input_ids = torch.tensor([tokens], dtype=torch.long)
    with torch.no_grad():
        if layers >= 0:
            hidden_states = model.embed_tokens(input_ids)
            cache_position = torch.arange(0, hidden_states.shape[1], device=hidden_states.device)
            position_ids = cache_position.unsqueeze(0)
            mask_kwargs = {
                "config": model.config,
                "input_embeds": hidden_states,
                "attention_mask": None,
                "cache_position": cache_position,
                "past_key_values": None,
                "position_ids": position_ids,
            }
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            if getattr(model, "has_sliding_layers", False):
                from transformers.masking_utils import create_sliding_window_causal_mask
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

            position_embeddings = model.rotary_emb(hidden_states, position_ids)
            if layers == 0:
                hidden = hidden_states.float()
            else:
                for idx, layer in enumerate(model.layers):
                    hidden_states = layer(
                        hidden_states,
                        attention_mask=causal_mask_mapping[layer.attention_type],
                        position_ids=position_ids,
                        past_key_values=None,
                        use_cache=False,
                        cache_position=cache_position,
                        position_embeddings=position_embeddings,
                    )
                    if idx + 1 >= layers:
                        break
                if apply_final_norm and layers >= model.config.num_hidden_layers:
                    hidden_states = model.norm(hidden_states)
                hidden = hidden_states.float()
        elif mode == "embed":
            hidden = model.embed_tokens(input_ids).float()
        else:
            outputs = model(input_ids)
            hidden = outputs[0].float()
    return hidden[0].cpu().numpy().astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--tokens", required=True, help="comma separated token ids")
    ap.add_argument("--lib", default="/tmp/ace_ggml_build/libacestep_ggml.dylib")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--compute-mb", type=int, default=512)
    ap.add_argument("--pt-dtype", default="bf16", choices=["f32", "bf16"])
    ap.add_argument("--mode", default="full", choices=["full", "embed"])
    ap.add_argument("--layers", type=int, default=-1, help="0 for embeddings, 1 for first layer, etc.")
    ap.add_argument("--final-norm", action="store_true", help="apply final norm when layers >= num_hidden_layers")
    args = ap.parse_args()

    tokens = parse_tokens(args.tokens)
    if not tokens:
        print("no tokens provided", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.lib):
        print(f"lib not found: {args.lib}", file=sys.stderr)
        sys.exit(1)

    ggml_out = load_ggml_embeddings(
        args.lib, args.model_dir, tokens, args.threads, args.compute_mb, args.mode, args.layers, args.final_norm
    )
    pt_out = load_pt_embeddings(args.model_dir, tokens, args.pt_dtype, args.mode, args.layers, args.final_norm)

    if ggml_out.shape != pt_out.shape:
        print(f"shape mismatch ggml={ggml_out.shape} pt={pt_out.shape}", file=sys.stderr)
        sys.exit(1)

    diff = ggml_out - pt_out
    mae = float(np.mean(np.abs(diff)))
    max_err = float(np.max(np.abs(diff)))

    # cosine similarity per token
    def cosine(a, b):
        denom = (np.linalg.norm(a) * np.linalg.norm(b))
        if denom == 0:
            return 0.0
        return float(np.dot(a, b) / denom)

    cosines = [cosine(ggml_out[i], pt_out[i]) for i in range(ggml_out.shape[0])]
    cos_mean = float(np.mean(cosines))

    print(f"tokens={len(tokens)} hidden={ggml_out.shape[1]}")
    print(f"mae={mae:.6f}")
    print(f"max_err={max_err:.6f}")
    print(f"cosine_mean={cos_mean:.6f}")


if __name__ == "__main__":
    main()
