#!/usr/bin/env python3
"""
Run non-ggml ACE-Step inference with real style + lyric inputs.

Defaults:
- style: acestep_ggml/reports/real_case/style_neo_soul.txt
- lyric: acestep_ggml/reports/real_case/lyric_neo_soul.txt
"""

from __future__ import annotations

import atexit
import argparse
import ctypes
import gc
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


class _TokenizedBatch:
    def __init__(self, input_ids, attention_mask):
        self.input_ids = input_ids
        self.attention_mask = attention_mask


class GGMLJSONTokenizerAdapter:
    """
    Adapter that mimics HuggingFace tokenizer call shape but uses tokenizer.json encoding,
    matching ggml tool-side tokenization behavior.
    """

    def __init__(self, hf_tokenizer, tokenizer_dir: Path):
        from tokenizers import Tokenizer  # type: ignore

        tok_json = tokenizer_dir / "tokenizer.json"
        if not tok_json.exists():
            raise FileNotFoundError(f"tokenizer.json not found: {tok_json}")

        self._hf = hf_tokenizer
        self._tok = Tokenizer.from_file(str(tok_json))
        self.pad_token_id = hf_tokenizer.pad_token_id
        if self.pad_token_id is None:
            # Keep behavior robust even if pad token is unset.
            self.pad_token_id = hf_tokenizer.eos_token_id if hf_tokenizer.eos_token_id is not None else 0

    def __getattr__(self, name):
        return getattr(self._hf, name)

    def encode(self, text: str, add_special_tokens: bool = False, truncation: bool = False, max_length: int | None = None, **kwargs):
        if add_special_tokens:
            ids = self._tok.encode(text).ids
        else:
            ids = self._hf.encode(text, add_special_tokens=False, **kwargs)
        if truncation and max_length is not None and max_length > 0 and len(ids) > max_length:
            ids = ids[:max_length]
        return ids

    def __call__(
        self,
        text,
        padding="longest",
        truncation: bool = False,
        max_length: int | None = None,
        return_tensors: str | None = None,
        **kwargs,
    ):
        import torch

        if isinstance(text, str):
            texts = [text]
        else:
            texts = list(text)

        ids_list = [self._tok.encode(t).ids for t in texts]
        if truncation and max_length is not None and max_length > 0:
            ids_list = [ids[:max_length] for ids in ids_list]

        if padding in (True, "longest"):
            pad_to = max(len(ids) for ids in ids_list) if ids_list else 0
        elif padding == "max_length" and max_length is not None and max_length > 0:
            pad_to = max_length
        else:
            pad_to = None

        padded = []
        masks = []
        for ids in ids_list:
            if pad_to is None:
                cur = ids
                mask = [1] * len(ids)
            else:
                n_pad = max(0, pad_to - len(ids))
                cur = ids + [self.pad_token_id] * n_pad
                mask = [1] * len(ids) + [0] * n_pad
            padded.append(cur)
            masks.append(mask)

        input_ids = torch.tensor(padded, dtype=torch.long)
        attention_mask = torch.tensor(masks, dtype=torch.long)

        if return_tensors is None or return_tensors == "pt":
            return _TokenizedBatch(input_ids=input_ids, attention_mask=attention_mask)
        return {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
        }


class AceInitParams(ctypes.Structure):
    _fields_ = [
        ("n_threads", ctypes.c_int32),
        ("use_metal", ctypes.c_int32),
        ("compute_buffer_bytes", ctypes.c_size_t),
    ]


class GGMLCAPIBridge:
    ACE_GGML_OK = 0

    def __init__(self, lib_path: Path, n_threads: int, compute_buffer_mb: int, use_metal: bool = False):
        self.lib_path = lib_path
        self.lib = ctypes.CDLL(str(lib_path))
        self.ctx = ctypes.c_void_p()
        self.closed = False
        self.use_metal = bool(use_metal)

        self._bind()
        params = AceInitParams(
            n_threads=int(n_threads),
            use_metal=1 if self.use_metal else 0,
            compute_buffer_bytes=int(compute_buffer_mb) * 1024 * 1024,
        )
        st = self.lib.ace_ggml_create(ctypes.byref(params), ctypes.byref(self.ctx))
        self._ensure_ok(st, "ace_ggml_create")
        atexit.register(self.close)

        self.audio_channels = 0
        self.hop_length = 0

    def _bind(self) -> None:
        self.lib.ace_ggml_create.argtypes = [ctypes.POINTER(AceInitParams), ctypes.POINTER(ctypes.c_void_p)]
        self.lib.ace_ggml_create.restype = ctypes.c_int
        self.lib.ace_ggml_destroy.argtypes = [ctypes.c_void_p]
        self.lib.ace_ggml_destroy.restype = None
        self.lib.ace_ggml_last_error.argtypes = [ctypes.c_void_p]
        self.lib.ace_ggml_last_error.restype = ctypes.c_char_p

        self.lib.ace_ggml_load_text_encoder.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ace_ggml_load_text_encoder.restype = ctypes.c_int
        self.lib.ace_ggml_load_dit.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ace_ggml_load_dit.restype = ctypes.c_int
        self.lib.ace_ggml_load_vae.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        self.lib.ace_ggml_load_vae.restype = ctypes.c_int
        self.lib.ace_ggml_vae_get_info.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
            ctypes.POINTER(ctypes.c_int32),
        ]
        self.lib.ace_ggml_vae_get_info.restype = ctypes.c_int

        self.lib.ace_ggml_text_encoder_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.ace_ggml_text_encoder_forward.restype = ctypes.c_int
        self.lib.ace_ggml_text_encoder_forward_embeddings.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_int32),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.ace_ggml_text_encoder_forward_embeddings.restype = ctypes.c_int

        self.lib.ace_ggml_vae_decode.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_int32,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_size_t,
        ]
        self.lib.ace_ggml_vae_decode.restype = ctypes.c_int

        self.lib.ace_ggml_dit_forward.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float),   # hidden_states
            ctypes.POINTER(ctypes.c_float),   # context_latents
            ctypes.POINTER(ctypes.c_float),   # encoder_hidden_states
            ctypes.POINTER(ctypes.c_int32),   # attention_mask
            ctypes.POINTER(ctypes.c_int32),   # encoder_attention_mask
            ctypes.c_int32,                   # seq_len
            ctypes.c_int32,                   # enc_len
            ctypes.c_float,                   # timestep
            ctypes.c_float,                   # timestep_r
            ctypes.POINTER(ctypes.c_float),   # out
            ctypes.c_size_t,                  # out_size
        ]
        self.lib.ace_ggml_dit_forward.restype = ctypes.c_int

    def _last_error(self) -> str:
        msg = self.lib.ace_ggml_last_error(self.ctx)
        return msg.decode("utf-8", errors="replace") if msg else "unknown error"

    def _ensure_ok(self, status: int, where: str) -> None:
        if status != self.ACE_GGML_OK:
            raise RuntimeError(f"{where} failed: {self._last_error()} (status={status})")

    def close(self) -> None:
        if not self.closed and self.ctx:
            self.lib.ace_ggml_destroy(self.ctx)
            self.closed = True

    def load_text_encoder(self, model_dir: Path) -> None:
        st = self.lib.ace_ggml_load_text_encoder(self.ctx, str(model_dir).encode("utf-8"))
        self._ensure_ok(st, "ace_ggml_load_text_encoder")

    def load_dit(self, model_dir: Path) -> None:
        st = self.lib.ace_ggml_load_dit(self.ctx, str(model_dir).encode("utf-8"))
        self._ensure_ok(st, "ace_ggml_load_dit")

    def load_vae(self, model_dir: Path) -> None:
        st = self.lib.ace_ggml_load_vae(self.ctx, str(model_dir).encode("utf-8"))
        self._ensure_ok(st, "ace_ggml_load_vae")

        latent_ch = ctypes.c_int32(0)
        audio_ch = ctypes.c_int32(0)
        hop = ctypes.c_int32(0)
        st = self.lib.ace_ggml_vae_get_info(self.ctx, ctypes.byref(latent_ch), ctypes.byref(audio_ch), ctypes.byref(hop))
        self._ensure_ok(st, "ace_ggml_vae_get_info")
        self.audio_channels = int(audio_ch.value)
        self.hop_length = int(hop.value)

    def text_forward_full(self, token_ids: np.ndarray, hidden_dim: int) -> np.ndarray:
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        n_tokens = int(token_ids.shape[0])
        out = np.empty((n_tokens * hidden_dim,), dtype=np.float32)
        st = self.lib.ace_ggml_text_encoder_forward(
            self.ctx,
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n_tokens),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.nbytes),
        )
        self._ensure_ok(st, "ace_ggml_text_encoder_forward")
        return out.reshape(n_tokens, hidden_dim)

    def text_forward_embeddings(self, token_ids: np.ndarray, hidden_dim: int) -> np.ndarray:
        token_ids = np.ascontiguousarray(token_ids, dtype=np.int32)
        n_tokens = int(token_ids.shape[0])
        out = np.empty((n_tokens * hidden_dim,), dtype=np.float32)
        st = self.lib.ace_ggml_text_encoder_forward_embeddings(
            self.ctx,
            token_ids.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(n_tokens),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.nbytes),
        )
        self._ensure_ok(st, "ace_ggml_text_encoder_forward_embeddings")
        return out.reshape(n_tokens, hidden_dim)

    def vae_decode_tfirst(self, latents_tfirst: np.ndarray) -> np.ndarray:
        """
        Args:
            latents_tfirst: [T, C] float32 contiguous
        Returns:
            audio [samples, channels] float32
        """
        if self.audio_channels <= 0 or self.hop_length <= 0:
            raise RuntimeError("VAE not initialized in ggml bridge")

        lat = np.ascontiguousarray(latents_tfirst, dtype=np.float32)
        n_frames = int(lat.shape[0])
        out_samples = n_frames * self.hop_length
        out = np.empty((out_samples * self.audio_channels,), dtype=np.float32)
        st = self.lib.ace_ggml_vae_decode(
            self.ctx,
            lat.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_int32(n_frames),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.nbytes),
        )
        self._ensure_ok(st, "ace_ggml_vae_decode")
        return out.reshape(out_samples, self.audio_channels)

    def dit_forward_tfirst(
        self,
        hidden_states_tfirst: np.ndarray,
        context_latents_tfirst: np.ndarray,
        encoder_hidden_states_tfirst: np.ndarray,
        attention_mask: np.ndarray,
        encoder_attention_mask: np.ndarray,
        timestep: float,
        timestep_r: float,
    ) -> np.ndarray:
        """
        Args:
            hidden_states_tfirst: [T, D]
            context_latents_tfirst: [T, Ctx]
            encoder_hidden_states_tfirst: [L, H]
            attention_mask: [T] int32
            encoder_attention_mask: [L] int32
        Returns:
            vt [T, D] float32
        """
        hs = np.ascontiguousarray(hidden_states_tfirst, dtype=np.float32)
        ctx = np.ascontiguousarray(context_latents_tfirst, dtype=np.float32)
        enc = np.ascontiguousarray(encoder_hidden_states_tfirst, dtype=np.float32)
        am = np.ascontiguousarray(attention_mask, dtype=np.int32)
        eam = np.ascontiguousarray(encoder_attention_mask, dtype=np.int32)

        if hs.ndim != 2 or ctx.ndim != 2 or enc.ndim != 2:
            raise ValueError("dit inputs must be 2D arrays [T,D]/[T,Ctx]/[L,H]")
        seq_len = int(hs.shape[0])
        enc_len = int(enc.shape[0])
        out = np.empty_like(hs, dtype=np.float32)

        st = self.lib.ace_ggml_dit_forward(
            self.ctx,
            hs.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctx.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            enc.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            am.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            eam.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            ctypes.c_int32(seq_len),
            ctypes.c_int32(enc_len),
            ctypes.c_float(float(timestep)),
            ctypes.c_float(float(timestep_r)),
            out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            ctypes.c_size_t(out.nbytes),
        )
        self._ensure_ok(st, "ace_ggml_dit_forward")
        return out


def _format_instruction(instruction: str) -> str:
    return instruction if instruction.endswith(":") else instruction + ":"


def _build_meta_string(bpm: int | None, keyscale: str, timesignature: str, duration: float | None) -> str:
    bpm_str = str(bpm) if bpm is not None else "N/A"
    keyscale_str = keyscale if keyscale else "N/A"
    timesig_str = timesignature if timesignature else "N/A"
    if duration is None:
        duration_str = "30 seconds"
    else:
        duration_str = f"{int(duration)} seconds"
    return (
        f"- bpm: {bpm_str}\n"
        f"- timesignature: {timesig_str}\n"
        f"- keyscale: {keyscale_str}\n"
        f"- duration: {duration_str}\n"
    )


def _build_style_prompt(caption: str, instruction: str, meta_str: str) -> str:
    return SFT_GEN_PROMPT.format(_format_instruction(instruction), caption, meta_str)


def _build_lyric_prompt(lyrics: str, language: str) -> str:
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def _resolve_tokenizer_dir(project_root: Path, tokenizer_dir_arg: str) -> Path:
    if tokenizer_dir_arg:
        p = Path(tokenizer_dir_arg)
        if not p.is_absolute():
            p = project_root / p
        return p
    checkpoints_tok = project_root / "checkpoints" / "Qwen3-Embedding-0.6B"
    if checkpoints_tok.exists():
        return checkpoints_tok
    return project_root / "Ace-Step1.5" / "Qwen3-Embedding-0.6B"


def _resolve_path_with_default(project_root: Path, path_arg: str, default_rel: str) -> Path:
    if path_arg:
        p = Path(path_arg)
        if not p.is_absolute():
            p = project_root / p
        return p
    return project_root / default_rel


def _install_ggml_text_encoder_backend(dit_handler, bridge: GGMLCAPIBridge) -> None:
    import torch

    hidden_dim = int(dit_handler.text_encoder.config.hidden_size)

    def infer_text_embeddings_ggml(self, text_token_idss):
        if not isinstance(text_token_idss, torch.Tensor):
            text_token_idss = torch.tensor(text_token_idss, dtype=torch.long)
        ids = text_token_idss.detach().cpu().numpy().astype(np.int32, copy=False)
        states = [bridge.text_forward_full(row, hidden_dim) for row in ids]
        out = np.stack(states, axis=0)
        return torch.from_numpy(out).to(self.device).to(self.dtype)

    def infer_lyric_embeddings_ggml(self, lyric_token_ids):
        if not isinstance(lyric_token_ids, torch.Tensor):
            lyric_token_ids = torch.tensor(lyric_token_ids, dtype=torch.long)
        ids = lyric_token_ids.detach().cpu().numpy().astype(np.int32, copy=False)
        states = [bridge.text_forward_embeddings(row, hidden_dim) for row in ids]
        out = np.stack(states, axis=0)
        return torch.from_numpy(out).to(self.device).to(self.dtype)

    dit_handler.infer_text_embeddings = types.MethodType(infer_text_embeddings_ggml, dit_handler)
    dit_handler.infer_lyric_embeddings = types.MethodType(infer_lyric_embeddings_ggml, dit_handler)


def _ggml_shift_schedule(shift: float) -> list[float]:
    s1 = [1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]
    s2 = [1.0, 0.9333333333, 0.8571428571, 0.7692307692, 0.6666666667, 0.5454545455, 0.4, 0.2222222222]
    s3 = [1.0, 0.9545454545, 0.9, 0.8333333333, 0.75, 0.6428571429, 0.5, 0.3]
    d1 = abs(shift - 1.0)
    d2 = abs(shift - 2.0)
    d3 = abs(shift - 3.0)
    if d1 <= d2 and d1 <= d3:
        return s1
    if d2 <= d1 and d2 <= d3:
        return s2
    return s3


def _install_ggml_dit_backend(dit_handler, bridge: GGMLCAPIBridge) -> None:
    import torch
    decoder = dit_handler.model.decoder
    original_forward = decoder.forward

    def _timestep_at(ts, b: int) -> float:
        if isinstance(ts, torch.Tensor):
            if ts.ndim == 0:
                return float(ts.item())
            if ts.ndim >= 1:
                return float(ts[b].item())
        if isinstance(ts, (list, tuple)):
            return float(ts[b])
        return float(ts)

    def decoder_forward_ggml(
        self,
        hidden_states,
        timestep,
        timestep_r,
        attention_mask,
        encoder_hidden_states,
        encoder_attention_mask,
        context_latents,
        use_cache=None,
        past_key_values=None,
        cache_position=None,
        position_ids=None,
        output_attentions=False,
        return_hidden_states=None,
        custom_layers_config=None,
        enable_early_exit=False,
        **flash_attn_kwargs,
    ):
        del use_cache, cache_position, position_ids, return_hidden_states, custom_layers_config, enable_early_exit, flash_attn_kwargs

        # Keep original behavior if shapes are unexpected.
        if hidden_states is None or hidden_states.dim() != 3:
            return original_forward(
                hidden_states=hidden_states,
                timestep=timestep,
                timestep_r=timestep_r,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
                use_cache=use_cache,
                past_key_values=past_key_values,
                cache_position=cache_position,
                position_ids=position_ids,
                output_attentions=output_attentions,
                return_hidden_states=return_hidden_states,
                custom_layers_config=custom_layers_config,
                enable_early_exit=enable_early_exit,
                **flash_attn_kwargs,
            )

        hs_np = np.ascontiguousarray(hidden_states.detach().float().cpu().numpy(), dtype=np.float32)
        ctx_np = np.ascontiguousarray(context_latents.detach().float().cpu().numpy(), dtype=np.float32)
        enc_np = np.ascontiguousarray(encoder_hidden_states.detach().float().cpu().numpy(), dtype=np.float32)

        if attention_mask is None:
            attn_np = np.ones((hs_np.shape[0], hs_np.shape[1]), dtype=np.int32)
        else:
            attn_np = np.ascontiguousarray((attention_mask.detach().cpu().numpy() > 0).astype(np.int32), dtype=np.int32)

        if encoder_attention_mask is None:
            enc_attn_np = np.ones((enc_np.shape[0], enc_np.shape[1]), dtype=np.int32)
        else:
            enc_attn_np = np.ascontiguousarray((encoder_attention_mask.detach().cpu().numpy() > 0).astype(np.int32), dtype=np.int32)

        bsz, seq_len, audio_dim = hs_np.shape
        out_np = np.empty((bsz, seq_len, audio_dim), dtype=np.float32)
        for b in range(bsz):
            out_np[b] = bridge.dit_forward_tfirst(
                hidden_states_tfirst=hs_np[b],
                context_latents_tfirst=ctx_np[b],
                encoder_hidden_states_tfirst=enc_np[b],
                attention_mask=attn_np[b],
                encoder_attention_mask=enc_attn_np[b],
                timestep=_timestep_at(timestep, b),
                timestep_r=_timestep_at(timestep_r, b),
            )

        pred = torch.from_numpy(out_np).to(hidden_states.device).to(hidden_states.dtype)
        outputs = (pred, past_key_values)
        if output_attentions:
            outputs += (None,)
        return outputs

    decoder.forward = types.MethodType(decoder_forward_ggml, decoder)
    # Mark runtime backend so handler logs can report the actual DiT path.
    setattr(dit_handler, "_ggml_dit_backend", "ggml-capi")
    setattr(dit_handler, "_ggml_dit_decoder_forward_hooked", True)


def _install_ggml_vae_backend(
    dit_handler,
    bridge: GGMLCAPIBridge,
    chunk_size_default: int,
    overlap_default: int,
    profile_chunks: bool = False,
) -> None:
    import torch

    def tiled_decode_ggml(self, latents, chunk_size=None, overlap: int | None = None, offload_wav_to_cpu=None):
        """
        latents: [B, C, T] -> return [B, C_audio, samples]
        """
        if not isinstance(latents, torch.Tensor):
            raise TypeError(f"latents must be torch.Tensor, got {type(latents)!r}")
        if latents.dim() != 3:
            raise ValueError(f"latents must have shape [B,C,T], got {tuple(latents.shape)}")

        B, C, T = latents.shape
        lat_np = latents.detach().float().cpu().numpy()

        if chunk_size is None:
            chunk_size = int(chunk_size_default)
        chunk_size = int(chunk_size)
        if chunk_size <= 0:
            chunk_size = int(chunk_size_default)

        if overlap is None:
            overlap = int(overlap_default)
        else:
            overlap = int(overlap)
        if overlap < 0:
            overlap = 0

        profile = bool(profile_chunks)
        total_t0 = time.perf_counter()

        if T <= chunk_size:
            wavs = []
            for i in range(B):
                chunk_t0 = time.perf_counter()
                lat_tfirst = np.ascontiguousarray(lat_np[i].transpose(1, 0), dtype=np.float32)
                wav_tc = bridge.vae_decode_tfirst(lat_tfirst)  # [samples, channels]
                wavs.append(wav_tc.T)  # [channels, samples]
                if profile:
                    dt = time.perf_counter() - chunk_t0
                    print(f"[ggml-vae] b={i} chunk=0/1 frames={T} decode={dt:.3f}s out_samples={wav_tc.shape[0]}")
            out_np = np.stack(wavs, axis=0).astype(np.float32, copy=False)
            out = torch.from_numpy(out_np)
            if offload_wav_to_cpu is None:
                offload_wav_to_cpu = self._should_offload_wav_to_cpu()
            if not offload_wav_to_cpu:
                out = out.to(self.device)
            if profile:
                print(f"[ggml-vae] total_decode={time.perf_counter() - total_t0:.3f}s")
            return out

        max_overlap = max(0, (chunk_size // 2) - 1)
        if overlap > max_overlap:
            overlap = max_overlap
        stride = chunk_size - 2 * overlap
        if stride <= 0:
            overlap = max(0, (chunk_size // 4))
            stride = chunk_size - 2 * overlap
            if stride <= 0:
                stride = max(1, chunk_size)
                overlap = 0

        num_steps = int(math.ceil(T / float(stride)))
        if profile:
            print(
                f"[ggml-vae] start tiled decode: B={B} C={C} T={T} "
                f"chunk_size={chunk_size} overlap={overlap} stride={stride} steps={num_steps}"
            )
        wavs = []
        for b in range(B):
            parts = []
            upsample_factor = None
            for i in range(num_steps):
                core_start = i * stride
                core_end = min(core_start + stride, T)
                win_start = max(0, core_start - overlap)
                win_end = min(T, core_end + overlap)

                chunk_t0 = time.perf_counter()
                lat_chunk_tfirst = np.ascontiguousarray(lat_np[b, :, win_start:win_end].transpose(1, 0), dtype=np.float32)
                wav_chunk_tc = bridge.vae_decode_tfirst(lat_chunk_tfirst)  # [samples, channels]
                chunk_dt = time.perf_counter() - chunk_t0

                if upsample_factor is None:
                    upsample_factor = float(wav_chunk_tc.shape[0]) / float(max(1, (win_end - win_start)))

                added_start = core_start - win_start
                added_end = win_end - core_end
                trim_start = int(round(added_start * upsample_factor))
                trim_end = int(round(added_end * upsample_factor))
                end_idx = wav_chunk_tc.shape[0] - trim_end if trim_end > 0 else wav_chunk_tc.shape[0]
                part = wav_chunk_tc[trim_start:end_idx, :]
                parts.append(part)
                if profile:
                    print(
                        f"[ggml-vae] b={b} chunk={i+1}/{num_steps} "
                        f"window=[{win_start},{win_end}) frames={win_end - win_start} "
                        f"decode={chunk_dt:.3f}s kept_samples={part.shape[0]}"
                    )

            full_tc = np.concatenate(parts, axis=0) if parts else np.zeros((0, bridge.audio_channels), dtype=np.float32)
            wavs.append(full_tc.T)

        out_np = np.stack(wavs, axis=0).astype(np.float32, copy=False)
        out = torch.from_numpy(out_np)

        if offload_wav_to_cpu is None:
            offload_wav_to_cpu = self._should_offload_wav_to_cpu()
        if not offload_wav_to_cpu:
            out = out.to(self.device)
        if profile:
            print(f"[ggml-vae] total_decode={time.perf_counter() - total_t0:.3f}s")
        return out

    dit_handler.tiled_decode = types.MethodType(tiled_decode_ggml, dit_handler)


def _release_torch_modules_for_ggml_backends(
    dit_handler,
    release_text_encoder: bool,
    release_vae: bool,
) -> None:
    import torch

    released = []
    if release_text_encoder and getattr(dit_handler, "text_encoder", None) is not None:
        try:
            dit_handler.text_encoder = dit_handler.text_encoder.to("cpu")
            released.append("text_encoder")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to move torch text_encoder to CPU: {exc}")

    if release_vae and getattr(dit_handler, "vae", None) is not None:
        try:
            vae_dtype_cpu = None
            if hasattr(dit_handler, "_get_vae_dtype"):
                vae_dtype_cpu = dit_handler._get_vae_dtype("cpu")
            vae_module = dit_handler.vae.to("cpu")
            if vae_dtype_cpu is not None:
                vae_module = vae_module.to(vae_dtype_cpu)
            dit_handler.vae = vae_module
            released.append("vae")
        except Exception as exc:  # noqa: BLE001
            print(f"[warn] failed to move torch VAE to CPU: {exc}")

    if released:
        gc.collect()
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            try:
                torch.mps.empty_cache()
            except Exception:  # noqa: BLE001
                pass
        print(f"[ggml-capi] released torch modules to CPU: {', '.join(released)}")


def _first_diff(a: list[int], b: list[int]) -> int | None:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    return None if len(a) == len(b) else n


def compare_tokenizers(
    *,
    tokenizer_dir: Path,
    style_prompt: str,
    lyric_prompt: str,
    style_max_len: int,
    lyric_max_len: int,
    preview_tokens: int,
    report_path: Path | None = None,
) -> dict:
    from tokenizers import Tokenizer  # type: ignore
    from transformers import AutoTokenizer

    tok_json = tokenizer_dir / "tokenizer.json"
    if not tok_json.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tok_json}")

    hf_tok = AutoTokenizer.from_pretrained(str(tokenizer_dir), use_fast=True, local_files_only=True)
    ggml_tok = Tokenizer.from_file(str(tok_json))

    py_style = hf_tok(
        style_prompt,
        padding="longest",
        truncation=True,
        max_length=style_max_len,
        return_tensors="pt",
    ).input_ids[0].tolist()
    py_lyric = hf_tok(
        lyric_prompt,
        padding="longest",
        truncation=True,
        max_length=lyric_max_len,
        return_tensors="pt",
    ).input_ids[0].tolist()

    gg_style = ggml_tok.encode(style_prompt).ids
    gg_lyric = ggml_tok.encode(lyric_prompt).ids
    if style_max_len > 0:
        gg_style = gg_style[:style_max_len]
    if lyric_max_len > 0:
        gg_lyric = gg_lyric[:lyric_max_len]

    style_diff = _first_diff(py_style, gg_style)
    lyric_diff = _first_diff(py_lyric, gg_lyric)

    result = {
        "tokenizer_dir": str(tokenizer_dir),
        "style": {
            "python_len": len(py_style),
            "ggml_len": len(gg_style),
            "equal": py_style == gg_style,
            "first_diff_index": style_diff,
            "python_preview": py_style[:preview_tokens],
            "ggml_preview": gg_style[:preview_tokens],
        },
        "lyric": {
            "python_len": len(py_lyric),
            "ggml_len": len(gg_lyric),
            "equal": py_lyric == gg_lyric,
            "first_diff_index": lyric_diff,
            "python_preview": py_lyric[:preview_tokens],
            "ggml_preview": gg_lyric[:preview_tokens],
        },
    }

    print(f"[tokenizer-compare] dir={tokenizer_dir}")
    print(
        f"[tokenizer-compare] style equal={result['style']['equal']} "
        f"len_py={result['style']['python_len']} len_ggml={result['style']['ggml_len']} "
        f"first_diff={result['style']['first_diff_index']}"
    )
    print(
        f"[tokenizer-compare] lyric equal={result['lyric']['equal']} "
        f"len_py={result['lyric']['python_len']} len_ggml={result['lyric']['ggml_len']} "
        f"first_diff={result['lyric']['first_diff_index']}"
    )

    if report_path is not None:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[tokenizer-compare] report saved: {report_path}")

    return result


def parse_args() -> argparse.Namespace:
    root_default = Path(__file__).resolve().parents[1]
    ggml_vae_metal_decode_default = os.environ.get("ACE_GGML_VAE_METAL_DECODE", "auto").strip().lower()
    if ggml_vae_metal_decode_default not in ("auto", "on", "off"):
        ggml_vae_metal_decode_default = "auto"
    ggml_vae_weight_map_default = os.environ.get("ACE_GGML_VAE_METAL_WEIGHT_MAP", "auto").strip().lower()
    if ggml_vae_weight_map_default not in ("auto", "on", "off"):
        ggml_vae_weight_map_default = "auto"
    try:
        ggml_vae_min_free_default = int(os.environ.get("ACE_GGML_VAE_METAL_MIN_FREE_MB", "4096"))
    except ValueError:
        ggml_vae_min_free_default = 4096

    parser = argparse.ArgumentParser(
        description="Non-ggml text2music generation with real style and lyrics."
    )
    parser.add_argument("--project-root", default=str(root_default))
    parser.add_argument("--config-path", default="acestep-v15-turbo")
    parser.add_argument("--device", default="auto", choices=["auto", "cuda", "mps", "xpu", "cpu"])
    parser.add_argument("--use-flash-attention", action="store_true")
    parser.add_argument("--offload-to-cpu", action="store_true")
    parser.add_argument("--offload-dit-to-cpu", action="store_true")
    parser.add_argument("--download-source", default="auto", choices=["auto", "huggingface", "modelscope"])

    parser.add_argument(
        "--style-file",
        default="acestep_ggml/reports/real_case/style_neo_soul.txt",
        help="Style/caption text file path (relative to project root or absolute).",
    )
    parser.add_argument(
        "--lyric-file",
        default="acestep_ggml/reports/real_case/lyric_neo_soul.txt",
        help="Lyrics text file path (relative to project root or absolute).",
    )
    parser.add_argument("--caption", default="", help="Fallback caption text if style-file is empty.")
    parser.add_argument("--lyrics", default="", help="Fallback lyrics text if lyric-file is empty.")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--bpm", type=int, default=None)
    parser.add_argument("--keyscale", default="")
    parser.add_argument("--timesignature", default="")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--instruction", default=DEFAULT_DIT_INSTRUCTION)

    parser.add_argument("--inference-steps", type=int, default=8)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--audio-format", default="wav", choices=["wav", "flac", "mp3"])
    parser.add_argument("--out-dir", default="outputs/non_ggml_real_case")

    parser.add_argument("--thinking", action="store_true", help="Enable 5Hz LM thinking.")
    parser.add_argument("--use-cot-metas", action="store_true")
    parser.add_argument("--use-cot-caption", action="store_true")
    parser.add_argument("--use-cot-language", action="store_true")
    parser.add_argument("--disable-constrained-decoding", action="store_true")
    parser.add_argument("--lm-model-path", default="acestep-5Hz-lm-1.7B")
    parser.add_argument("--lm-backend", default="pt", choices=["pt", "vllm", "mlx"])
    parser.add_argument("--lm-temperature", type=float, default=0.85)
    parser.add_argument("--lm-cfg-scale", type=float, default=2.0)
    parser.add_argument("--lm-top-k", type=int, default=0)
    parser.add_argument("--lm-top-p", type=float, default=0.9)

    # Stepwise parity validation controls.
    parser.add_argument("--tokenizer-backend", default="python", choices=["python", "ggml-json"])
    parser.add_argument("--tokenizer-dir", default="", help="Tokenizer dir with tokenizer.json. Default: checkpoints/Qwen3-Embedding-0.6B")
    parser.add_argument("--style-max-len", type=int, default=256)
    parser.add_argument("--lyric-max-len", type=int, default=2048)
    parser.add_argument("--compare-tokenizers", action="store_true", help="Compare Python tokenizer vs ggml tokenizer.json before generation.")
    parser.add_argument("--tokenizer-compare-only", action="store_true", help="Only run tokenizer comparison and exit.")
    parser.add_argument("--tokenizer-preview", type=int, default=64, help="How many token IDs to include in preview/report.")
    parser.add_argument("--tokenizer-report", default="", help="Optional tokenizer compare report JSON path.")

    parser.add_argument("--text-encoder-backend", default="python", choices=["python", "ggml-capi"])
    parser.add_argument("--dit-backend", default="python", choices=["python", "ggml-capi"])
    parser.add_argument("--vae-backend", default="python", choices=["python", "ggml-capi"])
    parser.add_argument("--ggml-lib", default="acestep_ggml/build/libacestep_ggml.dylib")
    parser.add_argument("--ggml-text-encoder-dir", default="", help="Directory for ggml C API text encoder load.")
    parser.add_argument("--ggml-dit-dir", default="", help="Directory for ggml C API DiT load.")
    parser.add_argument("--ggml-vae-dir", default="", help="Directory for ggml C API VAE load (root containing vae/).")
    parser.add_argument("--ggml-threads", type=int, default=4)
    parser.add_argument(
        "--ggml-use-metal",
        dest="ggml_use_metal",
        action="store_true",
        default=None,
        help="Request ggml Metal backend in C API (requires Metal-enabled build).",
    )
    parser.add_argument(
        "--ggml-no-metal",
        dest="ggml_use_metal",
        action="store_false",
        help="Force-disable ggml Metal backend in C API.",
    )
    parser.add_argument("--ggml-compute-buffer-mb", type=int, default=12288)
    parser.add_argument("--ggml-vae-chunk-size", type=int, default=32)
    parser.add_argument("--ggml-vae-overlap", type=int, default=8)
    parser.add_argument("--ggml-vae-metal-decode", default=ggml_vae_metal_decode_default, choices=["auto", "on", "off"], help="Metal VAE decode mode for ggml backend (ACE_GGML_VAE_METAL_DECODE).")
    parser.add_argument("--ggml-vae-metal-weight-map", default=ggml_vae_weight_map_default, choices=["auto", "on", "off"], help="Metal host_ptr weight mapping mode for VAE weights (ACE_GGML_VAE_METAL_WEIGHT_MAP).")
    parser.add_argument("--ggml-vae-metal-min-free-mb", type=int, default=ggml_vae_min_free_default, help="Minimum free memory headroom for ggml Metal VAE auto mode.")
    parser.add_argument(
        "--ggml-vae-transpose-conv-f32",
        action="store_true",
        help="Force VAE transposed-conv weights to fp32 in ggml backend (ACE_GGML_VAE_TRANSPOSE_CONV_F32=1).",
    )
    parser.add_argument(
        "--ggml-release-torch-unused",
        dest="ggml_release_torch_unused",
        action="store_true",
        default=True,
        help="Move replaced torch text_encoder/VAE modules to CPU when ggml backends are enabled.",
    )
    parser.add_argument(
        "--no-ggml-release-torch-unused",
        dest="ggml_release_torch_unused",
        action="store_false",
        help="Keep replaced torch text_encoder/VAE modules on their current device.",
    )
    parser.add_argument("--ggml-vae-profile", action="store_true", help="Enable C++ vae_decode profiling logs.")
    parser.add_argument("--ggml-vae-chunk-profile", action="store_true", help="Print Python-side per-chunk VAE decode timing.")
    return parser.parse_args()


def read_optional_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def ensure_checkpoints_link(project_root: Path) -> None:
    checkpoints_dir = project_root / "checkpoints"
    legacy_dir = project_root / "Ace-Step1.5"

    if checkpoints_dir.exists():
        return

    required = [
        "acestep-v15-turbo",
        "acestep-5Hz-lm-1.7B",
        "Qwen3-Embedding-0.6B",
        "vae",
    ]
    if not legacy_dir.exists():
        return
    if not all((legacy_dir / item).exists() for item in required):
        return

    try:
        checkpoints_dir.symlink_to(legacy_dir, target_is_directory=True)
        print(f"[setup] created symlink: {checkpoints_dir} -> {legacy_dir}")
    except Exception as exc:  # noqa: BLE001
        print(
            "[warn] failed to create checkpoints symlink automatically. "
            f"Please create it manually if needed. reason={exc}"
        )


def main() -> int:
    args = parse_args()
    if args.ggml_use_metal is None:
        # Default to Metal for macOS runs unless explicitly disabled.
        args.ggml_use_metal = (sys.platform == "darwin")
    project_root = Path(args.project_root).resolve()
    if not project_root.exists():
        print(f"[error] project root not found: {project_root}")
        return 1

    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    ensure_checkpoints_link(project_root)

    style_path = Path(args.style_file)
    if not style_path.is_absolute():
        style_path = project_root / style_path

    lyric_path = Path(args.lyric_file)
    if not lyric_path.is_absolute():
        lyric_path = project_root / lyric_path

    caption = read_optional_text(style_path) or args.caption.strip()
    lyrics = read_optional_text(lyric_path) or args.lyrics.strip()
    if not caption or not lyrics:
        print("[error] caption or lyrics is empty. Provide valid style/lyric files or --caption/--lyrics.")
        return 1

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = project_root / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    tokenizer_dir = _resolve_tokenizer_dir(project_root, args.tokenizer_dir)
    meta_str = _build_meta_string(
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        duration=args.duration,
    )
    style_prompt = _build_style_prompt(
        caption=caption,
        instruction=args.instruction,
        meta_str=meta_str,
    )
    lyric_prompt = _build_lyric_prompt(
        lyrics=lyrics,
        language=args.language,
    )

    if args.compare_tokenizers or args.tokenizer_compare_only:
        report_path = None
        if args.tokenizer_report:
            report_path = Path(args.tokenizer_report)
            if not report_path.is_absolute():
                report_path = project_root / report_path
        elif args.tokenizer_compare_only:
            report_path = out_dir / "tokenizer_compare.json"

        compare_tokenizers(
            tokenizer_dir=tokenizer_dir,
            style_prompt=style_prompt,
            lyric_prompt=lyric_prompt,
            style_max_len=args.style_max_len,
            lyric_max_len=args.lyric_max_len,
            preview_tokens=args.tokenizer_preview,
            report_path=report_path,
        )
        if args.tokenizer_compare_only:
            return 0

    # Late imports: keep startup fast for tokenizer-only checks.
    from acestep.handler import AceStepHandler
    from acestep.inference import GenerationConfig, GenerationParams, generate_music
    from acestep.llm_inference import LLMHandler
    from acestep.model_downloader import ensure_lm_model

    prefer_source = None if args.download_source == "auto" else args.download_source

    dit_handler = AceStepHandler()
    dit_status, dit_ok = dit_handler.initialize_service(
        project_root=str(project_root),
        config_path=args.config_path,
        device=args.device,
        use_flash_attention=args.use_flash_attention,
        offload_to_cpu=args.offload_to_cpu,
        offload_dit_to_cpu=args.offload_dit_to_cpu,
        prefer_source=prefer_source,
    )
    print(dit_status)
    if not dit_ok:
        return 1

    if args.tokenizer_backend == "ggml-json":
        try:
            dit_handler.text_tokenizer = GGMLJSONTokenizerAdapter(dit_handler.text_tokenizer, tokenizer_dir)
            print(f"[tokenizer-backend] using ggml-json tokenizer: {tokenizer_dir}")
        except Exception as exc:  # noqa: BLE001
            print(f"[error] failed to apply ggml-json tokenizer backend: {exc}")
            return 1
    else:
        print("[tokenizer-backend] using python(huggingface) tokenizer")

    ggml_bridge = None
    need_ggml_bridge = (
        args.text_encoder_backend == "ggml-capi"
        or args.dit_backend == "ggml-capi"
        or args.vae_backend == "ggml-capi"
    )
    if need_ggml_bridge:
        if args.ggml_vae_transpose_conv_f32:
            os.environ["ACE_GGML_VAE_TRANSPOSE_CONV_F32"] = "1"

        if args.ggml_use_metal:
            # Forward safe-default metal gating knobs to C++ backend.
            os.environ["ACE_GGML_VAE_METAL_DECODE"] = args.ggml_vae_metal_decode
            os.environ["ACE_GGML_VAE_METAL_WEIGHT_MAP"] = args.ggml_vae_metal_weight_map
            os.environ["ACE_GGML_VAE_METAL_MIN_FREE_MB"] = str(max(0, int(args.ggml_vae_metal_min_free_mb)))

        if args.ggml_vae_profile:
            # C++ side profiling switch read from env.
            os.environ["ACE_GGML_VAE_PROFILE"] = "1"

        if args.ggml_lib:
            ggml_lib = _resolve_path_with_default(project_root, args.ggml_lib, "acestep_ggml/build/libacestep_ggml.dylib")
        else:
            if args.ggml_use_metal:
                ggml_lib = _resolve_path_with_default(project_root, "", "acestep_ggml/build_metal/libacestep_ggml.dylib")
                if not ggml_lib.exists():
                    ggml_lib = _resolve_path_with_default(project_root, "", "acestep_ggml/build/libacestep_ggml.dylib")
            else:
                ggml_lib = _resolve_path_with_default(project_root, "", "acestep_ggml/build/libacestep_ggml.dylib")
        if not ggml_lib.exists():
            print(f"[error] ggml lib not found: {ggml_lib}")
            return 1

        ggml_bridge = GGMLCAPIBridge(
            lib_path=ggml_lib,
            n_threads=args.ggml_threads,
            compute_buffer_mb=args.ggml_compute_buffer_mb,
            use_metal=args.ggml_use_metal,
        )
        print(f"[ggml-capi] bridge created: lib={ggml_lib} use_metal={args.ggml_use_metal}")

        if args.text_encoder_backend == "ggml-capi":
            ggml_text_encoder_dir = _resolve_path_with_default(
                project_root,
                args.ggml_text_encoder_dir,
                "checkpoints/Qwen3-Embedding-0.6B",
            )
            if not ggml_text_encoder_dir.exists():
                ggml_text_encoder_dir = _resolve_path_with_default(
                    project_root,
                    args.ggml_text_encoder_dir,
                    "Ace-Step1.5/Qwen3-Embedding-0.6B",
                )
            ggml_bridge.load_text_encoder(ggml_text_encoder_dir)
            _install_ggml_text_encoder_backend(dit_handler, ggml_bridge)
            print(f"[text-encoder-backend] using ggml-capi: {ggml_text_encoder_dir}")
        else:
            print("[text-encoder-backend] using python(torch)")

        if args.dit_backend == "ggml-capi":
            ggml_dit_dir = _resolve_path_with_default(
                project_root,
                args.ggml_dit_dir,
                f"checkpoints/{args.config_path}",
            )
            if not ggml_dit_dir.exists():
                ggml_dit_dir = _resolve_path_with_default(
                    project_root,
                    args.ggml_dit_dir,
                    f"Ace-Step1.5/{args.config_path}",
                )
            ggml_bridge.load_dit(ggml_dit_dir)
            _install_ggml_dit_backend(dit_handler, ggml_bridge)
            print(f"[dit-backend] using ggml-capi: {ggml_dit_dir}")
        else:
            print("[dit-backend] using python(torch)")

        if args.vae_backend == "ggml-capi":
            ggml_vae_dir = _resolve_path_with_default(project_root, args.ggml_vae_dir, "checkpoints")
            if not ggml_vae_dir.exists():
                ggml_vae_dir = _resolve_path_with_default(project_root, args.ggml_vae_dir, "Ace-Step1.5")
            ggml_bridge.load_vae(ggml_vae_dir)
            _install_ggml_vae_backend(
                dit_handler,
                ggml_bridge,
                chunk_size_default=args.ggml_vae_chunk_size,
                overlap_default=args.ggml_vae_overlap,
                profile_chunks=args.ggml_vae_chunk_profile,
            )
            print(f"[vae-backend] using ggml-capi: {ggml_vae_dir}")
        else:
            print("[vae-backend] using python(torch)")

        if args.ggml_release_torch_unused:
            _release_torch_modules_for_ggml_backends(
                dit_handler,
                release_text_encoder=(args.text_encoder_backend == "ggml-capi"),
                release_vae=(args.vae_backend == "ggml-capi"),
            )
    else:
        print("[text-encoder-backend] using python(torch)")
        print("[dit-backend] using python(torch)")
        print("[vae-backend] using python(torch)")

    need_lm = bool(args.thinking or args.use_cot_metas or args.use_cot_caption or args.use_cot_language)
    llm_handler = None
    if need_lm:
        checkpoints_dir = project_root / "checkpoints"
        ok, msg = ensure_lm_model(
            model_name=args.lm_model_path,
            checkpoints_dir=checkpoints_dir,
            prefer_source=prefer_source,
        )
        print(msg)
        if not ok:
            return 1

        llm_handler = LLMHandler()
        lm_status, lm_ok = llm_handler.initialize(
            checkpoint_dir=str(checkpoints_dir),
            lm_model_path=args.lm_model_path,
            backend=args.lm_backend,
            device=args.device,
            offload_to_cpu=args.offload_to_cpu,
        )
        print(lm_status)
        if not lm_ok:
            return 1

    params = GenerationParams(
        task_type="text2music",
        caption=caption,
        lyrics=lyrics,
        vocal_language=args.language,
        bpm=args.bpm,
        keyscale=args.keyscale,
        timesignature=args.timesignature,
        duration=args.duration,
        inference_steps=args.inference_steps,
        seed=args.seed,
        shift=args.shift,
        thinking=args.thinking,
        lm_temperature=args.lm_temperature,
        lm_cfg_scale=args.lm_cfg_scale,
        lm_top_k=args.lm_top_k,
        lm_top_p=args.lm_top_p,
        use_cot_metas=args.use_cot_metas,
        use_cot_caption=args.use_cot_caption,
        use_cot_language=args.use_cot_language,
        use_constrained_decoding=not args.disable_constrained_decoding,
    )

    use_random_seed = args.seed < 0
    seeds = None if use_random_seed else [args.seed]
    config = GenerationConfig(
        batch_size=args.batch_size,
        use_random_seed=use_random_seed,
        seeds=seeds,
        audio_format=args.audio_format,
    )

    print("[run] starting non-ggml generation...")
    result = generate_music(
        dit_handler=dit_handler,
        llm_handler=llm_handler,
        params=params,
        config=config,
        save_dir=str(out_dir),
    )

    if not result.success:
        print(f"[error] generation failed: {result.error}")
        print(result.status_message)
        return 1

    print(f"[ok] generated {len(result.audios)} audio file(s)")
    for idx, audio in enumerate(result.audios, start=1):
        print(
            f"  {idx}. path={audio.get('path', '')} "
            f"seed={audio.get('params', {}).get('seed')} "
            f"silent={audio.get('silent')}"
        )
    print(f"[done] output dir: {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
