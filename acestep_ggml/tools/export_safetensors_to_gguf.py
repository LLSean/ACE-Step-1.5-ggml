#!/usr/bin/env python3
from __future__ import annotations

import argparse
import ctypes
import json
import math
import pathlib
import struct
import tempfile
from dataclasses import dataclass
from typing import Dict, Iterable, List, Tuple

import numpy as np


def _load_gguf_modules() -> Tuple[object, object]:
    import sys

    llama_root = pathlib.Path("/Users/fmh/project/llama.cpp/gguf-py")
    if str(llama_root) not in sys.path:
        sys.path.insert(0, str(llama_root))
    import gguf  # type: ignore
    from gguf.quants import quantize, quant_shape_to_byte_shape  # type: ignore

    return gguf, (quantize, quant_shape_to_byte_shape)


@dataclass
class TensorMeta:
    name: str
    dtype: str
    shape: Tuple[int, ...]
    data_start: int
    data_end: int

    @property
    def nbytes(self) -> int:
        return self.data_end - self.data_start


def parse_safetensors_header(path: pathlib.Path) -> Tuple[List[TensorMeta], int]:
    with path.open("rb") as f:
        header_len = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(header_len))
    data_offset = 8 + int(header_len)
    tensors: List[TensorMeta] = []
    for name, info in header.items():
        if name == "__metadata__":
            continue
        dtype = str(info["dtype"])
        shape = tuple(int(x) for x in info["shape"])
        data_offsets = info["data_offsets"]
        tensors.append(
            TensorMeta(
                name=name,
                dtype=dtype,
                shape=shape,
                data_start=data_offset + int(data_offsets[0]),
                data_end=data_offset + int(data_offsets[1]),
            )
        )
    tensors.sort(key=lambda x: x.name)
    return tensors, data_offset


def bf16_to_f32(x: np.ndarray) -> np.ndarray:
    y = x.astype(np.uint32) << 16
    return y.view(np.float32)


def open_source_array(mm: np.memmap, meta: TensorMeta) -> np.ndarray:
    if meta.dtype == "F32":
        dt = np.dtype("<f4")
    elif meta.dtype == "F16":
        dt = np.dtype("<f2")
    elif meta.dtype == "BF16":
        dt = np.dtype("<u2")
    elif meta.dtype == "I32":
        dt = np.dtype("<i4")
    elif meta.dtype == "I64":
        dt = np.dtype("<i8")
    else:
        raise ValueError(f"unsupported dtype: {meta.dtype} ({meta.name})")
    return np.ndarray(shape=meta.shape, dtype=dt, buffer=mm, offset=meta.data_start)


def allocate_memmap(path_prefix: str, dtype: np.dtype, shape: Tuple[int, ...]) -> Tuple[str, np.memmap]:
    tf = tempfile.NamedTemporaryFile(prefix=path_prefix, suffix=".bin", delete=False)
    tf.close()
    arr = np.memmap(tf.name, mode="w+", dtype=dtype, shape=shape)
    return tf.name, arr


def iter_row_chunks(total_rows: int, chunk_rows: int) -> Iterable[Tuple[int, int]]:
    i = 0
    while i < total_rows:
        j = min(i + chunk_rows, total_rows)
        yield i, j
        i = j


def parse_quant_arg(s: str, gguf_mod: object):
    q = s.strip().upper()
    qmap: Dict[str, str] = {
        "F16": "F16",
        "Q8": "Q8_0",
        "Q8_0": "Q8_0",
        "Q6": "Q6_K",
        "Q6_K": "Q6_K",
        "Q4": "Q4_K",
        "Q4_K": "Q4_K",
    }
    if q not in qmap:
        raise ValueError(f"unsupported quant: {s}")
    name = qmap[q]
    return name, getattr(gguf_mod.GGMLQuantizationType, name)


class GGMLChunkQuantizer:
    def __init__(self, lib_path: str):
        self.lib = ctypes.CDLL(lib_path)
        self.lib.ggml_quantize_chunk.argtypes = [
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float),
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_float),
        ]
        self.lib.ggml_quantize_chunk.restype = ctypes.c_size_t

    def quantize(self, f32_2d: np.ndarray, qtype_value: int, q_byte_row: int) -> np.ndarray:
        if f32_2d.ndim != 2 or f32_2d.dtype != np.float32:
            raise ValueError("f32_2d must be float32 [rows, row_size]")
        rows, row_size = f32_2d.shape
        out = np.empty((rows, q_byte_row), dtype=np.uint8)
        written = self.lib.ggml_quantize_chunk(
            qtype_value,
            f32_2d.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            out.ctypes.data_as(ctypes.c_void_p),
            ctypes.c_int64(0),
            ctypes.c_int64(rows),
            ctypes.c_int64(row_size),
            None,
        )
        expected = rows * q_byte_row
        if int(written) != int(expected):
            raise RuntimeError(f"ggml_quantize_chunk wrote {written}, expected {expected}")
        return out


def main() -> int:
    gguf, quant_mod = _load_gguf_modules()
    quantize_fn, quant_shape_to_byte_shape = quant_mod

    ap = argparse.ArgumentParser(description="Export safetensors to GGUF with optional quantization.")
    ap.add_argument("--input", required=True, help="input .safetensors")
    ap.add_argument("--output", required=True, help="output .gguf")
    ap.add_argument("--arch", default="acestep", help="GGUF architecture string")
    ap.add_argument("--name", default="", help="optional model name")
    ap.add_argument("--quant", default="Q8", help="F16|Q8|Q6|Q4")
    ap.add_argument("--chunk-rows", type=int, default=64, help="row-chunk size for quantization")
    ap.add_argument("--max-tensors", type=int, default=0, help="optional tensor limit for debugging")
    ap.add_argument("--log-every", type=int, default=20, help="print every N tensors (<=1 means all)")
    ap.add_argument("--ggml-lib", default="/tmp/ace_ggml_build/third_party/ggml/src/libggml-base.0.9.5.dylib")
    args = ap.parse_args()

    inp = pathlib.Path(args.input)
    out = pathlib.Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    qname, qtype = parse_quant_arg(args.quant, gguf)
    is_quantized = qname != "F16"
    quantizer = GGMLChunkQuantizer(args.ggml_lib) if is_quantized else None

    tensors, _ = parse_safetensors_header(inp)
    if args.max_tensors > 0:
        tensors = tensors[: args.max_tensors]

    print(f"[info] input={inp}")
    print(f"[info] output={out}")
    print(f"[info] quant={qname} tensors={len(tensors)}")

    mm = np.memmap(inp, mode="r", dtype=np.uint8)
    writer = gguf.GGUFWriter(str(out), arch=args.arch, use_temp_file=True)
    writer.add_type("model")
    if args.name:
        writer.add_name(args.name)
    writer.add_quantization_version(gguf.GGML_QUANT_VERSION)
    writer.add_string("general.quantized_by", "ace_ggml export_safetensors_to_gguf.py")

    temp_files: List[str] = []
    n_q = 0
    n_f16 = 0
    n_other = 0

    try:
        for i, meta in enumerate(tensors, start=1):
            src = open_source_array(mm, meta)
            if args.log_every <= 1 or i == 1 or i == len(tensors) or (i % args.log_every == 0):
                print(f"[{i:04d}/{len(tensors):04d}] {meta.name} shape={meta.shape} dtype={meta.dtype}")

            # keep integer tensors as-is
            if meta.dtype in ("I32", "I64"):
                src_copy = np.array(src, copy=True)
                writer.add_tensor(meta.name, src_copy)
                n_other += 1
                continue

            # float tensors
            can_try_quant = (
                is_quantized
                and len(meta.shape) >= 2
                and meta.shape[-1] > 0
                and (meta.shape[-1] % gguf.GGML_QUANT_SIZES[qtype][0] == 0)
            )

            if can_try_quant:
                row_size = int(meta.shape[-1])
                rows = int(np.prod(meta.shape[:-1]))
                byte_shape_2d = quant_shape_to_byte_shape((rows, row_size), qtype)
                q_byte_row = int(byte_shape_2d[-1])
                q_shape = tuple(meta.shape[:-1]) + (q_byte_row,)
                tmp_path, out_q = allocate_memmap("ace_gguf_q_", np.uint8, q_shape)
                temp_files.append(tmp_path)
                out_q_2d = out_q.reshape(rows, q_byte_row)

                src_2d = src.reshape(rows, row_size)
                for b, e in iter_row_chunks(rows, max(1, args.chunk_rows)):
                    block = src_2d[b:e]
                    if meta.dtype == "F32":
                        f32 = block.astype(np.float32, copy=False)
                    elif meta.dtype == "F16":
                        f32 = block.astype(np.float32)
                    elif meta.dtype == "BF16":
                        f32 = bf16_to_f32(block)
                    else:
                        raise ValueError(f"unexpected dtype: {meta.dtype}")
                    try:
                        q_block = quantize_fn(f32, qtype)
                    except NotImplementedError:
                        if quantizer is None:
                            raise
                        q_block = quantizer.quantize(f32, int(qtype), q_byte_row)
                    out_q_2d[b:e] = q_block

                writer.add_tensor(meta.name, out_q, raw_dtype=qtype)
                n_q += 1
                continue

            # fallback to F16 tensor
            tmp_path, out_f16 = allocate_memmap("ace_gguf_f16_", np.float16, meta.shape)
            temp_files.append(tmp_path)

            if meta.dtype == "F16":
                out_f16[...] = src
            elif meta.dtype == "F32":
                out_f16[...] = src.astype(np.float16)
            elif meta.dtype == "BF16":
                out_f16[...] = bf16_to_f32(src).astype(np.float16)
            else:
                raise ValueError(f"unexpected dtype: {meta.dtype}")
            writer.add_tensor(meta.name, out_f16)
            n_f16 += 1

        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=False)
        writer.close()
    finally:
        for p in temp_files:
            try:
                pathlib.Path(p).unlink(missing_ok=True)
            except Exception:
                pass

    print(f"[done] output={out}")
    print(f"[stats] quantized={n_q} f16_fallback={n_f16} other={n_other}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
