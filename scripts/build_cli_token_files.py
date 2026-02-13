#!/usr/bin/env python3
"""Prepare ace_ggml_cli token files from text prompts.

This script mirrors the prompt-building rules used in `run_non_ggml_real_case.py`:
- style prompt = instruction + caption + meta
- lyric prompt = language + lyrics

It tokenizes both prompts with `tokenizer.json` (ggml-json style), applies
max-length truncation, writes token-id files, and can print the sequence length
derived from duration (`audio_seconds * sample_rate / hop_length`).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tokenizers import Tokenizer  # type: ignore

DEFAULT_DIT_INSTRUCTION = "Fill the audio semantic mask based on the given conditions:"
SFT_GEN_PROMPT = """# Instruction
{}

# Caption
{}

# Metas
{}<|endoftext|>
"""


def read_optional_text(path: Path) -> str:
    """Read UTF-8 text file and trim; return empty string if missing."""
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8").strip()


def format_instruction(instruction: str) -> str:
    """Ensure the instruction line ends with ':' to match Python path."""
    return instruction if instruction.endswith(":") else instruction + ":"


def build_meta_string(
    bpm: int | None,
    keyscale: str,
    timesignature: str,
    duration: float | None,
) -> str:
    """Build metadata section exactly like run_non_ggml_real_case.py."""
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


def build_style_prompt(caption: str, instruction: str, meta_str: str) -> str:
    """Build style prompt text."""
    return SFT_GEN_PROMPT.format(format_instruction(instruction), caption, meta_str)


def build_lyric_prompt(lyrics: str, language: str) -> str:
    """Build lyric prompt text."""
    return f"# Languages\n{language}\n\n# Lyric\n{lyrics}<|endoftext|>"


def resolve_vae_config_path(vae_dir: Path) -> Path:
    """Resolve VAE config path from root dir or direct vae dir."""
    candidates = [vae_dir / "vae" / "config.json", vae_dir / "config.json"]
    for path in candidates:
        if path.exists():
            try:
                cfg = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            ratios = cfg.get("downsampling_ratios")
            if isinstance(ratios, list) and len(ratios) > 0:
                return path
    raise FileNotFoundError(f"VAE config.json not found under: {vae_dir}")


def load_hop_length_from_vae_config(vae_dir: Path) -> int:
    """Load hop_length from VAE config (product of downsampling_ratios)."""
    cfg_path = resolve_vae_config_path(vae_dir)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    ratios = cfg.get("downsampling_ratios")
    if not isinstance(ratios, list) or not ratios:
        raise ValueError(f"invalid downsampling_ratios in {cfg_path}")
    hop = 1
    for value in ratios:
        hop *= int(value)
    return int(hop)


def write_token_file(path: Path, token_ids: list[int]) -> None:
    """Write token ids as comma-separated integers."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(",".join(str(x) for x in token_ids), encoding="utf-8")


def parse_optional_bpm(raw: str) -> int | None:
    """Parse optional bpm from raw string."""
    text = raw.strip()
    if not text:
        return None
    return int(text)


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""
    parser = argparse.ArgumentParser(description="Prepare token files for ace_ggml_cli.")
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--tokenizer-dir", required=True)
    parser.add_argument("--style-file", required=True)
    parser.add_argument("--lyric-file", required=True)
    parser.add_argument("--caption", default="")
    parser.add_argument("--lyrics", default="")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--instruction", default=DEFAULT_DIT_INSTRUCTION)
    parser.add_argument("--bpm", default="")
    parser.add_argument("--keyscale", default="")
    parser.add_argument("--timesignature", default="")
    parser.add_argument("--duration", type=float, default=30.0)
    parser.add_argument("--style-max-len", type=int, default=256)
    parser.add_argument("--lyric-max-len", type=int, default=2048)
    parser.add_argument("--out-style-tokens-file", required=True)
    parser.add_argument("--out-lyric-tokens-file", required=True)
    parser.add_argument("--vae-dir", required=True)
    parser.add_argument("--sample-rate", type=int, default=48000)
    parser.add_argument("--print-seq-len", action="store_true")
    return parser.parse_args()


def main() -> int:
    """Build prompts, tokenize, write files, and optionally print seq_len."""
    args = parse_args()

    project_root = Path(args.project_root).resolve()
    tokenizer_dir = Path(args.tokenizer_dir)
    if not tokenizer_dir.is_absolute():
        tokenizer_dir = (project_root / tokenizer_dir).resolve()
    style_file = Path(args.style_file)
    if not style_file.is_absolute():
        style_file = (project_root / style_file).resolve()
    lyric_file = Path(args.lyric_file)
    if not lyric_file.is_absolute():
        lyric_file = (project_root / lyric_file).resolve()
    vae_dir = Path(args.vae_dir)
    if not vae_dir.is_absolute():
        vae_dir = (project_root / vae_dir).resolve()

    caption = read_optional_text(style_file) or args.caption.strip()
    lyrics = read_optional_text(lyric_file) or args.lyrics.strip()
    if not caption:
        raise ValueError("caption is empty (style file and --caption are both empty)")
    if not lyrics:
        raise ValueError("lyrics is empty (lyric file and --lyrics are both empty)")

    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.exists():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    tokenizer = Tokenizer.from_file(str(tokenizer_path))

    bpm = parse_optional_bpm(args.bpm)
    meta_str = build_meta_string(bpm, args.keyscale, args.timesignature, args.duration)
    style_prompt = build_style_prompt(caption, args.instruction, meta_str)
    lyric_prompt = build_lyric_prompt(lyrics, args.language)

    style_ids = tokenizer.encode(style_prompt).ids
    lyric_ids = tokenizer.encode(lyric_prompt).ids
    if args.style_max_len > 0:
        style_ids = style_ids[: args.style_max_len]
    if args.lyric_max_len > 0:
        lyric_ids = lyric_ids[: args.lyric_max_len]

    out_style = Path(args.out_style_tokens_file)
    out_lyric = Path(args.out_lyric_tokens_file)
    write_token_file(out_style, style_ids)
    write_token_file(out_lyric, lyric_ids)

    hop_length = load_hop_length_from_vae_config(vae_dir)
    if args.sample_rate <= 0:
        raise ValueError("sample-rate must be > 0")
    seq_len = max(1, int(round(float(args.duration) * float(args.sample_rate) / float(hop_length))))

    print(f"[prepare-cli] style_tokens={len(style_ids)} lyric_tokens={len(lyric_ids)} hop={hop_length}", flush=True)
    print(f"[prepare-cli] wrote style={out_style} lyric={out_lyric}", flush=True)
    if args.print_seq_len:
        print(seq_len)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
