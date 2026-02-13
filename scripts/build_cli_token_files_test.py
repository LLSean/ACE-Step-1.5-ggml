#!/usr/bin/env python3
"""Unit tests for build_cli_token_files.py helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_cli_token_files import (
    build_lyric_prompt,
    build_meta_string,
    build_style_prompt,
    load_hop_length_from_vae_config,
)


class BuildCliTokenFilesTest(unittest.TestCase):
    """Validate prompt formatting and VAE hop-length parsing."""

    def test_build_meta_string_default_values(self) -> None:
        """Meta string should match Python script semantics for missing fields."""
        text = build_meta_string(None, "", "", 8.9)
        self.assertIn("- bpm: N/A", text)
        self.assertIn("- timesignature: N/A", text)
        self.assertIn("- keyscale: N/A", text)
        self.assertIn("- duration: 8 seconds", text)

    def test_prompt_builders(self) -> None:
        """Style and lyric prompts should include required headers/tails."""
        style = build_style_prompt("caption", "Instruction", "- duration: 8 seconds\n")
        lyric = build_lyric_prompt("la la", "zh")
        self.assertIn("# Instruction", style)
        self.assertIn("Instruction:", style)
        self.assertIn("# Caption", style)
        self.assertIn("# Languages", lyric)
        self.assertTrue(lyric.endswith("<|endoftext|>"))

    def test_load_hop_length_from_vae_config(self) -> None:
        """Hop length should be product of downsampling ratios."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            vae_dir = root / "vae"
            vae_dir.mkdir(parents=True, exist_ok=True)
            cfg = {"downsampling_ratios": [2, 4, 8]}
            (vae_dir / "config.json").write_text(json.dumps(cfg), encoding="utf-8")
            self.assertEqual(load_hop_length_from_vae_config(root), 64)


if __name__ == "__main__":
    unittest.main()
