#!/usr/bin/env python3
from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import pathlib
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any


@dataclass
class Stage:
    name: str
    cmd: List[str]
    env: Dict[str, str] = field(default_factory=dict)
    timeout_s: int = 180


def _sanitize_filename(name: str) -> str:
    out = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            out.append(ch)
        else:
            out.append("_")
    return "".join(out)


def run_stage(stage: Stage, cwd: str, log_dir: pathlib.Path) -> Dict[str, Any]:
    env = os.environ.copy()
    env.update(stage.env)

    start = time.time()
    timed_out = False
    rc = 0
    output = ""
    try:
        proc = subprocess.run(
            stage.cmd,
            cwd=cwd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=stage.timeout_s,
            check=False,
        )
        rc = proc.returncode
        output = proc.stdout or ""
    except subprocess.TimeoutExpired as e:
        timed_out = True
        rc = 124
        output = (e.stdout or "") + "\n[timeout]\n"

    elapsed = time.time() - start
    ok = (rc == 0) and (not timed_out)

    log_path = log_dir / f"{_sanitize_filename(stage.name)}.log"
    with log_path.open("w", encoding="utf-8") as f:
        f.write(f"stage: {stage.name}\n")
        f.write(f"cwd: {cwd}\n")
        f.write(f"timeout_s: {stage.timeout_s}\n")
        f.write(f"return_code: {rc}\n")
        f.write(f"elapsed_s: {elapsed:.3f}\n")
        if stage.env:
            f.write("env_overrides:\n")
            for k, v in stage.env.items():
                f.write(f"  {k}={v}\n")
        f.write("command:\n")
        f.write("  " + " ".join(shlex.quote(x) for x in stage.cmd) + "\n\n")
        f.write(output)

    status = "PASS" if ok else ("TIMEOUT" if timed_out else "FAIL")
    print(f"[{status}] {stage.name} ({elapsed:.1f}s) -> {log_path}")
    return {
        "name": stage.name,
        "status": status,
        "ok": ok,
        "timed_out": timed_out,
        "return_code": rc,
        "elapsed_s": round(elapsed, 3),
        "timeout_s": stage.timeout_s,
        "log_path": str(log_path),
        "command": stage.cmd,
        "env_overrides": stage.env,
    }


def build_stages(args: argparse.Namespace) -> List[Stage]:
    root = pathlib.Path(args.model_root)
    qwen06 = root / "Qwen3-Embedding-0.6B"
    lm17 = root / "acestep-5Hz-lm-1.7B"
    dit = root / "acestep-v15-turbo"
    vae = root
    bin_path = args.bin
    threads = str(args.threads)

    stages: List[Stage] = [
        Stage(
            name="text_encoder_smoke",
            cmd=[
                bin_path,
                "--text-encoder",
                str(qwen06),
                "--tokens",
                args.tokens,
                "--layers",
                "4",
                "--final-norm",
                "--threads",
                threads,
            ],
            timeout_s=args.timeout_s,
        ),
        Stage(
            name="dit_smoke",
            cmd=[
                bin_path,
                "--dit",
                str(dit),
                "--seq-len",
                "2",
                "--enc-len",
                "2",
                "--timestep",
                "0.5",
                "--timestep-r",
                "0.0",
                "--threads",
                threads,
            ],
            timeout_s=args.timeout_s,
        ),
        Stage(
            name="vae_decode_smoke",
            cmd=[
                bin_path,
                "--vae",
                str(vae),
                "--latent-len",
                "2",
                "--seed",
                str(args.seed),
                "--threads",
                threads,
            ],
            timeout_s=args.timeout_s,
        ),
        Stage(
            name="pipeline_light_seq1",
            cmd=[
                bin_path,
                "--pipeline",
                "--text-encoder",
                str(qwen06),
                "--dit",
                str(dit),
                "--vae",
                str(vae),
                "--tokens",
                args.tokens,
                "--seq-len",
                "1",
                "--shift",
                "3",
                "--seed",
                str(args.seed),
                "--threads",
                threads,
            ],
            env={
                "ACE_GGML_ALLOW_TEXT_DIM_MISMATCH": "1",
                "ACE_GGML_TEXT_MAX_LAYERS": "0",
                "ACE_GGML_DIT_MAX_LAYERS": "2",
            },
            timeout_s=args.timeout_s,
        ),
        Stage(
            name="pipeline_mid_seq1",
            cmd=[
                bin_path,
                "--pipeline",
                "--text-encoder",
                str(qwen06),
                "--dit",
                str(dit),
                "--vae",
                str(vae),
                "--tokens",
                args.tokens,
                "--seq-len",
                "1",
                "--shift",
                "3",
                "--seed",
                str(args.seed),
                "--threads",
                threads,
            ],
            env={
                "ACE_GGML_ALLOW_TEXT_DIM_MISMATCH": "1",
                "ACE_GGML_TEXT_MAX_LAYERS": "4",
            },
            timeout_s=args.timeout_s,
        ),
        Stage(
            name="pipeline_mid_seq2",
            cmd=[
                bin_path,
                "--pipeline",
                "--text-encoder",
                str(qwen06),
                "--dit",
                str(dit),
                "--vae",
                str(vae),
                "--tokens",
                args.tokens,
                "--seq-len",
                "2",
                "--shift",
                "3",
                "--seed",
                str(args.seed),
                "--threads",
                threads,
            ],
            env={
                "ACE_GGML_ALLOW_TEXT_DIM_MISMATCH": "1",
                "ACE_GGML_TEXT_MAX_LAYERS": "4",
            },
            timeout_s=args.timeout_s,
        ),
    ]

    if args.run_heavy_lm:
        stages.append(
            Stage(
                name="pipeline_heavy_lm_seq1",
                cmd=[
                    bin_path,
                    "--pipeline",
                    "--text-encoder",
                    str(lm17),
                    "--dit",
                    str(dit),
                    "--vae",
                    str(vae),
                    "--tokens",
                    args.heavy_tokens,
                    "--seq-len",
                    "1",
                    "--shift",
                    "3",
                    "--seed",
                    str(args.seed),
                    "--threads",
                    threads,
                ],
                env={
                    "ACE_GGML_TEXT_MAX_LAYERS": str(args.heavy_text_layers),
                },
                timeout_s=args.heavy_timeout_s,
            )
        )

    return stages


def maybe_build(args: argparse.Namespace) -> bool:
    if not args.rebuild:
        return True

    steps = [
        [
            "cmake",
            "-S",
            "acestep_ggml",
            "-B",
            args.build_dir,
            "-DACE_GGML_ENABLE_METAL=OFF",
        ],
        [
            "cmake",
            "--build",
            args.build_dir,
            "-j",
            str(args.build_jobs),
        ],
    ]
    for cmd in steps:
        print("[BUILD]", " ".join(shlex.quote(x) for x in cmd))
        proc = subprocess.run(
            cmd,
            cwd=args.cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
        print(proc.stdout)
        if proc.returncode != 0:
            print(f"[FAIL] build step failed: return_code={proc.returncode}")
            return False
    return True


def parse_args() -> argparse.Namespace:
    default_model_root = "/Users/fmh/project/ACE-Step-1.5/Ace-Step1.5"
    default_cwd = "/Users/fmh/project/ACE-Step-1.5"
    default_build_dir = "/tmp/ace_ggml_build"
    default_bin = f"{default_build_dir}/ace_ggml_cli"
    ts = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    default_log_dir = f"/tmp/ace_ggml_stage_logs/{ts}"

    ap = argparse.ArgumentParser(description="Run staged Ace GGML pipeline tests with timeout and logs")
    ap.add_argument("--cwd", default=default_cwd)
    ap.add_argument("--model-root", default=default_model_root)
    ap.add_argument("--build-dir", default=default_build_dir)
    ap.add_argument("--bin", default=default_bin)
    ap.add_argument("--log-dir", default=default_log_dir)
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--build-jobs", type=int, default=8)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed", type=int, default=123)
    ap.add_argument("--tokens", default="1,2,3,4")
    ap.add_argument("--timeout-s", type=int, default=180)
    ap.add_argument("--keep-going", action="store_true")
    ap.add_argument("--run-heavy-lm", action="store_true")
    ap.add_argument("--heavy-timeout-s", type=int, default=900)
    ap.add_argument("--heavy-text-layers", type=int, default=4)
    ap.add_argument("--heavy-tokens", default="1,2,3,4,5,6,7,8")
    ap.add_argument(
        "--export-json",
        nargs="?",
        const="auto",
        default="",
        help="Export run summary JSON. Use without value to write <log-dir>/summary.json",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    log_dir = pathlib.Path(args.log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] cwd={args.cwd}")
    print(f"[INFO] model_root={args.model_root}")
    print(f"[INFO] bin={args.bin}")
    print(f"[INFO] log_dir={log_dir}")

    if not maybe_build(args):
        return 2

    stages = build_stages(args)
    passed = 0
    failed = 0
    stage_results: List[Dict[str, Any]] = []
    run_start = time.time()

    for stage in stages:
        result = run_stage(stage, args.cwd, log_dir)
        stage_results.append(result)
        ok = bool(result["ok"])
        if ok:
            passed += 1
        else:
            failed += 1
            if not args.keep_going:
                break

    total_elapsed = time.time() - run_start
    print(f"[SUMMARY] passed={passed} failed={failed} total={len(stages)} elapsed_s={total_elapsed:.1f}")

    if args.export_json:
        if args.export_json == "auto":
            export_path = log_dir / "summary.json"
        else:
            p = pathlib.Path(args.export_json)
            export_path = p if p.is_absolute() else pathlib.Path(args.cwd) / p
        payload = {
            "created_at_utc": dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z",
            "cwd": args.cwd,
            "model_root": args.model_root,
            "bin": args.bin,
            "log_dir": str(log_dir),
            "passed": passed,
            "failed": failed,
            "total": len(stages),
            "elapsed_s": round(total_elapsed, 3),
            "keep_going": bool(args.keep_going),
            "run_heavy_lm": bool(args.run_heavy_lm),
            "stages": stage_results,
        }
        export_path.parent.mkdir(parents=True, exist_ok=True)
        with export_path.open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        print(f"[INFO] summary_json={export_path}")

    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
