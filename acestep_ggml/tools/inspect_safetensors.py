#!/usr/bin/env python3
import argparse
import json
from collections import defaultdict
from safetensors import safe_open


def summarize(path: str, output_json: str | None = None) -> None:
    data = {
        "path": path,
        "tensors": [],
        "prefix_counts": {},
        "prefix_sizes": {},
        "total_tensors": 0,
    }

    prefix_counts = defaultdict(int)
    prefix_sizes = defaultdict(int)

    with safe_open(path, framework="np", device="cpu") as f:
        for key in f.keys():
            sl = f.get_slice(key)
            shape = sl.get_shape()
            dtype = sl.get_dtype()
            data["tensors"].append({
                "name": key,
                "shape": shape,
                "dtype": dtype,
            })
            data["total_tensors"] += 1
            parts = key.split(".")
            prefix = parts[0] if parts else key
            prefix_counts[prefix] += 1
            numel = 1
            for d in shape:
                numel *= d
            prefix_sizes[prefix] += numel

    data["prefix_counts"] = dict(prefix_counts)
    data["prefix_sizes"] = dict(prefix_sizes)

    if output_json:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"Wrote {output_json}")
        return

    print(f"== {path} ==")
    for t in data["tensors"]:
        print(f"{t['name']}\t{tuple(t['shape'])}\t{t['dtype']}")
    print(f"-- total tensors: {data['total_tensors']}")
    print("-- prefix counts:")
    for p, c in sorted(prefix_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {p}: {c} tensors, {prefix_sizes[p]} elements")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+", help="safetensors files")
    ap.add_argument("--json", dest="json_path", help="write JSON summary")
    args = ap.parse_args()
    for p in args.paths:
        summarize(p, output_json=args.json_path)


if __name__ == "__main__":
    main()
