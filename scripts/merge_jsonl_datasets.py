#!/usr/bin/env python3
"""Merge JSONL datasets with chunk-level interleaving.

This avoids placing all auxiliary rows at the end of a large train file while
remaining memory-safe for multi-GB datasets.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def read_chunk(handle, size: int) -> list[str]:
    out = []
    for _ in range(size):
        line = handle.readline()
        if not line:
            break
        if line.strip():
            out.append(line)
    return out


def merge(base_path: Path, aux_path: Path, output_path: Path, base_chunk: int, aux_chunk: int) -> dict:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counts = {"base": 0, "aux": 0, "output": 0}
    with base_path.open("r", encoding="utf-8") as base, aux_path.open("r", encoding="utf-8") as aux, output_path.open(
        "w", encoding="utf-8"
    ) as out:
        while True:
            base_rows = read_chunk(base, base_chunk)
            aux_rows = read_chunk(aux, aux_chunk)
            if not base_rows and not aux_rows:
                break
            for line in base_rows:
                out.write(line)
            for line in aux_rows:
                out.write(line)
            counts["base"] += len(base_rows)
            counts["aux"] += len(aux_rows)
            counts["output"] += len(base_rows) + len(aux_rows)
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description="Interleave JSONL datasets by chunks.")
    ap.add_argument("--base-train", type=Path, required=True)
    ap.add_argument("--base-val", type=Path, required=True)
    ap.add_argument("--aux-train", type=Path, required=True)
    ap.add_argument("--aux-val", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--base-chunk", type=int, default=4)
    ap.add_argument("--aux-chunk", type=int, default=1)
    args = ap.parse_args()

    stats = {
        "train": merge(args.base_train, args.aux_train, args.output_dir / "train.jsonl", args.base_chunk, args.aux_chunk),
        "val": merge(args.base_val, args.aux_val, args.output_dir / "val.jsonl", args.base_chunk, args.aux_chunk),
        "args": {
            "base_train": str(args.base_train),
            "base_val": str(args.base_val),
            "aux_train": str(args.aux_train),
            "aux_val": str(args.aux_val),
            "output_dir": str(args.output_dir),
            "base_chunk": args.base_chunk,
            "aux_chunk": args.aux_chunk,
        },
    }
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
