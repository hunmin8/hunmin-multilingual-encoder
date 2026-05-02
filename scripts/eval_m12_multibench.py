#!/usr/bin/env python3
"""Run multiple diagnostics for M12-style Hunmin encoders.

This evaluates a checkpoint from scripts/train_hunmin_m12_canonical.py across
several JSONL benchmarks with the same retrieval metrics. It is intentionally
inference-only: no training, no tokenizer changes, no corpus changes.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_hunmin_m12_canonical import (  # noqa: E402
    CharTagTokenizer,
    M12Config,
    M12Encoder,
    embed_texts,
    resolve_device,
)


def stable_score(row: dict, seed: int, salt: str) -> int:
    key = f"{seed}\t{salt}\t{row.get('id') or row.get('pair_id') or row.get('text_a', '')}\t{row.get('text_b', '')}"
    return int.from_bytes(hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest(), "big")


def read_rows(path: Path, max_scan: int | None = None) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if not line.strip() or line.startswith("\x00"):
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "text_a" in obj and "text_b" in obj:
                rows.append(obj)
            if max_scan and len(rows) >= max_scan:
                break
    return rows


def sample_rows(rows: list[dict], limit: int, seed: int, salt: str) -> list[dict]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    return sorted(rows, key=lambda r: stable_score(r, seed, salt))[:limit]


def group_lang_pair(row: dict) -> str:
    return f"{row.get('lang_a', '?')}->{row.get('lang_b', '?')}"


def length_bucket(row: dict) -> str:
    avg = (len(str(row.get("text_a") or "")) + len(str(row.get("text_b") or ""))) / 2
    if avg <= 60:
        return "short_<=60"
    if avg <= 140:
        return "medium_<=140"
    return "long_>140"


def group_rows(rows: list[dict], key: str) -> dict[str, list[dict]]:
    out: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        if key == "lang_pair":
            value = group_lang_pair(row)
        elif key == "source":
            value = str(row.get("source") or "unknown")
        elif key == "record_kind":
            value = str(row.get("record_kind") or "unknown")
        elif key == "length":
            value = length_bucket(row)
        else:
            raise ValueError(key)
        out[value].append(row)
    return dict(out)


def load_model(checkpoint: Path, device_name: str | None):
    ckpt = torch.load(checkpoint, map_location="cpu")
    cfg = M12Config(**ckpt["model_config"])
    vocab = ckpt["vocab"]
    tokenizer = CharTagTokenizer(vocab, cfg.max_len, cfg.special_tokens)
    model = M12Encoder(len(vocab), cfg, tokenizer.pad_id)
    model.load_state_dict(ckpt["model_state"])
    device = resolve_device(device_name)
    model.to(device)
    model.eval()
    return model, tokenizer, cfg, device, {
        "checkpoint": str(checkpoint),
        "step": ckpt.get("step"),
        "type": ckpt.get("type"),
        "model_name": cfg.model_name,
        "languages": cfg.languages,
        "vocab_size": len(vocab),
        "params": sum(p.numel() for p in model.parameters()),
        "max_len": cfg.max_len,
    }


def metrics_from_ranks(ranks: list[int]) -> dict:
    if not ranks:
        return {"n": 0}
    ranks_sorted = sorted(ranks)
    n = len(ranks)
    return {
        "n": n,
        "recall_at_1": sum(r == 1 for r in ranks) / n,
        "recall_at_5": sum(r <= 5 for r in ranks) / n,
        "recall_at_10": sum(r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "median_rank": ranks_sorted[n // 2],
        "p90_rank": ranks_sorted[min(n - 1, int(n * 0.9))],
    }


@torch.no_grad()
def eval_direction(
    rows: list[dict],
    model: M12Encoder,
    tokenizer: CharTagTokenizer,
    device: torch.device,
    batch_size: int,
    q_key: str,
    t_key: str,
    failure_samples: int,
) -> dict:
    q_texts = [str(row[q_key]) for row in rows]
    t_texts = [str(row[t_key]) for row in rows]
    q_z = embed_texts(model, tokenizer, q_texts, device, batch_size=batch_size)
    t_z = embed_texts(model, tokenizer, t_texts, device, batch_size=batch_size)
    sim = q_z @ t_z.t()
    ranks: list[int] = []
    failures: list[dict] = []
    topk = min(5, sim.size(1))
    top_scores, top_idxs = torch.topk(sim, k=topk, dim=1)
    for i in range(sim.size(0)):
        row_scores = sim[i]
        correct_score = row_scores[i]
        rank = int((row_scores > correct_score).sum().item() + 1)
        ranks.append(rank)
        if rank > 1 and len(failures) < failure_samples:
            failures.append(
                {
                    "rank": rank,
                    "query": q_texts[i],
                    "gold": t_texts[i],
                    "lang_pair": group_lang_pair(rows[i]),
                    "source": rows[i].get("source"),
                    "record_kind": rows[i].get("record_kind"),
                    "top": [
                        {
                            "score": float(top_scores[i, j].item()),
                            "text": t_texts[int(top_idxs[i, j].item())],
                            "is_gold": int(top_idxs[i, j].item()) == i,
                        }
                        for j in range(topk)
                    ],
                }
            )
    result = metrics_from_ranks(ranks)
    if failure_samples:
        result["failures"] = failures
    return result


def eval_rows(
    rows: list[dict],
    model: M12Encoder,
    tokenizer: CharTagTokenizer,
    device: torch.device,
    batch_size: int,
    failure_samples: int = 0,
) -> dict:
    result = {}
    for name, q_key, t_key in (
        ("text_a_to_text_b", "text_a", "text_b"),
        ("text_b_to_text_a", "text_b", "text_a"),
    ):
        result[name] = eval_direction(rows, model, tokenizer, device, batch_size, q_key, t_key, failure_samples)
    result["mean_recall_at_1"] = (
        result["text_a_to_text_b"]["recall_at_1"] + result["text_b_to_text_a"]["recall_at_1"]
    ) / 2
    result["mean_recall_at_5"] = (
        result["text_a_to_text_b"]["recall_at_5"] + result["text_b_to_text_a"]["recall_at_5"]
    ) / 2
    result["mean_mrr"] = (result["text_a_to_text_b"]["mrr"] + result["text_b_to_text_a"]["mrr"]) / 2
    return result


def cosine_stats(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    ordered = sorted(values)

    def pct(p: float) -> float:
        return ordered[min(len(ordered) - 1, max(0, round((len(ordered) - 1) * p)))]

    return {
        "n": len(values),
        "mean": statistics.fmean(values),
        "p50": pct(0.50),
        "p90": pct(0.90),
        "p95": pct(0.95),
        "min": ordered[0],
        "max": ordered[-1],
    }


@torch.no_grad()
def pair_cosines(rows: list[dict], model, tokenizer, device, batch_size: int) -> list[float]:
    if not rows:
        return []
    left = [str(row["text_a"]) for row in rows]
    right = [str(row["text_b"]) for row in rows]
    left_z = embed_texts(model, tokenizer, left, device, batch_size=batch_size)
    right_z = embed_texts(model, tokenizer, right, device, batch_size=batch_size)
    return (left_z * right_z).sum(dim=1).cpu().tolist()


def parse_benchmark(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        path = Path(spec)
        return path.stem, path
    name, path = spec.split("=", 1)
    return name, Path(path)


def benchmark_summary_markdown(results: dict) -> str:
    lines = [
        "# Hunmin M12 Multi Benchmark",
        "",
        f"checkpoint: `{results['model']['checkpoint']}`",
        f"model: `{results['model']['model_name']}`",
        f"step: `{results['model']['step']}`",
        "",
        "## Overall",
        "",
        "| benchmark | n | R@1 | R@5 | MRR |",
        "|---|---:|---:|---:|---:|",
    ]
    for name, bench in results["benchmarks"].items():
        overall = bench.get("overall", {})
        lines.append(
            f"| {name} | {overall.get('text_a_to_text_b', {}).get('n', 0)} "
            f"| {overall.get('mean_recall_at_1', 0):.4f} "
            f"| {overall.get('mean_recall_at_5', 0):.4f} "
            f"| {overall.get('mean_mrr', 0):.4f} |"
        )
    lines.extend(["", "## Notes", "", "- R@1/R@5/MRR are bidirectional averages unless stated otherwise."])
    return "\n".join(lines) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description="Run multi-benchmark diagnostics for M12-style encoders.")
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--benchmark", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--hard-negatives", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--overall-limit", type=int, default=12000)
    ap.add_argument("--group-limit", type=int, default=3000)
    ap.add_argument("--hard-limit", type=int, default=10000)
    ap.add_argument("--max-scan", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--failure-samples", type=int, default=10)
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--device")
    args = ap.parse_args()

    start = time.time()
    model, tokenizer, cfg, device, meta = load_model(args.checkpoint, args.device)
    results = {
        "model": meta,
        "args": {
            "overall_limit": args.overall_limit,
            "group_limit": args.group_limit,
            "hard_limit": args.hard_limit,
            "seed": args.seed,
            "device": str(device),
        },
        "benchmarks": {},
        "hard_negative": {},
    }

    for spec in args.benchmark:
        name, path = parse_benchmark(spec)
        all_rows = read_rows(path, args.max_scan or None)
        bench: dict = {"path": str(path), "rows_available": len(all_rows)}
        overall_rows = sample_rows(all_rows, args.overall_limit, args.seed, name)
        bench["overall"] = eval_rows(
            overall_rows,
            model,
            tokenizer,
            device,
            args.batch_size,
            failure_samples=args.failure_samples,
        )
        for group_name in ("lang_pair", "source", "record_kind", "length"):
            grouped = {}
            for value, group in sorted(group_rows(all_rows, group_name).items()):
                if len(group) < 20:
                    continue
                group_sample = sample_rows(group, args.group_limit, args.seed, f"{name}:{group_name}:{value}")
                grouped[value] = eval_rows(group_sample, model, tokenizer, device, args.batch_size)
            bench[f"by_{group_name}"] = grouped
        results["benchmarks"][name] = bench

    if args.hard_negatives and args.hard_negatives.exists():
        hard_rows = sample_rows(read_rows(args.hard_negatives), args.hard_limit, args.seed, "hard")
        first_bench = next(iter(results["benchmarks"].values()), None)
        positive_rows = []
        if first_bench:
            # Re-read the first benchmark to keep positive/hard cosine comparable.
            positive_rows = sample_rows(read_rows(Path(first_bench["path"])), args.hard_limit, args.seed, "positive-cosine")
        pos_cos = pair_cosines(positive_rows, model, tokenizer, device, args.batch_size)
        hard_cos = pair_cosines(hard_rows, model, tokenizer, device, args.batch_size)
        results["hard_negative"] = {
            "path": str(args.hard_negatives),
            "positive_cosine": cosine_stats(pos_cos),
            "hard_negative_cosine": cosine_stats(hard_cos),
        }
        if pos_cos and hard_cos:
            results["hard_negative"]["positive_minus_hard_mean"] = (
                results["hard_negative"]["positive_cosine"]["mean"]
                - results["hard_negative"]["hard_negative_cosine"]["mean"]
            )

    results["elapsed_sec"] = round(time.time() - start, 2)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output.with_suffix(".md").write_text(benchmark_summary_markdown(results), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "elapsed_sec": results["elapsed_sec"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
