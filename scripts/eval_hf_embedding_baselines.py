#!/usr/bin/env python3
"""Evaluate HuggingFace embedding baselines on Hunmin JSONL pair benchmarks.

This intentionally avoids sentence-transformers so the d2/H100 environments can
run it with only transformers + torch. Inputs are the same JSONL pair files used
for the Hunmin M12-style encoder evaluations.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import re
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


TAG_RE = re.compile(r"^\[[A-Z_]+\]\s*")

MODEL_PRESETS = {
    "mbert": {
        "name": "bert-base-multilingual-cased",
        "pooling": "mean",
        "prefix": "none",
    },
    "e5-base": {
        "name": "intfloat/multilingual-e5-base",
        "pooling": "mean",
        "prefix": "e5",
    },
    "labse": {
        "name": "sentence-transformers/LaBSE",
        "pooling": "cls_or_pooler",
        "prefix": "none",
    },
    "mpnet": {
        "name": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "pooling": "mean",
        "prefix": "none",
    },
    "bge-m3": {
        "name": "BAAI/bge-m3",
        "pooling": "cls",
        "prefix": "none",
    },
    "ko-sroberta": {
        "name": "jhgan/ko-sroberta-multitask",
        "pooling": "mean",
        "prefix": "none",
    },
    "kosimcse-roberta": {
        "name": "BM-K/KoSimCSE-roberta-multitask",
        "pooling": "cls",
        "prefix": "none",
    },
    "kosimcse-bert": {
        "name": "BM-K/KoSimCSE-bert-multitask",
        "pooling": "cls",
        "prefix": "none",
    },
    "kr-sbert": {
        "name": "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
        "pooling": "mean",
        "prefix": "none",
    },
    "koe5": {
        "name": "nlpai-lab/KoE5",
        "pooling": "mean",
        "prefix": "e5",
    },
    "kure-v1": {
        "name": "nlpai-lab/KURE-v1",
        "pooling": "cls",
        "prefix": "none",
    },
}


def strip_tag(text: str) -> str:
    return TAG_RE.sub("", text).strip()


def read_pairs(path: Path, limit: int, seed: int, strip_tags: bool) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            row = json.loads(line)
            if "text_a" not in row or "text_b" not in row:
                continue
            if strip_tags:
                row = dict(row)
                row["text_a"] = strip_tag(str(row["text_a"]))
                row["text_b"] = strip_tag(str(row["text_b"]))
            rows.append(row)
    random.Random(seed).shuffle(rows)
    return rows[:limit] if limit else rows


def l2_normalize(x: torch.Tensor) -> torch.Tensor:
    return F.normalize(x, p=2, dim=-1)


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class HFEncoder:
    def __init__(self, model_key: str, device: str, max_len: int, torch_dtype: str):
        if model_key not in MODEL_PRESETS:
            raise SystemExit(f"unknown model key {model_key}; choices={sorted(MODEL_PRESETS)}")
        cfg = MODEL_PRESETS[model_key]
        self.key = model_key
        self.name = cfg["name"]
        self.pooling = cfg["pooling"]
        self.prefix = cfg["prefix"]
        self.device = torch.device(device)
        self.max_len = max_len
        dtype = {
            "auto": None,
            "float32": torch.float32,
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
        }[torch_dtype]
        self.tokenizer = AutoTokenizer.from_pretrained(self.name, fix_mistral_regex=False)
        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.model = AutoModel.from_pretrained(self.name, **kwargs)
        self.model.to(self.device)
        self.model.eval()

    def _prefix_texts(self, texts: list[str], side: str) -> list[str]:
        if self.prefix == "e5":
            prefix = "query: " if side == "query" else "passage: "
            return [prefix + text for text in texts]
        return texts

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int, side: str) -> np.ndarray:
        texts = self._prefix_texts(texts, side)
        outs = []
        for start in range(0, len(texts), batch_size):
            chunk = texts[start:start + batch_size]
            batch = self.tokenizer(
                chunk,
                padding=True,
                truncation=True,
                max_length=self.max_len,
                return_tensors="pt",
            )
            batch = {k: v.to(self.device) for k, v in batch.items()}
            out = self.model(**batch)
            if self.pooling == "mean":
                pooled = mean_pool(out.last_hidden_state, batch["attention_mask"])
            elif self.pooling == "cls":
                pooled = out.last_hidden_state[:, 0]
            elif self.pooling == "cls_or_pooler":
                pooled = out.pooler_output if getattr(out, "pooler_output", None) is not None else out.last_hidden_state[:, 0]
            else:
                raise AssertionError(self.pooling)
            pooled = l2_normalize(pooled.float()).cpu().numpy().astype("float32")
            outs.append(pooled)
        return np.concatenate(outs, axis=0)


def retrieval_metrics(q: np.ndarray, t: np.ndarray, chunk_size: int = 512) -> dict:
    n = len(q)
    ranks = []
    top_scores = []
    for start in range(0, n, chunk_size):
        sim = q[start:start + chunk_size] @ t.T
        gold = np.arange(start, min(start + chunk_size, n))
        gold_scores = sim[np.arange(len(gold)), gold]
        rank = (sim > gold_scores[:, None]).sum(axis=1) + 1
        ranks.extend(rank.tolist())
        top_scores.extend(sim.max(axis=1).tolist())
    ranks_np = np.asarray(ranks)
    return {
        "n": n,
        "recall_at_1": float((ranks_np <= 1).mean()),
        "recall_at_5": float((ranks_np <= 5).mean()),
        "recall_at_10": float((ranks_np <= 10).mean()),
        "mrr": float((1.0 / ranks_np).mean()),
        "median_rank": int(np.median(ranks_np)),
        "mean_top_score": float(np.mean(top_scores)),
    }


def evaluate_rows(rows: list[dict], q_emb: np.ndarray, t_emb: np.ndarray) -> dict:
    out = {
        "text_a_to_text_b": retrieval_metrics(q_emb, t_emb),
        "text_b_to_text_a": retrieval_metrics(t_emb, q_emb),
    }
    out["mean_recall_at_1"] = (out["text_a_to_text_b"]["recall_at_1"] + out["text_b_to_text_a"]["recall_at_1"]) / 2
    out["mean_recall_at_5"] = (out["text_a_to_text_b"]["recall_at_5"] + out["text_b_to_text_a"]["recall_at_5"]) / 2
    out["mean_mrr"] = (out["text_a_to_text_b"]["mrr"] + out["text_b_to_text_a"]["mrr"]) / 2

    by_lang_pair = defaultdict(list)
    by_record_kind = defaultdict(list)
    by_source = defaultdict(list)
    for idx, row in enumerate(rows):
        by_lang_pair[f"{row.get('lang_a','?')}->{row.get('lang_b','?')}"].append(idx)
        by_record_kind[str(row.get("record_kind", "unknown"))].append(idx)
        by_source[str(row.get("source", "unknown"))].append(idx)

    def subset_metrics(indices: list[int]) -> dict:
        idx = np.asarray(indices, dtype=np.int64)
        if len(idx) < 2:
            return {}
        m1 = retrieval_metrics(q_emb[idx], t_emb[idx])
        m2 = retrieval_metrics(t_emb[idx], q_emb[idx])
        return {
            "n": len(idx),
            "mean_recall_at_1": (m1["recall_at_1"] + m2["recall_at_1"]) / 2,
            "mean_recall_at_5": (m1["recall_at_5"] + m2["recall_at_5"]) / 2,
            "mean_mrr": (m1["mrr"] + m2["mrr"]) / 2,
        }

    out["by_lang_pair"] = {k: subset_metrics(v) for k, v in sorted(by_lang_pair.items()) if len(v) >= 20}
    out["by_record_kind"] = {k: subset_metrics(v) for k, v in sorted(by_record_kind.items()) if len(v) >= 20}
    out["by_source"] = {k: subset_metrics(v) for k, v in sorted(by_source.items()) if len(v) >= 20}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", action="append", required=True, help=f"one of {','.join(MODEL_PRESETS)}")
    ap.add_argument("--benchmark", action="append", required=True, help="name=path.jsonl")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=3000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=20260501)
    ap.add_argument("--keep-tags", action="store_true")
    ap.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    args = ap.parse_args()

    benchmarks = []
    for item in args.benchmark:
        name, path = item.split("=", 1)
        rows = read_pairs(Path(path), args.limit, args.seed, strip_tags=not args.keep_tags)
        benchmarks.append((name, Path(path), rows))

    result = {
        "args": {
            "models": args.model,
            "benchmarks": [{"name": n, "path": str(p), "rows": len(r)} for n, p, r in benchmarks],
            "limit": args.limit,
            "batch_size": args.batch_size,
            "max_len": args.max_len,
            "keep_tags": args.keep_tags,
            "torch_dtype": args.torch_dtype,
        },
        "models": {},
    }

    for model_key in args.model:
        started = time.perf_counter()
        encoder = HFEncoder(model_key, args.device, args.max_len, args.torch_dtype)
        model_result = {"model_name": encoder.name, "pooling": encoder.pooling, "prefix": encoder.prefix, "benchmarks": {}}
        for bench_name, _path, rows in benchmarks:
            left = [str(row["text_a"]) for row in rows]
            right = [str(row["text_b"]) for row in rows]
            q = encoder.encode(left, args.batch_size, side="query")
            t = encoder.encode(right, args.batch_size, side="passage")
            model_result["benchmarks"][bench_name] = evaluate_rows(rows, q, t)
        model_result["elapsed_sec"] = round(time.perf_counter() - started, 2)
        result["models"][model_key] = model_result
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"model": model_key, "elapsed_sec": model_result["elapsed_sec"]}, ensure_ascii=False), flush=True)

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
