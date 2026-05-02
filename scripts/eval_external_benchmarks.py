#!/usr/bin/env python3
"""External benchmark evaluation for Hunmin encoders and HF baselines.

Benchmarks:
- FLORES-200 devtest: parallel sentence retrieval over selected languages.
- KLUE STS validation: cosine-vs-human-score correlation.

This is deliberately separate from internal OPUS/query-doc training benchmarks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))


LANGS = {
    "en": ("eng_Latn", "[EN]"),
    "ko": ("kor_Hang", "[KO]"),
    "ja": ("jpn_Jpan", "[JA]"),
    "zh": ("zho_Hans", "[ZH]"),
    "fr": ("fra_Latn", "[FR]"),
    "de": ("deu_Latn", "[DE]"),
    "es": ("spa_Latn", "[ES]"),
}

HF_PRESETS = {
    "mbert": ("bert-base-multilingual-cased", "mean", "none"),
    "e5-base": ("intfloat/multilingual-e5-base", "mean", "e5"),
    "labse": ("sentence-transformers/LaBSE", "cls_or_pooler", "none"),
    "mpnet": ("sentence-transformers/paraphrase-multilingual-mpnet-base-v2", "mean", "none"),
    "bge-m3": ("BAAI/bge-m3", "cls", "none"),
    # Korean-focused public encoders. These are evaluated with the same
    # transformer-only loader as the multilingual baselines so the comparison
    # remains reproducible in minimal training environments.
    "ko-sroberta": ("jhgan/ko-sroberta-multitask", "mean", "none"),
    "kosimcse-roberta": ("BM-K/KoSimCSE-roberta-multitask", "cls", "none"),
    "kosimcse-bert": ("BM-K/KoSimCSE-bert-multitask", "cls", "none"),
    "kr-sbert": ("snunlp/KR-SBERT-V40K-klueNLI-augSTS", "mean", "none"),
    "koe5": ("nlpai-lab/KoE5", "mean", "e5"),
    "kure-v1": ("nlpai-lab/KURE-v1", "cls", "none"),
}


def normalize_np(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype="float32")
    n = np.linalg.norm(x, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return x / n


def retrieval_metrics(q: np.ndarray, t: np.ndarray, chunk_size: int = 512) -> dict:
    q = normalize_np(q)
    t = normalize_np(t)
    n = len(q)
    ranks = []
    for start in range(0, n, chunk_size):
        sim = q[start:start + chunk_size] @ t.T
        gold = np.arange(start, min(start + chunk_size, n))
        gold_scores = sim[np.arange(len(gold)), gold]
        ranks.extend(((sim > gold_scores[:, None]).sum(axis=1) + 1).tolist())
    ranks = np.asarray(ranks)
    return {
        "n": int(n),
        "recall_at_1": float((ranks <= 1).mean()),
        "recall_at_5": float((ranks <= 5).mean()),
        "recall_at_10": float((ranks <= 10).mean()),
        "mrr": float((1.0 / ranks).mean()),
        "median_rank": int(np.median(ranks)),
    }


class Encoder:
    key: str

    def encode(self, texts: list[str], batch_size: int, side: str = "passage") -> np.ndarray:
        raise NotImplementedError


class HunminM12Encoder(Encoder):
    def __init__(self, key: str, checkpoint: Path, device: str):
        from train_hunmin_m12_canonical import CharTagTokenizer, M12Config, M12Encoder, embed_texts

        self.key = key
        self.device = torch.device(device)
        self.embed_texts = embed_texts
        ckpt = torch.load(checkpoint, map_location="cpu")
        cfg = M12Config(**ckpt["model_config"])
        vocab = ckpt["vocab"]
        tok = CharTagTokenizer(vocab, cfg.max_len, cfg.special_tokens)
        model = M12Encoder(tok.vocab_size, cfg, tok.pad_id)
        model.load_state_dict(ckpt["model_state"])
        model.to(self.device)
        model.eval()
        self.cfg = cfg
        self.tokenizer = tok
        self.model = model
        self.meta = {
            "type": "hunmin_m12",
            "checkpoint": str(checkpoint),
            "model_name": cfg.model_name,
            "params": sum(p.numel() for p in model.parameters()),
            "vocab_size": tok.vocab_size,
            "languages": cfg.languages,
        }

    def encode(self, texts: list[str], batch_size: int, side: str = "passage") -> np.ndarray:
        z = self.embed_texts(self.model, self.tokenizer, texts, self.device, batch_size)
        return z.numpy().astype("float32")


def mean_pool(last_hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).to(last_hidden.dtype)
    return (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)


class HFEncoder(Encoder):
    def __init__(self, key: str, device: str, max_len: int, torch_dtype: str):
        from transformers import AutoModel, AutoTokenizer

        if key not in HF_PRESETS:
            raise SystemExit(f"unknown HF preset {key}")
        name, pooling, prefix = HF_PRESETS[key]
        self.key = key
        self.name = name
        self.pooling = pooling
        self.prefix = prefix
        self.device = torch.device(device)
        self.max_len = max_len
        dtype = {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16, "auto": None}[torch_dtype]
        kwargs = {}
        if dtype is not None:
            kwargs["torch_dtype"] = dtype
        self.tokenizer = AutoTokenizer.from_pretrained(name, fix_mistral_regex=False)
        self.model = AutoModel.from_pretrained(name, **kwargs)
        self.model.to(self.device)
        self.model.eval()
        self.meta = {"type": "hf", "model_name": name, "pooling": pooling, "prefix": prefix}

    def _prefix(self, texts: list[str], side: str) -> list[str]:
        if self.prefix == "e5":
            p = "query: " if side == "query" else "passage: "
            return [p + x for x in texts]
        return texts

    @torch.no_grad()
    def encode(self, texts: list[str], batch_size: int, side: str = "passage") -> np.ndarray:
        texts = self._prefix(texts, side)
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
            pooled = F.normalize(pooled.float(), dim=-1)
            outs.append(pooled.cpu().numpy().astype("float32"))
        return np.concatenate(outs, axis=0)


def read_flores_split(root: Path, split: str, langs: list[str], tagged: bool) -> dict[str, list[str]]:
    out = {}
    for lang in langs:
        flores_code, tag = LANGS[lang]
        path = root / split / f"{flores_code}.{split}"
        rows = [line.rstrip("\n") for line in path.read_text(encoding="utf-8").splitlines()]
        out[lang] = [f"{tag} {x}" for x in rows] if tagged else rows
    return out


def eval_flores(encoder: Encoder, root: Path, split: str, langs: list[str], batch_size: int, tagged: bool) -> dict:
    texts = read_flores_split(root, split, langs, tagged=tagged)
    emb = {lang: encoder.encode(texts[lang], batch_size, side="passage") for lang in langs}
    tasks = {}
    for src in langs:
        for tgt in langs:
            if src == tgt:
                continue
            key = f"{src}->{tgt}"
            tasks[key] = retrieval_metrics(emb[src], emb[tgt])
    en_tasks = {k: v for k, v in tasks.items() if k.startswith("en->") or "->en" in k}
    return {
        "split": split,
        "langs": langs,
        "sentences": len(next(iter(texts.values()))),
        "tasks": tasks,
        "mean_recall_at_1": float(np.mean([v["recall_at_1"] for v in tasks.values()])),
        "mean_recall_at_5": float(np.mean([v["recall_at_5"] for v in tasks.values()])),
        "mean_mrr": float(np.mean([v["mrr"] for v in tasks.values()])),
        "en_mean_recall_at_1": float(np.mean([v["recall_at_1"] for v in en_tasks.values()])),
        "en_mean_recall_at_5": float(np.mean([v["recall_at_5"] for v in en_tasks.values()])),
    }


def eval_klue_sts(encoder: Encoder, batch_size: int, tagged: bool) -> dict:
    from datasets import load_dataset

    ds = load_dataset("klue", "sts", split="validation")
    s1 = [row["sentence1"] for row in ds]
    s2 = [row["sentence2"] for row in ds]
    if tagged:
        s1 = ["[KO] " + x for x in s1]
        s2 = ["[KO] " + x for x in s2]
    labels = np.asarray([float(row["labels"]["real-label"]) for row in ds], dtype="float32")
    binary = np.asarray([int(row["labels"]["binary-label"]) for row in ds], dtype="int32")
    z1 = normalize_np(encoder.encode(s1, batch_size, side="query"))
    z2 = normalize_np(encoder.encode(s2, batch_size, side="passage"))
    sims = (z1 * z2).sum(axis=1)
    spear = spearmanr(labels, sims).correlation
    pear = pearsonr(labels, sims).statistic
    # Best threshold on this validation set. Diagnostic only.
    thresholds = np.linspace(float(sims.min()), float(sims.max()), 200)
    acc = max(float(((sims >= t).astype("int32") == binary).mean()) for t in thresholds)
    return {
        "n": len(s1),
        "spearman": float(spear),
        "pearson": float(pear),
        "binary_best_acc": acc,
        "sim_mean": float(sims.mean()),
        "sim_std": float(sims.std()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--hunmin", action="append", default=[], help="key=checkpoint.pt")
    ap.add_argument("--hf", action="append", default=[], help=f"HF preset: {','.join(HF_PRESETS)}")
    ap.add_argument("--flores-root", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--langs", default="en,ko,ja,zh,fr,de,es")
    ap.add_argument("--flores-split", default="devtest")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--max-len", type=int, default=160)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--torch-dtype", choices=["auto", "float32", "float16", "bfloat16"], default="float16")
    args = ap.parse_args()

    encoders: list[Encoder] = []
    for item in args.hunmin:
        key, ckpt = item.split("=", 1)
        encoders.append(HunminM12Encoder(key, Path(ckpt), args.device))
    for key in args.hf:
        encoders.append(HFEncoder(key, args.device, args.max_len, args.torch_dtype))

    langs = [x.strip() for x in args.langs.split(",") if x.strip()]
    result = {
        "args": {
            "langs": langs,
            "flores_split": args.flores_split,
            "flores_root": str(args.flores_root),
            "batch_size": args.batch_size,
        },
        "models": {},
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    for enc in encoders:
        started = time.perf_counter()
        tagged = isinstance(enc, HunminM12Encoder)
        row = {
            "meta": enc.meta,
            "flores": eval_flores(enc, args.flores_root, args.flores_split, langs, args.batch_size, tagged=tagged),
            "klue_sts": eval_klue_sts(enc, args.batch_size, tagged=tagged),
        }
        row["elapsed_sec"] = round(time.perf_counter() - started, 2)
        result["models"][enc.key] = row
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"model": enc.key, "elapsed_sec": row["elapsed_sec"]}, ensure_ascii=False), flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
