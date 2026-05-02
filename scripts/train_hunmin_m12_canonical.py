#!/usr/bin/env python3
"""Train a legacy-12M-style Hunmin encoder on current canonical data.

This is an ablation, not a replacement for Stage2. It keeps the old small
encoder idea: char/special-token vocabulary, 6-layer Transformer, mean pooling,
and contrastive learning. The difference is that training rows come from the
current canonical Hunmin/UHPS multi-view JSONL data, not from the old legacy
transcription cache.

No BPE vocabulary is trained. The vocabulary is a bounded character-frequency
vocabulary plus fixed special tags.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, IterableDataset


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "[KO]", "[JA]", "[ZH]", "[EN]", "[HUNMIN]", "[UHPS]"]
MODEL_NAME = "hunmin-lite-13m-4lang-canonical-run1"
LANGUAGES = ["ko", "ja", "zh", "en"]
LANGUAGE_COUNT = len(LANGUAGES)


@dataclass
class M12Config:
    model_name: str = MODEL_NAME
    languages: list[str] = field(default_factory=lambda: list(LANGUAGES))
    special_tokens: list[str] = field(default_factory=lambda: list(SPECIAL_TOKENS))
    max_len: int = 160
    d_model: int = 384
    n_heads: int = 8
    n_layers: int = 6
    ff_dim: int = 1536
    dropout: float = 0.1
    temperature: float = 0.05
    max_vocab: int = 6000
    vocab_max_lines: int = 1_000_000
    steps: int = 20_000
    batch_size: int = 128
    lr: float = 2e-4
    weight_decay: float = 0.01
    eval_interval: int = 1000
    save_interval: int = 5000
    log_interval: int = 50
    shuffle_buffer: int = 20_000
    eval_rows: int = 12_000
    seed: int = 20260501


class CharTagTokenizer:
    def __init__(self, vocab: dict[str, int], max_len: int, special_tokens: list[str] | None = None):
        self.vocab = vocab
        self.id_to_token = {idx: tok for tok, idx in vocab.items()}
        self.max_len = max_len
        self.pad_id = vocab["<PAD>"]
        self.unk_id = vocab["<UNK>"]
        self.special_tokens = special_tokens or list(SPECIAL_TOKENS)
        self.scan_tokens = sorted([tok for tok in self.special_tokens if tok.startswith("[")], key=len, reverse=True)

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def _next_special(self, text: str, start: int) -> int:
        positions = [text.find(tok, start) for tok in self.scan_tokens]
        positions = [p for p in positions if p >= 0]
        return min(positions) if positions else len(text)

    def tokens(self, text: str) -> list[str]:
        out: list[str] = []
        i = 0
        while i < len(text):
            matched = None
            for tok in self.scan_tokens:
                if text.startswith(tok, i):
                    matched = tok
                    break
            if matched:
                out.append(matched)
                i += len(matched)
                continue
            j = self._next_special(text, i)
            out.extend(text[i:j])
            i = j
        return out

    def encode(self, text: str) -> tuple[list[int], list[int]]:
        ids = [self.vocab.get(tok, self.unk_id) for tok in self.tokens(text)[: self.max_len]]
        mask = [1] * len(ids)
        if len(ids) < self.max_len:
            pad_n = self.max_len - len(ids)
            ids.extend([self.pad_id] * pad_n)
            mask.extend([0] * pad_n)
        return ids, mask


class M12Encoder(nn.Module):
    def __init__(self, vocab_size: int, cfg: M12Config, pad_id: int):
        super().__init__()
        self.pad_id = pad_id
        self.token_emb = nn.Embedding(vocab_size, cfg.d_model, padding_idx=pad_id)
        self.pos_emb = nn.Embedding(cfg.max_len, cfg.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.ff_dim,
            dropout=cfg.dropout,
            activation="gelu",
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, cfg.n_layers)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        bsz, seq_len = input_ids.shape
        pos = torch.arange(seq_len, device=input_ids.device).unsqueeze(0).expand(bsz, seq_len)
        x = self.token_emb(input_ids) + self.pos_emb(pos)
        x = self.encoder(x, src_key_padding_mask=attention_mask == 0)
        mask = attention_mask.unsqueeze(-1).to(x.dtype)
        pooled = (x * mask).sum(dim=1) / mask.sum(dim=1).clamp_min(1.0)
        return F.normalize(pooled, dim=-1)


class CanonicalJsonlDataset(IterableDataset):
    def __init__(self, path: Path, tokenizer: CharTagTokenizer, seed: int, shuffle_buffer: int):
        self.path = path
        self.tokenizer = tokenizer
        self.seed = seed
        self.shuffle_buffer = shuffle_buffer

    @staticmethod
    def choose_pair(obj: dict, rng: random.Random) -> tuple[str, str]:
        pairs = obj.get("positive_pairs") or [["text_a", "text_b"]]
        valid = [
            pair for pair in pairs
            if isinstance(pair, list) and len(pair) == 2 and pair[0] in obj and pair[1] in obj
        ]
        if not valid:
            raise ValueError(f"no valid positive_pairs for {obj.get('id') or obj.get('pair_id')}")
        left_key, right_key = rng.choice(valid)
        return str(obj[left_key]), str(obj[right_key])

    def __iter__(self):
        worker = torch.utils.data.get_worker_info()
        worker_id = worker.id if worker else 0
        rng = random.Random(self.seed + worker_id)
        buffer: list[tuple[str, str]] = []
        while True:
            with self.path.open("r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    item = self.choose_pair(json.loads(line), rng)
                    if self.shuffle_buffer <= 1:
                        yield self._encode(item)
                        continue
                    buffer.append(item)
                    if len(buffer) >= self.shuffle_buffer:
                        idx = rng.randrange(len(buffer))
                        yield self._encode(buffer.pop(idx))
                rng.shuffle(buffer)
                while buffer:
                    yield self._encode(buffer.pop())

    def _encode(self, pair: tuple[str, str]):
        left_ids, left_mask = self.tokenizer.encode(pair[0])
        right_ids, right_mask = self.tokenizer.encode(pair[1])
        return (
            torch.tensor(left_ids, dtype=torch.long),
            torch.tensor(left_mask, dtype=torch.long),
            torch.tensor(right_ids, dtype=torch.long),
            torch.tensor(right_mask, dtype=torch.long),
        )


def iter_pair_texts(path: Path, limit: int | None = None) -> Iterable[str]:
    seen = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            obj = json.loads(line)
            pairs = obj.get("positive_pairs") or []
            keys = set()
            for pair in pairs:
                if isinstance(pair, list) and len(pair) == 2:
                    keys.update(pair)
            if not keys:
                keys.update(k for k in ("text_a", "text_b", "hunmin_a", "hunmin_b") if k in obj)
            for key in keys:
                value = obj.get(key)
                if isinstance(value, str) and value:
                    yield value
            seen += 1
            if limit is not None and seen >= limit:
                break


def build_vocab(train_path: Path, cfg: M12Config) -> dict[str, int]:
    counts: Counter[str] = Counter()
    scanner = CharTagTokenizer({tok: i for i, tok in enumerate(cfg.special_tokens)}, cfg.max_len, cfg.special_tokens)
    for text in iter_pair_texts(train_path, cfg.vocab_max_lines):
        counts.update(tok for tok in scanner.tokens(text) if tok not in cfg.special_tokens)
    vocab = {tok: i for i, tok in enumerate(cfg.special_tokens)}
    for token, _count in counts.most_common(max(0, cfg.max_vocab - len(vocab))):
        if token not in vocab:
            vocab[token] = len(vocab)
    return vocab


def contrastive_loss(z1: torch.Tensor, z2: torch.Tensor, temperature: float) -> torch.Tensor:
    logits = z1 @ z2.t() / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return (F.cross_entropy(logits, labels) + F.cross_entropy(logits.t(), labels)) / 2


@torch.no_grad()
def embed_texts(model: M12Encoder, tokenizer: CharTagTokenizer, texts: list[str], device: torch.device, batch_size: int) -> torch.Tensor:
    out = []
    model.eval()
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        ids = []
        masks = []
        for text in chunk:
            row_ids, row_mask = tokenizer.encode(text)
            ids.append(row_ids)
            masks.append(row_mask)
        input_ids = torch.tensor(ids, dtype=torch.long, device=device)
        attention_mask = torch.tensor(masks, dtype=torch.long, device=device)
        out.append(model(input_ids, attention_mask).cpu())
    return torch.cat(out, dim=0)


def retrieval_metrics(q: torch.Tensor, t: torch.Tensor) -> dict:
    sim = q @ t.t()
    ranks = []
    for i in range(sim.size(0)):
        order = sim[i].argsort(descending=True)
        rank = (order == i).nonzero(as_tuple=True)[0].item() + 1
        ranks.append(rank)
    n = len(ranks)
    return {
        "n": n,
        "recall_at_1": sum(r == 1 for r in ranks) / n,
        "recall_at_5": sum(r <= 5 for r in ranks) / n,
        "recall_at_10": sum(r <= 10 for r in ranks) / n,
        "mrr": sum(1 / r for r in ranks) / n,
        "median_rank": sorted(ranks)[n // 2],
    }


def read_eval_rows(path: Path, limit: int) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rows.append(json.loads(line))
            if len(rows) >= limit:
                break
    return rows


@torch.no_grad()
def evaluate(model: M12Encoder, tokenizer: CharTagTokenizer, val_path: Path, device: torch.device, batch_size: int, limit: int) -> dict:
    rows = read_eval_rows(val_path, limit)
    tasks = [
        ("text_a_to_text_b", "text_a", "text_b"),
        ("text_b_to_text_a", "text_b", "text_a"),
        ("hunmin_a_to_hunmin_b", "hunmin_a", "hunmin_b"),
        ("text_a_to_hunmin_a", "text_a", "hunmin_a"),
        ("text_b_to_hunmin_b", "text_b", "hunmin_b"),
        ("hunmin_a_to_text_a", "hunmin_a", "text_a"),
        ("hunmin_b_to_text_b", "hunmin_b", "text_b"),
        ("text_a_to_meaning", "text_a", "meaning"),
        ("text_b_to_meaning", "text_b", "meaning"),
        ("text_a_to_uhps_a", "text_a", "uhps_a"),
        ("text_b_to_uhps_b", "text_b", "uhps_b"),
        ("hunmin_a_to_uhps_a", "hunmin_a", "uhps_a"),
        ("hunmin_b_to_uhps_b", "hunmin_b", "uhps_b"),
    ]
    result = {}
    for name, q_key, t_key in tasks:
        task_rows = [row for row in rows if q_key in row and t_key in row]
        if not task_rows:
            continue
        q = [str(row[q_key]) for row in task_rows]
        t = [str(row[t_key]) for row in task_rows]
        q_z = embed_texts(model, tokenizer, q, device, batch_size)
        t_z = embed_texts(model, tokenizer, t, device, batch_size)
        result[name] = retrieval_metrics(q_z, t_z)
    if result:
        result["score_recall_at_1_mean"] = sum(v["recall_at_1"] for v in result.values()) / len(result)
    result["eval_rows"] = len(rows)
    return result


def save_checkpoint(path: Path, model: M12Encoder, cfg: M12Config, vocab: dict[str, int], step: int, metrics: dict | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state": model.state_dict(),
            "model_config": asdict(cfg),
            "vocab": vocab,
            "step": step,
            "metrics": metrics or {},
            "type": "hunmin_m12_canonical",
        },
        path,
    )


def load_config(path: Path | None) -> M12Config:
    if path is None:
        return M12Config()
    raw = json.loads(path.read_text(encoding="utf-8"))
    return M12Config(**raw.get("model", raw))


def resolve_device(name: str | None) -> torch.device:
    if name:
        return torch.device(name)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", type=Path)
    ap.add_argument("--train", type=Path, default=Path("output/embedder_run8_semantic_uhps/train.jsonl"))
    ap.add_argument("--val", type=Path, default=Path("output/embedder_run8_semantic_uhps/val.jsonl"))
    ap.add_argument("--output-dir", type=Path, default=Path(f"models/{MODEL_NAME}"))
    ap.add_argument("--steps", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--vocab-max-lines", type=int)
    ap.add_argument("--eval-rows", type=int)
    ap.add_argument("--eval-interval", type=int)
    ap.add_argument("--max-vocab", type=int)
    ap.add_argument("--device")
    ap.add_argument("--num-workers", type=int, default=0)
    args = ap.parse_args()

    cfg = load_config(args.config)
    if args.steps is not None:
        cfg.steps = args.steps
    if args.batch_size is not None:
        cfg.batch_size = args.batch_size
    if args.vocab_max_lines is not None:
        cfg.vocab_max_lines = args.vocab_max_lines
    if args.eval_rows is not None:
        cfg.eval_rows = args.eval_rows
    if args.eval_interval is not None:
        cfg.eval_interval = args.eval_interval
    if args.max_vocab is not None:
        cfg.max_vocab = args.max_vocab

    random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    device = resolve_device(args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    vocab_path = args.output_dir / "vocab.json"
    if vocab_path.exists():
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
    else:
        vocab = build_vocab(args.train, cfg)
        vocab_path.write_text(json.dumps(vocab, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    tokenizer = CharTagTokenizer(vocab, cfg.max_len, cfg.special_tokens)
    model = M12Encoder(tokenizer.vocab_size, cfg, tokenizer.pad_id).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, cfg.steps)
    scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")

    dataset = CanonicalJsonlDataset(args.train, tokenizer, cfg.seed, cfg.shuffle_buffer)
    loader = DataLoader(dataset, batch_size=cfg.batch_size, num_workers=args.num_workers, pin_memory=device.type == "cuda")
    iterator = iter(loader)
    log_path = args.output_dir / "log.jsonl"
    metadata = {
        "model_name": cfg.model_name,
        "model_family": "hunmin-lite",
        "declared_size": "13m",
        "language_count": len(cfg.languages),
        "languages": cfg.languages,
        "config": asdict(cfg),
        "train": str(args.train),
        "val": str(args.val),
        "output_dir": str(args.output_dir),
        "vocab_size": tokenizer.vocab_size,
        "params": sum(p.numel() for p in model.parameters()),
        "device": str(device),
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, ensure_ascii=False))

    best_score = -math.inf
    start = time.time()
    model.train()
    recent = []
    for step in range(1, cfg.steps + 1):
        left_ids, left_mask, right_ids, right_mask = next(iterator)
        left_ids = left_ids.to(device, non_blocking=True)
        left_mask = left_mask.to(device, non_blocking=True)
        right_ids = right_ids.to(device, non_blocking=True)
        right_mask = right_mask.to(device, non_blocking=True)

        with torch.amp.autocast("cuda", enabled=device.type == "cuda"):
            z_left = model(left_ids, left_mask)
            z_right = model(right_ids, right_mask)
            loss = contrastive_loss(z_left, z_right, cfg.temperature)

        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()
        scheduler.step()

        recent.append(float(loss.item()))
        if len(recent) > cfg.log_interval:
            recent.pop(0)
        if step % cfg.log_interval == 0:
            row = {
                "step": step,
                "train_loss": sum(recent) / len(recent),
                "lr": scheduler.get_last_lr()[0],
                "elapsed_sec": round(time.time() - start, 2),
            }
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

        if step % cfg.eval_interval == 0 or step == cfg.steps:
            metrics = evaluate(model, tokenizer, args.val, device, cfg.batch_size, cfg.eval_rows)
            score = metrics.get("score_recall_at_1_mean", 0.0)
            row = {"step": step, "eval": metrics, "elapsed_sec": round(time.time() - start, 2)}
            print(json.dumps(row, ensure_ascii=False), flush=True)
            with log_path.open("a", encoding="utf-8") as f:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
            if score > best_score:
                best_score = score
                save_checkpoint(args.output_dir / "best.pt", model, cfg, vocab, step, metrics)
            model.train()

        if step % cfg.save_interval == 0 or step == cfg.steps:
            save_checkpoint(args.output_dir / f"checkpoint_step_{step}.pt", model, cfg, vocab, step)
            save_checkpoint(args.output_dir / "last.pt", model, cfg, vocab, step)


if __name__ == "__main__":
    main()
