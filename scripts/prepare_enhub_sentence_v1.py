#!/usr/bin/env python3
"""Prepare EN-hub sentence-pair data for Hunmin encoder training.

This script intentionally builds a simple encoder dataset:

    [EN] sentence <-> [XX] translated sentence

The goal is to recover the clean 12M-style cross-lingual sentence objective
without mixing in search-layer artifacts. Hunmin/UHPS views are optional
auxiliary views; text<->text remains the primary positive pair.

No scribe rules, tokenizer rules, specs, or model files are modified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]


LANG_TAGS = {
    "en": "[EN]",
    "ko": "[KO]",
    "ja": "[JA]",
    "zh": "[ZH]",
    "fr": "[FR]",
    "de": "[DE]",
    "es": "[ES]",
}

SCRIPT_RE = {
    "en": re.compile(r"[A-Za-z]"),
    "ko": re.compile(r"[\uac00-\ud7a3]"),
    "ja": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "zh": re.compile(r"[\u3400-\u9fff]"),
    "fr": re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]"),
    "de": re.compile(r"[A-Za-zÄÖÜäöüß]"),
    "es": re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]"),
}

LATIN_WORD_RE = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+")
URL_RE = re.compile(r"https?://|www\.", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")
SPACE_RE = re.compile(r"\s+")
HTML_RE = re.compile(r"&[a-z]+;|<[^>]+>", re.I)


DEFAULT_SOURCES = [
    "/home/dragon/hunmin/data/opus_en_ko.tsv:en:ko:250000",
    "/home/dragon/hunmin/data/opus_en_ja.tsv:en:ja:250000",
    "/home/dragon/hunmin/data/opus_en_zh.tsv:en:zh:250000",
    "/home/dragon/hunmin/data/opus_en_fr.tsv:en:fr:250000",
    "/home/dragon/hunmin/data/opus_de_en.tsv:de:en:250000",
    "/home/dragon/hunmin/data/opus_en_es.tsv:en:es:250000",
]


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    left_lang: str
    right_lang: str
    max_rows: int
    skip_lines: int = 0

    @property
    def other_lang(self) -> str:
        if self.left_lang == "en" and self.right_lang != "en":
            return self.right_lang
        if self.right_lang == "en" and self.left_lang != "en":
            return self.left_lang
        raise ValueError(f"source must contain exactly one English side: {self}")

    @property
    def name(self) -> str:
        suffix = f"_skip{self.skip_lines}" if self.skip_lines else ""
        return f"{self.path.stem}_{self.left_lang}_{self.right_lang}{suffix}"


def parse_source(value: str, default_max_rows: int) -> SourceSpec:
    parts = value.split(":")
    if len(parts) not in {3, 4, 5}:
        raise argparse.ArgumentTypeError("source must be path:left_lang:right_lang[:max_rows[:skip_lines]]")
    path = Path(parts[0])
    left_lang = parts[1].lower()
    right_lang = parts[2].lower()
    if left_lang not in LANG_TAGS or right_lang not in LANG_TAGS:
        raise argparse.ArgumentTypeError(f"unsupported langs: {left_lang}, {right_lang}")
    if (left_lang == "en") == (right_lang == "en"):
        raise argparse.ArgumentTypeError("source must have exactly one en side")
    max_rows = int(parts[3]) if len(parts) >= 4 else default_max_rows
    skip_lines = int(parts[4]) if len(parts) == 5 else 0
    return SourceSpec(path=path, left_lang=left_lang, right_lang=right_lang, max_rows=max_rows, skip_lines=skip_lines)


def normalize_text(text: str) -> str:
    text = text.replace("\ufeff", "").strip()
    text = SPACE_RE.sub(" ", text)
    return text


def dedupe_norm(text: str) -> str:
    return SPACE_RE.sub(" ", text.casefold().strip())


def text_ok(text: str, lang: str, min_chars: int, max_chars: int, strict_script: bool) -> tuple[bool, str]:
    if len(text) < min_chars:
        return False, "too_short"
    if len(text) > max_chars:
        return False, "too_long"
    if "\t" in text or "\n" in text or CONTROL_RE.search(text):
        return False, "control_or_tab"
    if URL_RE.search(text):
        return False, "url"
    if HTML_RE.search(text):
        return False, "html"
    if strict_script and not SCRIPT_RE[lang].search(text):
        return False, "missing_lang_script"
    return True, ""


def length_ratio_ok(en_text: str, other_text: str, max_ratio: float) -> bool:
    left = max(1, len(en_text))
    right = max(1, len(other_text))
    return max(left, right) / min(left, right) <= max_ratio


def latin_jaccard(a: str, b: str) -> float:
    aw = {w.casefold() for w in LATIN_WORD_RE.findall(a) if len(w) > 1}
    bw = {w.casefold() for w in LATIN_WORD_RE.findall(b) if len(w) > 1}
    if not aw or not bw:
        return 0.0
    return len(aw & bw) / len(aw | bw)


def split_for_english(en_text: str, val_ppm: int, test_ppm: int) -> str:
    digest = hashlib.blake2b(dedupe_norm(en_text).encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 1_000_000
    if bucket < test_ppm:
        return "test"
    if bucket < test_ppm + val_ppm:
        return "val"
    return "train"


def stable_id(source_name: str, en_text: str, other_lang: str, other_text: str) -> str:
    raw = f"{source_name}\t{dedupe_norm(en_text)}\t{other_lang}\t{dedupe_norm(other_text)}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()


_PUBLIC_TRANSCRIBE = None
_PUBLIC_VERSION = None


def public_transcribe():
    global _PUBLIC_TRANSCRIBE, _PUBLIC_VERSION
    if _PUBLIC_TRANSCRIBE is None:
        try:
            import importlib.metadata as md
            from hunmin import transcribe
        except Exception as exc:
            raise RuntimeError(
                "Auxiliary Hunmin/UHPS views require public hunmin. "
                "Install with: pip install 'hunmin[cjk]==2.4.4'"
            ) from exc
        _PUBLIC_TRANSCRIBE = transcribe
        try:
            _PUBLIC_VERSION = md.version("hunmin")
        except Exception:
            _PUBLIC_VERSION = "unknown"
    return _PUBLIC_TRANSCRIBE, _PUBLIC_VERSION


def maybe_transcribe(text: str, lang: str, level: int) -> str:
    transcribe, _version = public_transcribe()
    return transcribe(text, lang, level=level)


def positive_pairs(aux: str, include_phonetic_cross: bool, text_pair_weight: int) -> list[list[str]]:
    text_pairs = [["text_a", "text_b"], ["text_b", "text_a"]]
    pairs = text_pairs * max(1, text_pair_weight)
    if aux in {"hunmin", "both"}:
        pairs.extend([
            ["text_a", "hunmin_a"],
            ["hunmin_a", "text_a"],
            ["text_b", "hunmin_b"],
            ["hunmin_b", "text_b"],
        ])
        if include_phonetic_cross:
            pairs.extend([["hunmin_a", "hunmin_b"], ["hunmin_b", "hunmin_a"]])
    if aux in {"uhps", "both"}:
        pairs.extend([
            ["text_a", "uhps_a"],
            ["uhps_a", "text_a"],
            ["text_b", "uhps_b"],
            ["uhps_b", "text_b"],
        ])
        if include_phonetic_cross:
            pairs.extend([["uhps_a", "uhps_b"], ["uhps_b", "uhps_a"]])
    if aux == "both":
        pairs.extend([
            ["hunmin_a", "uhps_a"],
            ["uhps_a", "hunmin_a"],
            ["hunmin_b", "uhps_b"],
            ["uhps_b", "hunmin_b"],
        ])
    return pairs


def make_record(
    *,
    source_name: str,
    source_line: int,
    en_text: str,
    other_lang: str,
    other_text: str,
    split: str,
    aux: str,
    include_phonetic_cross: bool,
    text_pair_weight: int,
) -> dict:
    row_id = stable_id(source_name, en_text, other_lang, other_text)
    record = {
        "id": f"enhub_{row_id}",
        "pair_id": f"enhub_{row_id}",
        "source": source_name,
        "source_line": source_line,
        "split": split,
        "hub_lang": "en",
        "lang_a": "en",
        "lang_b": other_lang,
        "text_a": f"[EN] {en_text}",
        "text_b": f"{LANG_TAGS[other_lang]} {other_text}",
        "positive_pairs": positive_pairs(aux, include_phonetic_cross, text_pair_weight),
    }
    if other_lang == "ko":
        record["meaning"] = f"[KO] {other_text}"
    if aux in {"hunmin", "both"}:
        record["hunmin_a"] = f"[HUNMIN] {maybe_transcribe(en_text, 'en', 1)}"
        record["hunmin_b"] = f"[HUNMIN] {maybe_transcribe(other_text, other_lang, 1)}"
    if aux in {"uhps", "both"}:
        record["uhps_a"] = f"[UHPS] {maybe_transcribe(en_text, 'en', 4)}"
        record["uhps_b"] = f"[UHPS] {maybe_transcribe(other_text, other_lang, 4)}"
    return record


def open_outputs(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)
    return {
        "train": (output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }


def close_outputs(handles: dict[str, object]) -> None:
    for handle in handles.values():
        handle.close()


def write_row(handles: dict[str, object], split: str, row: dict) -> None:
    handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")


def iter_source_lines(spec: SourceSpec) -> Iterable[tuple[int, str, str]]:
    with spec.path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= spec.skip_lines:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            yield line_no, normalize_text(parts[0]), normalize_text(parts[1])


def should_stop_language(counts_by_lang_split: dict[str, Counter], lang: str, args) -> bool:
    limits = {
        "train": args.per_lang_train_max,
        "val": args.per_lang_val_max,
        "test": args.per_lang_test_max,
    }
    current = counts_by_lang_split[lang]
    return all(current[split] >= limit for split, limit in limits.items())


def prepare(sources: list[SourceSpec], args) -> dict:
    rng = random.Random(args.seed)
    handles = open_outputs(args.output_dir)
    stats: dict = {
        "dataset": "hunmin_enhub_sentence_v1",
        "description": "EN-hub sentence-pair encoder dataset. text-text is the main objective.",
        "sources": [asdict(src) | {"path": str(src.path), "name": src.name, "other_lang": src.other_lang} for src in sources],
        "args": vars(args) | {"output_dir": str(args.output_dir), "source": args.source},
        "counts": {
            "by_split": Counter(),
            "by_lang": Counter(),
            "by_lang_split": defaultdict(Counter),
            "by_source": Counter(),
            "skip": Counter(),
        },
        "samples": [],
    }
    seen_pairs: set[tuple[str, str, str]] = set()
    seen_en_by_lang: set[tuple[str, str]] = set()
    try:
        ordered_sources = list(sources)
        rng.shuffle(ordered_sources)
        for spec in ordered_sources:
            source_counts: Counter = Counter()
            if not spec.path.exists():
                source_counts["missing"] += 1
                stats["counts"]["skip"][f"{spec.name}:missing"] += 1
                continue
            other_lang = spec.other_lang
            if should_stop_language(stats["counts"]["by_lang_split"], other_lang, args):
                source_counts["language_already_full"] += 1
                continue
            for line_no, left_text, right_text in iter_source_lines(spec):
                source_counts["seen"] += 1
                if source_counts["accepted"] >= spec.max_rows:
                    break
                if should_stop_language(stats["counts"]["by_lang_split"], other_lang, args):
                    source_counts["language_full"] += 1
                    break
                en_text = left_text if spec.left_lang == "en" else right_text
                other_text = right_text if spec.left_lang == "en" else left_text
                if not en_text or not other_text:
                    source_counts["skip_blank"] += 1
                    continue
                if dedupe_norm(en_text) == dedupe_norm(other_text):
                    source_counts["skip_identical"] += 1
                    continue
                ok_en, reason_en = text_ok(en_text, "en", args.min_chars, args.max_chars, args.strict_script)
                ok_other, reason_other = text_ok(other_text, other_lang, args.min_chars, args.max_chars, args.strict_script)
                if not ok_en:
                    source_counts[f"skip_en_{reason_en}"] += 1
                    continue
                if not ok_other:
                    source_counts[f"skip_other_{reason_other}"] += 1
                    continue
                if not length_ratio_ok(en_text, other_text, args.max_length_ratio):
                    source_counts["skip_length_ratio"] += 1
                    continue
                if other_lang in {"fr", "de", "es"} and latin_jaccard(en_text, other_text) >= args.max_latin_jaccard:
                    source_counts["skip_latin_overlap"] += 1
                    continue
                en_key = dedupe_norm(en_text)
                other_key = dedupe_norm(other_text)
                if args.one_translation_per_en and (other_lang, en_key) in seen_en_by_lang:
                    source_counts["skip_duplicate_en_for_lang"] += 1
                    continue
                pair_key = (other_lang, en_key, other_key)
                if pair_key in seen_pairs:
                    source_counts["skip_duplicate_pair"] += 1
                    continue
                split = split_for_english(en_text, args.val_ppm, args.test_ppm)
                split_limit = {
                    "train": args.per_lang_train_max,
                    "val": args.per_lang_val_max,
                    "test": args.per_lang_test_max,
                }[split]
                if stats["counts"]["by_lang_split"][other_lang][split] >= split_limit:
                    source_counts[f"skip_{split}_full"] += 1
                    continue
                try:
                    row = make_record(
                        source_name=spec.name,
                        source_line=line_no,
                        en_text=en_text,
                        other_lang=other_lang,
                        other_text=other_text,
                        split=split,
                        aux=args.aux,
                        include_phonetic_cross=args.include_phonetic_cross,
                        text_pair_weight=args.text_pair_weight,
                    )
                except Exception:
                    source_counts["skip_aux_transcribe_error"] += 1
                    continue
                write_row(handles, split, row)
                seen_pairs.add(pair_key)
                seen_en_by_lang.add((other_lang, en_key))
                source_counts["accepted"] += 1
                stats["counts"]["by_split"][split] += 1
                stats["counts"]["by_lang"][other_lang] += 1
                stats["counts"]["by_lang_split"][other_lang][split] += 1
                stats["counts"]["by_source"][spec.name] += 1
                if len(stats["samples"]) < args.sample_rows:
                    stats["samples"].append(row)
            stats["counts"]["skip"].update({f"{spec.name}:{k}": v for k, v in source_counts.items() if k.startswith("skip_") or k in {"missing", "language_full", "language_already_full"}})
    finally:
        close_outputs(handles)

    if _PUBLIC_VERSION:
        stats["hunmin_public_version"] = _PUBLIC_VERSION
    stats["counts"]["by_split"] = dict(stats["counts"]["by_split"])
    stats["counts"]["by_lang"] = dict(stats["counts"]["by_lang"])
    stats["counts"]["by_source"] = dict(stats["counts"]["by_source"])
    stats["counts"]["skip"] = dict(stats["counts"]["skip"])
    stats["counts"]["by_lang_split"] = {lang: dict(counter) for lang, counter in stats["counts"]["by_lang_split"].items()}
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare EN-hub sentence-pair data for Hunmin encoders.")
    ap.add_argument("--source", action="append", help="path:left_lang:right_lang[:max_rows[:skip_lines]]. Defaults to local d2 OPUS sources.")
    ap.add_argument("--output-dir", type=Path, default=Path("output/enhub_sentence_v1"))
    ap.add_argument("--default-max-rows", type=int, default=250_000)
    ap.add_argument("--per-lang-train-max", type=int, default=200_000)
    ap.add_argument("--per-lang-val-max", type=int, default=20_000)
    ap.add_argument("--per-lang-test-max", type=int, default=20_000)
    ap.add_argument("--val-ppm", type=int, default=50_000, help="Validation split parts per million by English hash.")
    ap.add_argument("--test-ppm", type=int, default=50_000, help="Test split parts per million by English hash.")
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=240)
    ap.add_argument("--max-length-ratio", type=float, default=3.5)
    ap.add_argument("--max-latin-jaccard", type=float, default=0.88)
    ap.add_argument("--strict-script", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--one-translation-per-en", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--aux", choices=["none", "hunmin", "uhps", "both"], default="none")
    ap.add_argument("--include-phonetic-cross", action="store_true")
    ap.add_argument(
        "--text-pair-weight",
        type=int,
        default=1,
        help="Repeat text<->text positives so auxiliary views do not dilute the main cross-lingual objective.",
    )
    ap.add_argument("--sample-rows", type=int, default=20)
    ap.add_argument("--seed", type=int, default=20260501)
    args = ap.parse_args()

    raw_sources = args.source or DEFAULT_SOURCES
    sources = [parse_source(value, args.default_max_rows) for value in raw_sources]
    stats = prepare(sources, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output_dir": str(args.output_dir),
        "counts": stats["counts"],
        "aux": args.aux,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
