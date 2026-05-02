#!/usr/bin/env python3
"""Prepare direct non-English language-pair rows for encoder training.

This is intentionally simple and encoder-only:

    [KO] sentence <-> [JA] sentence
    [KO] sentence <-> [ZH] sentence
    [JA] sentence <-> [DE] sentence

It does not use Hunmin transcription, UHPS, BPE, scribe rules, or search-layer
clusters. The output JSONL format matches train_hunmin_m12_canonical.py.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path


LANG_TAGS = {
    "ko": "[KO]",
    "ja": "[JA]",
    "zh": "[ZH]",
    "en": "[EN]",
    "fr": "[FR]",
    "de": "[DE]",
    "es": "[ES]",
}

SCRIPT_RE = {
    "ko": re.compile(r"[\uac00-\ud7a3]"),
    "ja": re.compile(r"[\u3040-\u30ff\u3400-\u9fff]"),
    "zh": re.compile(r"[\u3400-\u9fff]"),
    "en": re.compile(r"[A-Za-z]"),
    "fr": re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]"),
    "de": re.compile(r"[A-Za-zÄÖÜäöüß]"),
    "es": re.compile(r"[A-Za-zÁÉÍÓÚÜÑáéíóúüñ]"),
}

SPACE_RE = re.compile(r"\s+")
URL_RE = re.compile(r"https?://|www\.", re.I)
HTML_RE = re.compile(r"&[a-z]+;|<[^>]+>", re.I)
CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f]")


DEFAULT_SOURCES = [
    # Higher-confidence direct CJK subtitles / OPUS.
    "/home/dragon/hunmin/data/opensub_ja_ko.tsv:ja:ko:250000",
    "/home/dragon/hunmin/data/opensub_zh_ko.tsv:ko:zh:250000",
    "/home/dragon/hunmin/data/opus_ja_ko.tsv:ja:ko:100000",
    # WikiMatrix direct CJK. Keep filtered and capped because it is noisier.
    "/home/dragon/hunmin/data/wikimatrix_ja_ko.tsv:ja:ko:300000",
    "/home/dragon/hunmin/data/wikimatrix_ko_zh.tsv:ko:zh:300000",
    "/home/dragon/hunmin/data/wikimatrix_ja_zh.tsv:ja:zh:300000",
    # Direct bridges from weak CJK languages to strong European languages.
    # These WikiMatrix files are often ordered as European first, CJK second.
    "/home/dragon/hunmin/data/wikimatrix_ja_de.tsv:de:ja:150000",
    "/home/dragon/hunmin/data/wikimatrix_ja_fr.tsv:fr:ja:150000",
    "/home/dragon/hunmin/data/wikimatrix_ja_es.tsv:es:ja:150000",
    "/home/dragon/hunmin/data/wikimatrix_ko_de.tsv:de:ko:150000",
    "/home/dragon/hunmin/data/wikimatrix_ko_fr.tsv:fr:ko:150000",
    "/home/dragon/hunmin/data/wikimatrix_ko_es.tsv:es:ko:150000",
]


@dataclass(frozen=True)
class SourceSpec:
    path: Path
    left_lang: str
    right_lang: str
    max_rows: int
    skip_lines: int = 0

    @property
    def name(self) -> str:
        suffix = f"_skip{self.skip_lines}" if self.skip_lines else ""
        return f"{self.path.stem}_{self.left_lang}_{self.right_lang}{suffix}"


def parse_source(value: str) -> SourceSpec:
    parts = value.split(":")
    if len(parts) not in {4, 5}:
        raise argparse.ArgumentTypeError("source must be path:left_lang:right_lang:max_rows[:skip_lines]")
    left_lang = parts[1].lower()
    right_lang = parts[2].lower()
    if left_lang not in LANG_TAGS or right_lang not in LANG_TAGS:
        raise argparse.ArgumentTypeError(f"unsupported source langs: {left_lang}, {right_lang}")
    return SourceSpec(
        path=Path(parts[0]),
        left_lang=left_lang,
        right_lang=right_lang,
        max_rows=int(parts[3]),
        skip_lines=int(parts[4]) if len(parts) == 5 else 0,
    )


def normalize_text(text: str) -> str:
    return SPACE_RE.sub(" ", text.replace("\ufeff", "").strip())


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


def length_ratio_ok(left: str, right: str, max_ratio: float) -> bool:
    a = max(1, len(left))
    b = max(1, len(right))
    return max(a, b) / min(a, b) <= max_ratio


def split_for_pair(source_name: str, left: str, right: str, val_ppm: int, test_ppm: int) -> str:
    raw = f"{source_name}\t{dedupe_norm(left)}\t{dedupe_norm(right)}"
    digest = hashlib.blake2b(raw.encode("utf-8"), digest_size=8).digest()
    bucket = int.from_bytes(digest, "big") % 1_000_000
    if bucket < test_ppm:
        return "test"
    if bucket < test_ppm + val_ppm:
        return "val"
    return "train"


def stable_id(source_name: str, line_no: int, left_lang: str, left: str, right_lang: str, right: str) -> str:
    raw = f"{source_name}\t{line_no}\t{left_lang}\t{dedupe_norm(left)}\t{right_lang}\t{dedupe_norm(right)}"
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()


def make_record(spec: SourceSpec, line_no: int, left: str, right: str, split: str) -> dict:
    row_id = stable_id(spec.name, line_no, spec.left_lang, left, spec.right_lang, right)
    return {
        "id": f"direct_{row_id}",
        "pair_id": f"direct_{row_id}",
        "record_kind": "direct_language_pair",
        "source": spec.name,
        "source_line": line_no,
        "split": split,
        "lang_a": spec.left_lang,
        "lang_b": spec.right_lang,
        "text_a": f"{LANG_TAGS[spec.left_lang]} {left}",
        "text_b": f"{LANG_TAGS[spec.right_lang]} {right}",
        "positive_pairs": [["text_a", "text_b"], ["text_b", "text_a"]],
    }


def iter_source(spec: SourceSpec):
    with spec.path.open("r", encoding="utf-8", errors="replace") as f:
        for line_no, line in enumerate(f, start=1):
            if line_no <= spec.skip_lines:
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            yield line_no, normalize_text(parts[0]), normalize_text(parts[1])


def prepare(sources: list[SourceSpec], args) -> dict:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    handles = {
        "train": (args.output_dir / "train.jsonl").open("w", encoding="utf-8"),
        "val": (args.output_dir / "val.jsonl").open("w", encoding="utf-8"),
        "test": (args.output_dir / "test.jsonl").open("w", encoding="utf-8"),
    }
    stats = {
        "dataset": "direct_pair_v1_weak_bridge",
        "description": "Filtered direct language-pair positives for weak cross-lingual bridge training.",
        "sources": [asdict(src) | {"path": str(src.path), "name": src.name} for src in sources],
        "args": vars(args) | {"output_dir": str(args.output_dir), "source": args.source},
        "counts": {
            "by_split": Counter(),
            "by_lang_pair": Counter(),
            "by_lang_pair_split": defaultdict(Counter),
            "by_source": Counter(),
            "skip": Counter(),
        },
        "samples": [],
    }
    seen: set[tuple[str, str, str, str]] = set()
    try:
        for spec in sources:
            source_counts: Counter = Counter()
            if not spec.path.exists():
                source_counts["missing"] += 1
                stats["counts"]["skip"][f"{spec.name}:missing"] += 1
                continue
            lang_pair = f"{spec.left_lang}-{spec.right_lang}"
            for line_no, left, right in iter_source(spec):
                source_counts["seen"] += 1
                if source_counts["accepted"] >= spec.max_rows:
                    break
                if not left or not right:
                    source_counts["skip_blank"] += 1
                    continue
                if dedupe_norm(left) == dedupe_norm(right):
                    source_counts["skip_identical"] += 1
                    continue
                ok_left, reason_left = text_ok(left, spec.left_lang, args.min_chars, args.max_chars, args.strict_script)
                ok_right, reason_right = text_ok(right, spec.right_lang, args.min_chars, args.max_chars, args.strict_script)
                if not ok_left:
                    source_counts[f"skip_left_{reason_left}"] += 1
                    continue
                if not ok_right:
                    source_counts[f"skip_right_{reason_right}"] += 1
                    continue
                if not length_ratio_ok(left, right, args.max_length_ratio):
                    source_counts["skip_length_ratio"] += 1
                    continue
                pair_key = (spec.left_lang, dedupe_norm(left), spec.right_lang, dedupe_norm(right))
                reverse_key = (spec.right_lang, dedupe_norm(right), spec.left_lang, dedupe_norm(left))
                if pair_key in seen or reverse_key in seen:
                    source_counts["skip_duplicate_pair"] += 1
                    continue
                split = split_for_pair(spec.name, left, right, args.val_ppm, args.test_ppm)
                row = make_record(spec, line_no, left, right, split)
                handles[split].write(json.dumps(row, ensure_ascii=False) + "\n")
                seen.add(pair_key)
                source_counts["accepted"] += 1
                stats["counts"]["by_split"][split] += 1
                stats["counts"]["by_lang_pair"][lang_pair] += 1
                stats["counts"]["by_lang_pair_split"][lang_pair][split] += 1
                stats["counts"]["by_source"][spec.name] += 1
                if len(stats["samples"]) < args.sample_rows:
                    stats["samples"].append(row)
            stats["counts"]["skip"].update(
                {f"{spec.name}:{k}": v for k, v in source_counts.items() if k.startswith("skip_") or k == "missing"}
            )
    finally:
        for handle in handles.values():
            handle.close()

    stats["counts"]["by_split"] = dict(stats["counts"]["by_split"])
    stats["counts"]["by_lang_pair"] = dict(stats["counts"]["by_lang_pair"])
    stats["counts"]["by_source"] = dict(stats["counts"]["by_source"])
    stats["counts"]["skip"] = dict(stats["counts"]["skip"])
    stats["counts"]["by_lang_pair_split"] = {
        key: dict(value) for key, value in stats["counts"]["by_lang_pair_split"].items()
    }
    return stats


def main() -> None:
    ap = argparse.ArgumentParser(description="Prepare direct language-pair encoder data.")
    ap.add_argument("--source", action="append", help="path:left_lang:right_lang:max_rows[:skip_lines]")
    ap.add_argument("--output-dir", type=Path, default=Path("output/direct_pair_v1_weak_bridge"))
    ap.add_argument("--min-chars", type=int, default=4)
    ap.add_argument("--max-chars", type=int, default=240)
    ap.add_argument("--max-length-ratio", type=float, default=3.5)
    ap.add_argument("--val-ppm", type=int, default=30000)
    ap.add_argument("--test-ppm", type=int, default=30000)
    ap.add_argument("--strict-script", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--sample-rows", type=int, default=30)
    args = ap.parse_args()

    sources = [parse_source(src) for src in (args.source or DEFAULT_SOURCES)]
    stats = prepare(sources, args)
    (args.output_dir / "stats.json").write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(args.output_dir), "counts": stats["counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
