# Hunmin Encoder H100 Upload

Purpose: run encoder-only BGE-replacement experiments on an H100 without the search layer.

## Data

Uses the simple 12M-style EN-hub sentence-pair objective:

```text
output/enhub_sentence_v1_text_only_6lang/train.shuffled.jsonl
output/enhub_sentence_v1_text_only_6lang/val.shuffled.jsonl
```

Languages:

```text
en, ko, ja, zh, fr, de, es
```

Training rows:

```text
train: 1,200,000
val:   120,000
```

## Recommended First Run

Start with the 80M-class main encoder:

```bash
cd /root/hunmin_v1
PYTHON_BIN=/path/to/python scripts/start_hunmin_main_80m_7lang_enhub_textonly_h100_run1.sh
```

If the system Python already has PyTorch:

```bash
cd /root/hunmin_v1
scripts/start_hunmin_main_80m_7lang_enhub_textonly_h100_run1.sh
```

Monitor:

```bash
cd /root/hunmin_v1
tail -f output/logs/hunmin-main-80m-7lang-enhub-textonly-h100-run1_$(date +%Y%m%d).log
nvidia-smi
```

## Optional Larger Run

After the 80M run is confirmed:

```bash
cd /root/hunmin_v1
scripts/start_hunmin_main_155m_7lang_enhub_textonly_h100_run1.sh
```

## Current Local Baseline

On d2 RTX 4090:

```text
hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2
cross R@1: 0.57975 / 0.58133
cross R@5: 0.72258 / 0.72125
```

Goal for H100 80M:

```text
cross R@1 >= 0.70 first
then scale toward 0.80+
```

Do not modify scribe, tokenizer, corpus canonical, or search-layer code for this run.
