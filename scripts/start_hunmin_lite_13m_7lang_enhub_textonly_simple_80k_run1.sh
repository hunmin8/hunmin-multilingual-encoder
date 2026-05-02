#!/bin/bash
set -euo pipefail
cd /home/dragon/hunmin_v1
mkdir -p output/logs
python3 scripts/train_hunmin_m12_canonical.py \
  --config configs/hunmin_lite_13m_7lang_enhub_textonly_simple_80k_run1.json \
  --train output/enhub_sentence_v1_text_only_6lang/train.shuffled.jsonl \
  --val output/enhub_sentence_v1_text_only_6lang/val.shuffled.jsonl \
  --output-dir models/hunmin-lite-13m-7lang-enhub-textonly-simple-80k-run1 \
  --device cuda \
  2>&1 | tee output/logs/hunmin-lite-13m-7lang-enhub-textonly-simple-80k-run1.train.log
