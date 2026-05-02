#!/usr/bin/env bash
set -euo pipefail

cd /home/dragon/hunmin_v1

RUN_NAME="hunmin-lite-13m-7lang-enhub-textonly-simple-40k-run2"
LOG_DIR="output/logs"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d).log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

mkdir -p "${LOG_DIR}"

nohup .venv_run7/bin/python scripts/train_hunmin_m12_canonical.py \
  --config configs/hunmin_lite_13m_7lang_enhub_textonly_simple_40k_run2.json \
  --train output/enhub_sentence_v1_text_only_6lang/train.shuffled.jsonl \
  --val output/enhub_sentence_v1_text_only_6lang/val.shuffled.jsonl \
  --output-dir "models/${RUN_NAME}" \
  > "${LOG_PATH}" 2>&1 &

echo $! > "${PID_PATH}"
echo "started ${RUN_NAME}"
echo "pid=$(cat "${PID_PATH}")"
echo "log=${LOG_PATH}"
