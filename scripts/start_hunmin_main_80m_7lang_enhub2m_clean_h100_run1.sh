#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_NAME="hunmin-main-80m-7lang-enhub2m-clean-h100-run1"
PYTHON_BIN="${PYTHON_BIN:-python3}"
LOG_DIR="output/logs"
LOG_PATH="${LOG_DIR}/${RUN_NAME}_$(date +%Y%m%d_%H%M%S).log"
PID_PATH="${LOG_DIR}/${RUN_NAME}.pid"

mkdir -p "${LOG_DIR}" "models/${RUN_NAME}"

nohup "${PYTHON_BIN}" scripts/train_hunmin_m12_canonical.py \
  --config configs/hunmin_main_80m_7lang_enhub2m_clean_h100_run1.json \
  --train output/enhub_sentence_v2_text_only_6lang_2m_clean/train.shuffled.jsonl \
  --val output/enhub_sentence_v2_text_only_6lang_2m_clean/val.shuffled.jsonl \
  --output-dir "models/${RUN_NAME}" \
  > "${LOG_PATH}" 2>&1 &

echo $! > "${PID_PATH}"
echo "started ${RUN_NAME}"
echo "pid=$(cat "${PID_PATH}")"
echo "log=${LOG_PATH}"
