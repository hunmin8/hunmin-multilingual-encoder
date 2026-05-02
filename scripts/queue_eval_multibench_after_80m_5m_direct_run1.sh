#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

RUN="hunmin-main-80m-7lang-enhub5m-direct1m-textonly-h100-run1"
PIDFILE="output/logs/${RUN}.pid"
LOG="output/logs/eval-multibench-after-${RUN}_$(date +%Y%m%d).log"
mkdir -p output/logs

if [ -f "$PIDFILE" ]; then
  PID="$(cat "$PIDFILE" 2>/dev/null || true)"
  while [ -n "$PID" ] && ps -p "$PID" >/dev/null 2>&1; do
    echo "$(date -Is) waiting for $RUN pid $PID" | tee -a "$LOG"
    sleep 120
  done
fi

OUT_DIR="output/eval_multibench_$(date +%Y%m%d)_80m_5m_direct"
mkdir -p "$OUT_DIR"

CKPT="models/${RUN}/best.pt"
python3 scripts/eval_m12_multibench.py \
  --checkpoint "$CKPT" \
  --benchmark internal_5m_direct_val=output/enhub_sentence_v1_text_only_6lang_5m_plus_direct_v1/val.jsonl \
  --benchmark internal_1m2_val=output/enhub_sentence_v1_text_only_6lang/val.shuffled.jsonl \
  --output "$OUT_DIR/${RUN}.json" \
  --overall-limit 12000 \
  --group-limit 3000 \
  --batch-size 512 \
  --failure-samples 20 \
  --device cuda \
  >> "$LOG" 2>&1

python3 - <<PY | tee -a "$LOG"
import json
from pathlib import Path

p = Path("$OUT_DIR/${RUN}.json")
d = json.loads(p.read_text())
summary = {
  "output": str(p),
  "model": d["model"],
  "benchmarks": {
    k: {
      "r1": v["overall"]["mean_recall_at_1"],
      "r5": v["overall"]["mean_recall_at_5"],
      "mrr": v["overall"]["mean_mrr"],
    }
    for k, v in d["benchmarks"].items()
  },
}
Path("$OUT_DIR/summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\\n")
print(json.dumps(summary, ensure_ascii=False, indent=2))
PY
