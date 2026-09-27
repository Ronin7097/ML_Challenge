#!/usr/bin/env bash
set -euo pipefail
umask 077
export OMP_NUM_THREADS=6
for attempt in $(seq 1 480); do
  if [[ -f work/qwen_tune/train_India/complete.json &&
        -f work/qwen_tune/train_US/complete.json ]]; then
    break
  fi
  sleep 15
done
test -f work/qwen_tune/train_India/complete.json
test -f work/qwen_tune/train_US/complete.json
.venv/bin/python -u src/qwen_policy_tune.py --data work/data \
  --owner-model work/full_owner --qwen-scores work/qwen_tune \
  --output work/qwen_policy_tune --base-threshold .62
.venv/bin/python -u src/evaluate_tune.py --data work/data \
  --prediction work/qwen_policy_tune/tune_matching_results.tsv \
  --baseline work/baseline/tune/matching_results.tsv \
  --output work/qwen_policy_tune/paired_report.json
