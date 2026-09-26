#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-2}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
for attempt in $(seq 1 240); do
  if [[ -f work/qwen_score_smoke_a6000/train_India/complete.json &&
        -f work/qwen_score_smoke_a6000/train_US/complete.json ]]; then
    break
  fi
  sleep 15
done
test -f work/qwen_score_smoke_a6000/train_India/complete.json
test -f work/qwen_score_smoke_a6000/train_US/complete.json
.venv/bin/python -u src/score_qwen_groups.py --data work/data \
  --retrieval work/full_retrieval --owner-model work/full_owner \
  --base-model work/qwen_base --adapter work/qwen_fit/adapter \
  --output work/qwen_tune --batch-size 32 \
  --route-min .2 --route-max .8 \
  --route-close-min 1 --route-close-margin 0 \
  --restrict-roles tune
