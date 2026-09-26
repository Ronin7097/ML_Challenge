#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-2}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
for attempt in $(seq 1 240); do
  [[ -f work/hn1_probe/report.json ]] && break
  sleep 15
done
test -f work/hn1_probe/report.json
test -f work/mined_direct/complete.json
if [[ ! -f work/qwen_fit/complete.json ]]; then
.venv/bin/python -u src/train_qwen.py --pairs work/mined_direct/pairs.parquet \
  --base-model work/qwen_base --output work/qwen_fit
fi
