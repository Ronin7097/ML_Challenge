#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-2}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
if [[ ! -f work/mined_direct/complete.json ]]; then
.venv/bin/python -u src/mine_hard_negatives.py --data work/data --vectors work/pilot_vectors \
  --model work/pilot_encoder/model --output work/mined_direct
fi
if [[ ! -f work/hn1_encoder/complete.json ]]; then
.venv/bin/python -u src/train_mined_encoder.py --pairs work/mined_direct/pairs.parquet \
  --base-model work/pilot_encoder/model --output work/hn1_encoder
fi
.venv/bin/python -u src/encode.py --data work/data --model work/hn1_encoder/model \
  --output work/hn1_vectors --glob 'train_s1_*.parquet' --batch-size 512
if [[ ! -f work/hn1_probe/report.json ]]; then
.venv/bin/python -u src/probe_retrieval.py --data work/data --vectors work/hn1_vectors \
  --model work/hn1_encoder/model --output work/hn1_probe
fi
