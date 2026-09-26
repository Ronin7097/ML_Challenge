#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-0}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
.venv/bin/python src/test_gradient_cache.py
if [[ ! -f work/pilot_smoke/complete.json ]]; then
.venv/bin/python -u src/train_encoder.py --pairs work/data/encoder_pairs.parquet \
  --base-model work/base_model --output work/pilot_smoke --batch-size 32 --microbatch 8 --max-steps 2
fi
if [[ ! -f work/pilot_encoder/complete.json ]]; then
.venv/bin/python -u src/train_encoder.py --pairs work/data/encoder_pairs.parquet \
  --base-model work/base_model --output work/pilot_encoder --batch-size 512 --microbatch 64 --max-steps 300 --resume
fi
.venv/bin/python -u src/encode.py --data work/data --model work/pilot_encoder/model \
  --output work/pilot_vectors --glob 'train_s1_*.parquet' --batch-size 512
if [[ ! -f work/pilot_probe/report.json ]]; then
.venv/bin/python -u src/probe_retrieval.py --data work/data --vectors work/pilot_vectors \
  --model work/pilot_encoder/model --output work/pilot_probe
fi
