#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-2}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
mkdir -p "$TMPDIR"
for attempt in $(seq 1 480); do
  if [[ -f work/qwen_tune/train_India/complete.json &&
        -f work/qwen_tune/train_US/complete.json ]]; then
    break
  fi
  sleep 15
done
test -f work/qwen_tune/train_India/complete.json
test -f work/qwen_tune/train_US/complete.json
test -f work/full_encoder/model/model.safetensors
expected=192974ed0c8ed02ee2c48dad6dbf3d7447143b18fe7cd91d2bb8b5a106b931bd
actual=$(sha256sum work/full_encoder/model/model.safetensors | cut -d' ' -f1)
[[ "$actual" == "$expected" ]]
.venv/bin/python -u src/encode.py --data work/data \
  --model work/full_encoder/model --output work/full_test_vectors \
  --glob 'test_s1_*.parquet' --batch-size 512
.venv/bin/python -u src/retrieve_owners.py --data work/data \
  --vectors work/full_test_vectors --model work/full_encoder/model \
  --output work/full_test_retrieval --split test --batch-size 512 --k 10
.venv/bin/python -u src/build_features.py --data work/data \
  --retrieval work/full_test_retrieval --output work/full_test_features \
  --split test --workers 6
