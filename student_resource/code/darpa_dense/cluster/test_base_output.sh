#!/usr/bin/env bash
set -euo pipefail
umask 077
export OMP_NUM_THREADS=6
for attempt in $(seq 1 2880); do
  if [[ -f work/full_test_features/test_France/complete.json &&
        -f work/full_test_features/test_India/complete.json &&
        -f work/full_test_features/test_US/complete.json ]]; then
    break
  fi
  sleep 15
done
for country in France India US; do
  test -f "work/full_test_features/test_${country}/complete.json"
done
.venv/bin/python -u src/predict_frozen.py --data work/data \
  --features work/full_test_features --retrieval work/full_test_retrieval \
  --model work/full_owner --threshold .62 \
  --source1-tsv ../student_resource/dataset/test/test_source1.tsv \
  --output work/full_test_base_output \
  --routing-output work/full_test_base_output/routing \
  --split test
test -f work/full_test_base_output/run.json
