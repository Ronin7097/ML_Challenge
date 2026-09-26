#!/usr/bin/env bash
set -euo pipefail
umask 077
export CUDA_VISIBLE_DEVICES=${DARPA_CUDA_DEVICE:-2}
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
mkdir -p "$TMPDIR"
for attempt in $(seq 1 180); do
  [[ -f work/pilot_probe/report.json ]] && break
  sleep 15
done
.venv/bin/python - <<'CHECK'
import json
from pathlib import Path
r=json.loads(Path('work/pilot_probe/report.json').read_text())
if r['recall']['10']<.99:
    raise RuntimeError('Pilot top-10 recall below 99%; review before full candidate processing')
print('Tune retrieval gate passed; no F0.5 claim',r['recall'],flush=True)
CHECK
.venv/bin/python -u src/retrieve_owners.py --data work/data --vectors work/pilot_vectors \
  --model work/pilot_encoder/model --output work/control_retrieval --country India --max-rows 1024
.venv/bin/python - <<'CHECK'
import json,numpy as np
from pathlib import Path
p=Path('work/control_retrieval/train_India')
r=json.loads((p/'progress.json').read_text());n=r['rows_done']
i=np.load(p/'indices.npy',mmap_mode='r')[:n];s=np.load(p/'scores.npy',mmap_mode='r')[:n]
assert n>=1024 and np.isfinite(s).all() and (s[:,:-1]>=s[:,1:]).all()
assert (i>=0).all() and (i<r['signature']['owner_rows']).all()
assert all(len(set(row))==len(row) for row in i)
print('Full-pool retrieval smoke passed',n,flush=True)
CHECK
.venv/bin/python -u src/retrieve_owners.py --data work/data --vectors work/pilot_vectors \
  --model work/pilot_encoder/model --output work/control_retrieval
.venv/bin/python -u src/build_features.py --data work/data --retrieval work/control_retrieval \
  --output work/control_features --workers 6
if [[ ! -f work/control_owner/complete.json ]]; then
.venv/bin/python -u src/fit_owner.py --data work/data --features work/control_features --output work/control_owner
fi
.venv/bin/python src/evaluate_tune.py --data work/data \
  --prediction work/control_owner/tune_matching_results.tsv \
  --baseline work/baseline/tune/matching_results.tsv --output work/control_owner/paired_tune_report.json
if [[ ! -f work/mined_pairs/complete.json ]]; then
.venv/bin/python -u src/prepare_qwen.py --data work/data --retrieval work/control_retrieval --output work/mined_pairs
fi
