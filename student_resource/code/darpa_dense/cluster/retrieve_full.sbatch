#!/usr/bin/env bash
#SBATCH --job-name=DARPA-retrieve
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=08:00:00
#SBATCH --propagate=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 077
ulimit -v unlimited
module purge
module load pytorch/2.10
export TMPDIR="$PWD/tmp"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
export TOKENIZERS_PARALLELISM=false OMP_NUM_THREADS=6
mkdir -p "$TMPDIR"
.venv/bin/python -m pip freeze > "logs/environment-${SLURM_JOB_ID}.txt"
test -f work/full_encoder/complete.json
test -f work/full_probe/report.json
if [[ ! -f work/full_retrieval/train_India/complete.json ]]; then
  srun .venv/bin/python -u src/retrieve_owners.py --data work/data \
    --vectors work/full_vectors --model work/full_encoder/model \
    --output work/full_retrieval --country India --max-rows 1024
  .venv/bin/python - <<'PY'
import json
from pathlib import Path
import numpy as np
p=Path('work/full_retrieval/train_India')
r=json.loads((p/'progress.json').read_text());n=r['rows_done']
i=np.load(p/'indices.npy',mmap_mode='r')[:n]
s=np.load(p/'scores.npy',mmap_mode='r')[:n]
assert n>=1024 and np.isfinite(s).all() and (s[:,:-1]>=s[:,1:]).all()
assert (i>=0).all() and (i<r['signature']['owner_rows']).all()
assert all(len(set(row))==len(row) for row in i)
print('Full-encoder retrieval smoke passed',n,flush=True)
PY
fi
srun .venv/bin/python -u src/retrieve_owners.py --data work/data \
  --vectors work/full_vectors --model work/full_encoder/model \
  --output work/full_retrieval
