#!/usr/bin/env bash
#SBATCH --job-name=DARPA-features
#SBATCH --partition=cpu-short
#SBATCH --qos=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --time=12:00:00
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
test -f work/full_retrieval/train_India/complete.json
test -f work/full_retrieval/train_US/complete.json
srun .venv/bin/python -u src/build_features.py --data work/data \
  --retrieval work/full_retrieval --output work/full_features \
  --workers 6 --max-chunks 1
.venv/bin/python - <<'PY'
import json
from pathlib import Path
import pyarrow.parquet as pq
for country in ('India', 'US'):
    root=Path('work/full_features')/('train_'+country)
    config=json.loads((root/'config.json').read_text())
    table=pq.read_table(root/'features-00000.parquet',columns=['target_row','owner_row'])
    k=config['retrieval']['k']
    rows=table['target_row'].to_numpy()
    assert len(rows)>0 and len(rows)%k==0
    assert all(len(set(rows[i:i+k]))==1 for i in range(0,len(rows),k))
    print('Full feature chunk smoke passed',country,len(rows),flush=True)
PY
srun .venv/bin/python -u src/build_features.py --data work/data \
  --retrieval work/full_retrieval --output work/full_features --workers 6
