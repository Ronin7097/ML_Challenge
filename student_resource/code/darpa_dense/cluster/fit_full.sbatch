#!/usr/bin/env bash
#SBATCH --job-name=DARPA-owner-fit
#SBATCH --partition=cpu-short
#SBATCH --qos=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --propagate=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 077
ulimit -v unlimited
module purge
module load pytorch/2.10
export TMPDIR="$PWD/tmp" OMP_NUM_THREADS=6
mkdir -p "$TMPDIR"
.venv/bin/python -m pip freeze > "logs/environment-${SLURM_JOB_ID}.txt"
test -f work/full_features/train_India/complete.json
test -f work/full_features/train_US/complete.json
if [[ ! -f work/full_owner/complete.json ]]; then
  srun .venv/bin/python -u src/fit_owner.py --data work/data \
    --features work/full_features --output work/full_owner --bags 5
fi
