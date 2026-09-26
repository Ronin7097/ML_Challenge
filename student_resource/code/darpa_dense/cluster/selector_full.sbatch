#!/usr/bin/env bash
#SBATCH --job-name=DARPA-selector
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
test -f work/full_features/train_India/complete.json
test -f work/full_features/train_US/complete.json
test -f work/full_owner/pair.txt
if [[ ! -f work/full_selector/complete.json ]]; then
  srun .venv/bin/python -u src/fit_selector.py --data work/data \
    --features work/full_features --pair-model work/full_owner/pair.txt \
    --output work/full_selector
fi
test -f work/baseline/tune/matching_results.tsv
srun .venv/bin/python -u src/evaluate_tune.py --data work/data \
  --prediction work/full_selector/tune_matching_results.tsv \
  --baseline work/baseline/tune/matching_results.tsv \
  --output work/full_selector/paired_tune_report.json
