#!/usr/bin/env bash
#SBATCH --job-name=DARPA-evaluate
#SBATCH --partition=cpu-short
#SBATCH --qos=cpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00
#SBATCH --propagate=NONE
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err
set -euo pipefail
umask 077
ulimit -v unlimited
module purge
module load pytorch/2.10
export TMPDIR="$PWD/tmp" OMP_NUM_THREADS=2
mkdir -p "$TMPDIR"
test -f work/full_owner/complete.json
test -f work/baseline/tune/matching_results.tsv
srun .venv/bin/python -u src/evaluate_tune.py --data work/data \
  --prediction work/full_owner/tune_matching_results.tsv \
  --baseline work/baseline/tune/matching_results.tsv \
  --output work/full_owner/paired_tune_report.json
