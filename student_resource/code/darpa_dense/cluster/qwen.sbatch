#!/usr/bin/env bash
#SBATCH --job-name=DARPA-Qwen06
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
test -f work/mined_direct/complete.json
test -f work/qwen_base/provenance.json
if [[ ! -f work/qwen_fit/complete.json ]]; then
srun .venv/bin/python -u src/train_qwen.py --pairs work/mined_direct/pairs.parquet \
  --base-model work/qwen_base --output work/qwen_fit --batch-size 16 --accumulate 4
fi
