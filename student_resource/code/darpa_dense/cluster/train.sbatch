#!/usr/bin/env bash
#SBATCH --job-name=DARPA-granite
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
srun .venv/bin/python src/test_gradient_cache.py
if [[ ! -f work/full_encoder/complete.json ]]; then
srun .venv/bin/python -u src/train_encoder.py --pairs work/data/encoder_pairs.parquet \
  --base-model work/base_model --output work/full_encoder --batch-size 1024 --microbatch 128 --resume
fi
srun .venv/bin/python -u src/encode.py --data work/data --model work/full_encoder/model \
  --output work/full_vectors --glob 'train_s1_*.parquet' --batch-size 1024
if [[ ! -f work/full_probe/report.json ]]; then
srun .venv/bin/python -u src/probe_retrieval.py --data work/data --vectors work/full_vectors \
  --model work/full_encoder/model --output work/full_probe
fi
