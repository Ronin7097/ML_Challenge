#!/usr/bin/env bash
#SBATCH --job-name=DARPA-Qwen-score
#SBATCH --partition=gpu-short
#SBATCH --qos=gpu-short
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=24G
#SBATCH --gres=gpu:a100-80gb:1
#SBATCH --time=02:00:00
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
test -f work/full_owner/complete.json
test -f work/qwen_fit/complete.json
test -f work/qwen_fit/adapter/adapter_model.safetensors
srun .venv/bin/python -u src/score_qwen_groups.py --data work/data \
  --retrieval work/full_retrieval --owner-model work/full_owner \
  --base-model work/qwen_base --adapter work/qwen_fit/adapter \
  --output work/qwen_score_smoke --max-groups 1000 --batch-size 32
