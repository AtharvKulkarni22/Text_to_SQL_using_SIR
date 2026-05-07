#!/bin/bash
#SBATCH --job-name=train_sir
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00
#SBATCH --output=logs/slurm/%x-%j.out
#SBATCH --error=logs/slurm/%x-%j.err

set -euo pipefail

# Go to repo root no matter where sbatch is called from
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$REPO_ROOT"

# Optional cluster modules
module --force purge 2>/dev/null || true
module load cuda/12.2 2>/dev/null || true

# Project paths
export PROJECT_ROOT="${PROJECT_ROOT:-$REPO_ROOT}"
export TRAIN_SCRIPT="${TRAIN_SCRIPT:-$REPO_ROOT/scripts/Training/train_grpo_sir.py}"
export SPIDER_DIR="${SPIDER_DIR:-$REPO_ROOT/data/spider}"
export OUTPUT_DIR="${OUTPUT_DIR:-$REPO_ROOT/outputs/trained_models/Qwen3_SIR}"
export LOG_DIR="${LOG_DIR:-$REPO_ROOT/logs}"
export VENV_PATH="${VENV_PATH:-$REPO_ROOT/env}"

# Cache paths
export HF_HOME="${HF_HOME:-$REPO_ROOT/.cache/hf}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-$HF_HOME/hub}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HF_HOME/datasets}"
export TRANSFORMERS_CACHE="${TRANSFORMERS_CACHE:-$HF_HOME/transformers}"
export WANDB_DIR="${WANDB_DIR:-$LOG_DIR/wandb}"

# Runtime settings
export TRANSFORMERS_NO_TORCHVISION=1
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
export NCCL_P2P_LEVEL=SYS
export NCCL_IB_DISABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

mkdir -p "$LOG_DIR/slurm" "$OUTPUT_DIR" "$OUTPUT_DIR/logs"
mkdir -p "$HUGGINGFACE_HUB_CACHE" "$HF_DATASETS_CACHE" "$TRANSFORMERS_CACHE" "$WANDB_DIR"

if [ ! -d "$VENV_PATH" ]; then
  echo "Virtual environment not found at: $VENV_PATH"
  echo "Create it first, for example:"
  echo "bash scripts/Training/setup_env.sh"
  exit 1
fi

source "$VENV_PATH/bin/activate"

python - <<'PY'
import os
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
print("num gpus:", torch.cuda.device_count())
for i in range(torch.cuda.device_count()):
    p = torch.cuda.get_device_properties(i)
    print(f"[{i}] {torch.cuda.get_device_name(i)}  {p.total_memory/1e9:.1f} GB")
print("TRAIN_SCRIPT:", os.environ.get("TRAIN_SCRIPT"))
print("SPIDER_DIR:", os.environ.get("SPIDER_DIR"))
print("OUTPUT_DIR:", os.environ.get("OUTPUT_DIR"))
PY

srun python "$TRAIN_SCRIPT"