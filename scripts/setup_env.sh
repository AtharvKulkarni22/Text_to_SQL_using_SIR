#!/usr/bin/env bash
set -e

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_DIR="$PROJECT_ROOT/env"

echo "Setting up Text-to-SQL using SIR environment"
echo "Project root: $PROJECT_ROOT"

cd "$PROJECT_ROOT"

python3 -m venv "$ENV_DIR"
source "$ENV_DIR/bin/activate"

mkdir -p \
  .cache/pip \
  hf/transformers \
  hf/datasets \
  hf/hub \
  logs \
  data \
  outputs

export PIP_CACHE_DIR="$PROJECT_ROOT/.cache/pip"
export HF_HOME="$PROJECT_ROOT/hf"
export TRANSFORMERS_CACHE="$PROJECT_ROOT/hf/transformers"
export HF_DATASETS_CACHE="$PROJECT_ROOT/hf/datasets"
export HUGGINGFACE_HUB_CACHE="$PROJECT_ROOT/hf/hub"
export WANDB_DIR="$PROJECT_ROOT/logs"
export TOKENIZERS_PARALLELISM=false

python -m pip install --upgrade pip setuptools wheel

pip install \
  torch torchvision torchaudio \
  transformers datasets accelerate peft trl wandb \
  pandas numpy scipy sentencepiece protobuf safetensors \
  sqlparse tqdm sqlglot

echo ""
echo "Setup complete."
echo "To activate the environment later, run:"
echo "source env/bin/activate"