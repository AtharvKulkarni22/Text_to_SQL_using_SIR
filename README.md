# Text_to_SQL_using_SIR

This repository contains the Code for Enhancing Text-to-SQL with Schema-Grounded Symbolic Intermediate Representations.

A neuro-symbolic Text-to-SQL project that improves SQL generation by introducing a **Schema-Grounded Symbolic Intermediate Representation (SIR)** between natural language and final SQL.

Instead of mapping:

`Question + Schema -> SQL`

this project uses:

`Question + Schema -> SIR -> SQL`

The goal is to improve:
- schema grounding
- SQL validity
- execution success
- interpretability of the reasoning process

The training pipeline uses GRPO-based reinforcement learning to reward valid symbolic plans and correct final SQL. The inference pipeline evaluates the two-stage SIR-to-SQL generation process on Spider.

---

## Repository structure

```text
Text_to_SQL_using_SIR/
├── LICENSE
├── README.md
└── scripts/
    ├── setup_env.sh
    ├── Inference/
    │   ├── inference_sir.py
    │   └── run_inference_sir.slurm
    └── Training/
        ├── run_train_grpo_sir.slurm
        └── train_grpo_sir.py
```

### What each file does

- `scripts/setup_env.sh`  
  Creates a local Python virtual environment and installs the required packages.

- `scripts/Training/train_grpo_sir.py`  
  Trains the SIR-based Text-to-SQL model with GRPO, LoRA, Spider supervision, schema-aware rewards, and SQL execution rewards.

- `scripts/Training/run_train_grpo_sir.slurm`  
  Example Slurm job script for launching training on an HPC cluster.

- `scripts/Inference/inference_sir.py`  
  Runs two-stage inference: first generates SIR JSON, repairs and validates it, then generates final SQL and evaluates execution and schema metrics.

- `scripts/Inference/run_inference_sir.slurm`  
  Example Slurm job script for running inference on a trained checkpoint or base model.

---

## Method overview

This repository implements a neuro-symbolic Text-to-SQL framework:

1. The model reads the database schema and natural-language question.
2. It first produces a **Symbolic Intermediate Representation (SIR)** in JSON.
3. The SIR is validated and repaired against the database schema.
4. The model then generates final SQLite SQL from the repaired SIR.
5. SQL is executed against the Spider database to compute evaluation metrics.

### Why use SIR?

The symbolic plan makes the model’s reasoning more explicit by exposing:
- tables
- joins
- selected columns
- filters
- grouping
- ordering
- limits
- aggregation

This helps reduce schema hallucinations and improve execution robustness. Furthermore, the model recieves rewards based on this reasoning process (SIR) therby improving the downstream SQL generation.

---

## Dataset

This project uses the **Spider** benchmark.

Download Spider here:

```text
https://drive.google.com/file/d/1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J/view
```

After downloading and extracting, place Spider here:

```text
Text_to_SQL_using_SIR/
└── data/
    └── spider/
        ├── tables.json
        └── database/
```

The scripts use `data/spider` by default.

---

## Environment setup

From the repository root:

```bash
bash scripts/setup_env.sh
```

Then activate the environment:

```bash
source env/bin/activate
```

### Main dependencies

This project uses:
- `torch`
- `transformers`
- `datasets`
- `accelerate`
- `peft`
- `trl`
- `sqlparse`
- `sqlglot`
- `pandas`
- `numpy`

## Default paths used by the code

The public versions of the scripts are portable and use repo-relative defaults.

### Training defaults
- Spider: `data/spider`
- Outputs: `outputs/trained_models/Qwen3_SIR`
- Cache: `.cache/hf`

### Inference defaults
- Spider: `data/spider`
- Outputs: `outputs/inference/sir_sql_baseline`
- Checkpoint: optional, user-provided

All of these can also be overridden with environment variables or CLI arguments.

---

## Model compatibility

The training and inference scripts in this repository are developed for the **Qwen 3-4B** model family. They may require adaptation for other model families, especially if the tokenizer, chat template behavior, generation format, or fine-tuning interface differs.

---

## Training

### What training does

The training script:
- loads Spider from Hugging Face
- loads local Spider schema metadata and SQLite databases
- formats prompts for SIR and SQL generation
- computes rewards for:
  - SIR parseability
  - SIR validity
  - table and column grounding
  - SQL parse success
  - SQL schema validity
  - SQL execution success
  - execution correctness
  - normalized SQL exact match

### Run training locally

From the repo root:

```bash
source env/bin/activate
python scripts/Training/train_grpo_sir.py
```

### Run training on Slurm

Submit:

```bash
sbatch scripts/Training/run_train_grpo_sir.slurm
```

### Training outputs

By default, training writes to:

```text
outputs/trained_models/Qwen3_SIR/
```

Typical contents include:
- model checkpoints
- tokenizer files
- training logs
- `train_logs.jsonl`

---

## Inference

### What inference does

The inference script:
1. loads a Spider split from Hugging Face
2. loads local Spider schemas and SQLite DBs
3. generates SIR JSON
4. repairs and validates the SIR
5. generates SQL from the repaired SIR
6. executes predicted SQL and gold SQL
7. saves detailed results and summary metrics

### Run inference with the base model

```bash
source env/bin/activate
python scripts/Inference/inference_sir.py
```

### Run inference with a trained checkpoint

```bash
source env/bin/activate
python scripts/Inference/inference_sir.py \
  --checkpoint_path outputs/trained_models/Qwen3_SIR/checkpoint-3000
```

### Useful inference command

```bash
python scripts/Inference/inference_sir.py \
  --base_model_id Qwen/Qwen3-4B \
  --spider_dir data/spider \
  --output_dir outputs/inference/Qwen3_SIR \
  --split validation \
  --checkpoint_path outputs/trained_models/Qwen3_SIR/checkpoint-3000
```

### Run inference on Slurm

```bash
sbatch scripts/Inference/run_inference_sir.slurm
```

---

## Inference outputs

The inference script writes:
- `predictions.jsonl` — detailed per-example outputs
- `predictions.csv` — compact tabular summary
- `summary.json` — aggregate metrics

Common metrics include:
- execution accuracy
- normalized exact match
- SQL parse success rate
- SQL execution success rate
- SQL schema-valid rate
- SIR parse success rate
- SIR valid rate
- SIR table grounding match rate

---

## Expected Spider split behavior

- Training uses:
  - `train`
  - `validation` for evaluation

- Inference defaults to:
  - `validation`

---

## Slurm notes

The provided `.slurm` files are examples and may need edits for your cluster, especially:
- partition
- qos
- account
- GPU type
- memory
- time limit
---

## Recommended workflow

1. Clone the repo
2. Download and extract Spider into `data/spider`
3. Run `bash scripts/setup_env.sh`
4. Activate the environment
5. Train with:
   ```bash
   python scripts/Training/train_grpo_sir.py
   ```
6. Run inference with:
   ```bash
   python scripts/Inference/inference_sir.py --checkpoint_path <checkpoint_dir>
   ```

## Contact Info
- Atharv Kulkarni (athkulk@cs.unc.edu)

## License

See `LICENSE`.
