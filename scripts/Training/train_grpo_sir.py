#!/usr/bin/env python3

import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1"

import re
import json
import time
import math
import random
import sqlite3
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import numpy as np
import torch
from datasets import Dataset, load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainerCallback
from transformers.trainer_utils import get_last_checkpoint

try:
    from trl import GRPOTrainer, GRPOConfig
except Exception:
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        from trl.trainer.grpo_config import GRPOConfig
    except Exception as e:
        raise ImportError(
            "GRPOTrainer not found. Please install/upgrade trl. Original error: " + str(e)
        )

from peft import LoraConfig
import sqlparse
from sqlglot import parse_one, exp

# =========================================================
# PATHS / CONFIG
# =========================================================

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3-4B")

THIS_FILE = Path(__file__).resolve()
PROJECT_ROOT = Path(os.environ.get(
    "PROJECT_ROOT",
    THIS_FILE.parent.parent if THIS_FILE.parent.name == "scripts" else THIS_FILE.parent,
)).resolve()

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
SPIDER_DIR = Path(os.environ.get("SPIDER_DIR", DATA_DIR / "spider")).resolve()
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "trained_models" / "Qwen3_SIR")).resolve()
CACHE_ROOT = Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "hf")).resolve()
TRANSFORMERS_CACHE = Path(os.environ.get("TRANSFORMERS_CACHE", CACHE_ROOT / "transformers")).resolve()
HF_DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", CACHE_ROOT / "datasets")).resolve()
HUGGINGFACE_HUB_CACHE = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", CACHE_ROOT / "hub")).resolve()
WANDB_DIR = Path(os.environ.get("WANDB_DIR", PROJECT_ROOT / "logs")).resolve()

os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUGGINGFACE_HUB_CACHE))
os.environ.setdefault("WANDB_DIR", str(WANDB_DIR))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

for path in [
    PROJECT_ROOT,
    DATA_DIR,
    SPIDER_DIR,
    OUTPUT_DIR,
    CACHE_ROOT,
    TRANSFORMERS_CACHE,
    HF_DATASETS_CACHE,
    HUGGINGFACE_HUB_CACHE,
    WANDB_DIR,
]:
    path.mkdir(parents=True, exist_ok=True)

TABLES_JSON_PATH = SPIDER_DIR / "tables.json"
SPIDER_DATABASE_DIR = SPIDER_DIR / "database"
LOG_JSONL = OUTPUT_DIR / "train_logs.jsonl"
CACHE_DIR = str(TRANSFORMERS_CACHE)

SEED = 10

TRAIN_SPLIT = "train"
EVAL_SPLIT = "validation"
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = 128

MAX_PROMPT_TOKENS = 2048
MAX_COMPLETION_TOKENS = 700

NUM_GENERATIONS = 4
BATCH_SIZE = 1
GRAD_ACCUM_STEPS = 2
EPOCHS = 1

LR = 1e-5
WEIGHT_DECAY = 0.0
WARMUP_RATIO = 0.03
CLIP_GRAD_NORM = 1.0
BETA = 0.02

TEMPERATURE = 0.7
TOP_P = 0.8
TOP_K = 20

LOG_EVERY = 10
SAVE_EVERY_STEPS = 100
SAVE_TOTAL_LIMIT = None

LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Reward weights
W_SIR_PARSE = 20.0
W_SIR_VALID = 40.0
W_SIR_TABLE_F1 = 60.0
W_SIR_COL_F1 = 40.0

W_SQL_PARSE = 20.0
W_SQL_SCHEMA = 40.0
W_SQL_EXEC_OK = 40.0
W_SQL_TABLE_F1 = 60.0
W_SQL_COL_F1 = 40.0
W_EXEC_CORRECT = 700.0
W_SQL_EXACT = 100.0

# =========================================================
# LOGGING
# =========================================================

def is_main_process() -> bool:
    return int(os.environ.get("RANK", "0")) == 0

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

logger = logging.getLogger(__name__)


def append_jsonl(path: Path, obj: Dict[str, Any]):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


# =========================================================
# UTILS
# =========================================================

def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def normalize_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", (text or "").strip())


def normalize_sql_string(sql: str) -> str:
    if not sql:
        return ""
    try:
        formatted = sqlparse.format(
            sql,
            keyword_case="upper",
            strip_comments=True,
            reindent=False,
            use_space_around_operators=True,
        )
        return normalize_whitespace(formatted).rstrip(";")
    except Exception:
        return normalize_whitespace(sql).rstrip(";")


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


def make_serializable(x: Any) -> Any:
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, bytes):
        return x.decode("utf-8", errors="replace")
    if isinstance(x, (list, tuple)):
        return [make_serializable(v) for v in x]
    if isinstance(x, dict):
        return {str(k): make_serializable(v) for k, v in x.items()}
    return str(x)


def result_to_canonical(rows: List[Tuple[Any, ...]], order_sensitive: bool) -> List[List[Any]]:
    cooked = [list(map(make_serializable, row)) for row in rows]
    if order_sensitive:
        return cooked
    return sorted(cooked, key=lambda r: json.dumps(r, ensure_ascii=False, sort_keys=True))


def query_has_order_by(sql: str) -> bool:
    return bool(re.search(r"\border\s+by\b", sql or "", flags=re.IGNORECASE))


def prf1(pred: Set[str], gold: Set[str]) -> Dict[str, float]:
    if not pred and not gold:
        return {"precision": 1.0, "recall": 1.0, "f1": 1.0, "jaccard": 1.0}
    if not pred:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "jaccard": 0.0}
    if not gold:
        return {"precision": 0.0, "recall": 1.0, "f1": 0.0, "jaccard": 0.0}

    inter = len(pred & gold)
    p = inter / max(1, len(pred))
    r = inter / max(1, len(gold))
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    j = inter / max(1, len(pred | gold))
    return {"precision": p, "recall": r, "f1": f1, "jaccard": j}


def validate_paths() -> None:
    if not TABLES_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Spider tables.json not found at: {TABLES_JSON_PATH}\n"
            f"Set SPIDER_DIR or place Spider at: {SPIDER_DIR}"
        )
    if not SPIDER_DATABASE_DIR.exists():
        raise FileNotFoundError(
            f"Spider database directory not found at: {SPIDER_DATABASE_DIR}\n"
            f"Set SPIDER_DIR or place Spider at: {SPIDER_DIR}"
        )


# =========================================================
# SPIDER SCHEMA
# =========================================================

def load_spider_tables_json(tables_json_path: Path) -> Dict[str, Dict[str, Any]]:
    with tables_json_path.open("r", encoding="utf-8") as f:
        tables = json.load(f)

    out = {}
    for db in tables:
        db_id = db["db_id"]
        table_names_original = db["table_names_original"]
        column_names_original = db["column_names_original"]
        column_types = db.get("column_types", [])
        foreign_keys = db.get("foreign_keys", [])

        tables_out = {}
        col_idx_to_qualified = {}

        for idx, (table_idx, col_name) in enumerate(column_names_original):
            if table_idx == -1:
                continue
            table_name = table_names_original[table_idx]
            tables_out.setdefault(table_name, {"columns": []})
            tables_out[table_name]["columns"].append(col_name)
            col_idx_to_qualified[idx] = f"{table_name}.{col_name}"

        fk_readable = []
        for c1, c2 in foreign_keys:
            if c1 in col_idx_to_qualified and c2 in col_idx_to_qualified:
                fk_readable.append((col_idx_to_qualified[c1], col_idx_to_qualified[c2]))

        out[db_id] = {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "column_names_original": column_names_original,
            "column_types": column_types,
            "tables": tables_out,
            "foreign_keys_readable": fk_readable,
            "schema_table_set": set(table_names_original),
            "schema_column_set": set(
                c for t in tables_out for c in tables_out[t]["columns"]
            ),
            "qualified_column_set": set(
                f"{t}.{c}" for t in tables_out for c in tables_out[t]["columns"]
            ),
        }
    return out


def schema_to_prompt(schema: Dict[str, Any]) -> str:
    lines = [f"Database ID: {schema['db_id']}", "Schema:"]
    for table_name in schema["table_names_original"]:
        cols = schema["tables"][table_name]["columns"]
        lines.append(f"- {table_name}({', '.join(cols)})")
    if schema["foreign_keys_readable"]:
        lines.append("Foreign keys:")
        for a, b in schema["foreign_keys_readable"]:
            lines.append(f"- {a} = {b}")
    else:
        lines.append("Foreign keys: none")
    return "\n".join(lines)


# =========================================================
# SQL ANALYSIS
# =========================================================

def parse_sql_ast(sql: str) -> Tuple[Optional[exp.Expression], Optional[str]]:
    try:
        ast = parse_one(sql, dialect="sqlite")
        return ast, None
    except Exception as e:
        return None, str(e)


def extract_tables_and_columns(sql: str) -> Dict[str, Any]:
    ast, parse_err = parse_sql_ast(sql)
    if ast is None:
        return {
            "parse_ok": False,
            "parse_error": parse_err,
            "tables": set(),
            "columns": set(),
            "qualified_columns": set(),
            "table_alias_map": {},
        }

    tables: Set[str] = set()
    columns: Set[str] = set()
    qualified_columns: Set[str] = set()
    table_alias_map: Dict[str, str] = {}

    for t in ast.find_all(exp.Table):
        table_name = t.name
        if table_name:
            tables.add(table_name)
            alias = t.alias_or_name
            if alias:
                table_alias_map[alias] = table_name

    for c in ast.find_all(exp.Column):
        col_name = c.name
        if not col_name:
            continue
        columns.add(col_name)
        qualifier = c.table
        if qualifier:
            base = table_alias_map.get(qualifier, qualifier)
            qualified_columns.add(f"{base}.{col_name}")

    return {
        "parse_ok": True,
        "parse_error": None,
        "tables": tables,
        "columns": columns,
        "qualified_columns": qualified_columns,
        "table_alias_map": table_alias_map,
    }


def classify_schema_adherence(sql: str, schema: Dict[str, Any]) -> Dict[str, Any]:
    info = extract_tables_and_columns(sql)
    if not info["parse_ok"]:
        return {
            "parse_ok": False,
            "schema_valid": False,
            "invalid_tables": [],
            "invalid_columns": [],
            "tables": [],
            "columns": [],
        }

    pred_tables = set(info["tables"])
    pred_columns = set(info["columns"])
    pred_qualified_columns = set(info["qualified_columns"])

    schema_tables = schema["schema_table_set"]
    schema_columns = schema["schema_column_set"]
    schema_qualified = schema["qualified_column_set"]

    invalid_tables = sorted([t for t in pred_tables if t not in schema_tables])
    invalid_columns = []

    for qc in pred_qualified_columns:
        if qc not in schema_qualified:
            invalid_columns.append(qc)

    for c in pred_columns:
        if c == "*":
            continue
        if c not in schema_columns:
            invalid_columns.append(c)

    schema_valid = (len(invalid_tables) == 0 and len(invalid_columns) == 0)

    return {
        "parse_ok": True,
        "schema_valid": schema_valid,
        "invalid_tables": invalid_tables,
        "invalid_columns": sorted(set(invalid_columns)),
        "tables": sorted(pred_tables),
        "columns": sorted(pred_columns),
    }


def execute_sql(db_path: str, sql: str, timeout_sec: float = 30.0) -> Dict[str, Any]:
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout_sec)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return {"ok": True, "rows": rows, "error": None}
    except Exception as e:
        return {"ok": False, "rows": None, "error": str(e)}
    finally:
        if conn is not None:
            conn.close()


# =========================================================
# SIR
# =========================================================

SIR_SCHEMA_TEMPLATE = {
    "task_type": "select",
    "tables": [],
    "joins": [],
    "select": [],
    "filters": [],
    "group_by": [],
    "having": [],
    "order_by": [],
    "limit": None,
    "set_op": None,
    "subquery": None,
    "notes": ""
}


def default_sir() -> Dict[str, Any]:
    return json.loads(json.dumps(SIR_SCHEMA_TEMPLATE))


def normalize_qualified_column(name: str) -> str:
    return normalize_whitespace(name).replace(" ", "")


def is_valid_table(table: str, schema: Dict[str, Any]) -> bool:
    return table in schema["schema_table_set"]


def is_valid_column_reference(col: str, schema: Dict[str, Any]) -> bool:
    if not isinstance(col, str) or not col:
        return False
    if col == "*":
        return True
    if "." in col:
        return normalize_qualified_column(col) in {
            normalize_qualified_column(x) for x in schema["qualified_column_set"]
        }
    return col in schema["schema_column_set"]


def extract_json_object(text: str) -> Optional[str]:
    text = strip_think_blocks(text).strip()
    fenced = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()

    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        return text[start:end + 1]
    return None


def parse_sir_json(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    raw = extract_json_object(text)
    if raw is None:
        return None, "no_json_found"
    try:
        obj = json.loads(raw)
        if not isinstance(obj, dict):
            return None, "json_not_object"
        return obj, None
    except Exception as e:
        return None, str(e)


def repair_sir(sir: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    repaired = default_sir()
    repaired.update({k: v for k, v in sir.items() if k in repaired})

    for key in ["tables", "joins", "select", "filters", "group_by", "having", "order_by"]:
        if not isinstance(repaired[key], list):
            repaired[key] = []

    repaired["tables"] = [t for t in repaired["tables"] if isinstance(t, str) and is_valid_table(t, schema)]

    clean_select = []
    for s in repaired["select"]:
        if not isinstance(s, dict):
            continue
        col = s.get("column", "*")
        if col == "*" or is_valid_column_reference(col, schema):
            clean_select.append({
                "type": s.get("type", "column"),
                "column": col,
                "agg": s.get("agg"),
                "alias": s.get("alias"),
                "distinct": bool(s.get("distinct", False)),
            })
    repaired["select"] = clean_select

    clean_filters = []
    for f in repaired["filters"]:
        if not isinstance(f, dict):
            continue
        col = f.get("column")
        op = f.get("op")
        if is_valid_column_reference(col, schema) and isinstance(op, str):
            clean_filters.append({
                "column": col,
                "op": op,
                "value": f.get("value"),
                "value_type": f.get("value_type", "literal"),
            })
    repaired["filters"] = clean_filters

    repaired["group_by"] = [g for g in repaired["group_by"] if isinstance(g, str) and is_valid_column_reference(g, schema)]

    clean_having = []
    for h in repaired["having"]:
        if not isinstance(h, dict):
            continue
        col = h.get("column")
        op = h.get("op")
        if is_valid_column_reference(col, schema) and isinstance(op, str):
            clean_having.append({
                "column": col,
                "agg": h.get("agg"),
                "op": op,
                "value": h.get("value"),
            })
    repaired["having"] = clean_having

    clean_order_by = []
    for o in repaired["order_by"]:
        if not isinstance(o, dict):
            continue
        col = o.get("column")
        if is_valid_column_reference(col, schema):
            clean_order_by.append({
                "column": col,
                "direction": str(o.get("direction", "ASC")).upper(),
                "agg": o.get("agg"),
            })
    repaired["order_by"] = clean_order_by

    clean_joins = []
    for j in repaired["joins"]:
        if not isinstance(j, dict):
            continue
        l = j.get("left")
        r = j.get("right")
        if is_valid_column_reference(l, schema) and is_valid_column_reference(r, schema):
            clean_joins.append({"left": l, "right": r})
    repaired["joins"] = clean_joins

    if repaired["limit"] is not None:
        try:
            repaired["limit"] = int(repaired["limit"])
            if repaired["limit"] <= 0:
                repaired["limit"] = None
        except Exception:
            repaired["limit"] = None

    inferred_tables = set(repaired["tables"])
    for sec in ["select", "filters", "having", "order_by"]:
        for item in repaired[sec]:
            if isinstance(item, dict):
                col = item.get("column")
                if isinstance(col, str) and "." in col:
                    inferred_tables.add(col.split(".", 1)[0])
    for g in repaired["group_by"]:
        if "." in g:
            inferred_tables.add(g.split(".", 1)[0])
    for j in repaired["joins"]:
        for side in ["left", "right"]:
            col = j[side]
            if "." in col:
                inferred_tables.add(col.split(".", 1)[0])
    repaired["tables"] = sorted(t for t in inferred_tables if is_valid_table(t, schema))
    return repaired


def classify_sir_validity(sir: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    invalid_tables = []
    invalid_columns = []

    for t in sir.get("tables", []):
        if not is_valid_table(t, schema):
            invalid_tables.append(t)

    def check_col(c):
        if isinstance(c, str) and not is_valid_column_reference(c, schema):
            invalid_columns.append(c)

    for s in sir.get("select", []):
        if isinstance(s, dict):
            check_col(s.get("column"))
    for f in sir.get("filters", []):
        if isinstance(f, dict):
            check_col(f.get("column"))
    for g in sir.get("group_by", []):
        check_col(g)
    for h in sir.get("having", []):
        if isinstance(h, dict):
            check_col(h.get("column"))
    for o in sir.get("order_by", []):
        if isinstance(o, dict):
            check_col(o.get("column"))
    for j in sir.get("joins", []):
        if isinstance(j, dict):
            check_col(j.get("left"))
            check_col(j.get("right"))

    sir_tables = set(sir.get("tables", []))
    sir_cols = set()
    for sec in ["select", "filters", "having", "order_by"]:
        for item in sir.get(sec, []):
            if isinstance(item, dict) and isinstance(item.get("column"), str):
                sir_cols.add(item["column"])
    for g in sir.get("group_by", []):
        if isinstance(g, str):
            sir_cols.add(g)

    valid = len(invalid_tables) == 0 and len(invalid_columns) == 0
    return {
        "sir_valid": valid,
        "sir_invalid_tables": sorted(set(invalid_tables)),
        "sir_invalid_columns": sorted(set(invalid_columns)),
        "sir_tables": sorted(sir_tables),
        "sir_columns": sorted(sir_cols),
    }


# =========================================================
# PROMPT
# =========================================================

SYSTEM_MSG = """You are an expert SQLite semantic parser.

You must answer in EXACTLY this format:

<SIR_JSON>
{... valid JSON object for the symbolic intermediate representation ...}
</SIR_JSON>

<FINAL_SQL>
SELECT ...
</FINAL_SQL>

The SIR JSON schema is:
{
  "task_type": "select | count | aggregation | existence | comparison | set_op",
  "tables": ["table1", "table2"],
  "joins": [{"left": "table1.col_a", "right": "table2.col_b"}],
  "select": [{"type": "column | star", "column": "table.col or *", "agg": null, "alias": null, "distinct": false}],
  "filters": [{"column": "table.col", "op": "= | != | > | >= | < | <= | like | in | not in", "value": "literal", "value_type": "literal"}],
  "group_by": ["table.col"],
  "having": [{"column": "table.col", "agg": "COUNT | AVG | MIN | MAX | SUM", "op": ">", "value": 1}],
  "order_by": [{"column": "table.col", "direction": "ASC | DESC", "agg": null}],
  "limit": null,
  "set_op": null,
  "subquery": null,
  "notes": "brief reasoning summary"
}

Rules:
- Use only schema-valid tables and columns.
- Prefer qualified columns like table.column.
- Then produce final SQLite SQL implementing the SIR.
- Return only those two tagged sections.
"""


def build_prompt(schema_text: str, question: str) -> str:
    return f"""{schema_text}

Question:
{question}
""".strip()


def qwen_chat_prompt(tokenizer, schema_text: str, question: str) -> str:
    messages = [
        {"role": "system", "content": SYSTEM_MSG},
        {"role": "user", "content": build_prompt(schema_text, question)},
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )


def extract_sir_and_sql_from_completion(text: str) -> Tuple[Optional[Dict[str, Any]], Optional[str], Dict[str, Any]]:
    text = strip_think_blocks(text or "")

    sir_match = re.search(r"<SIR_JSON>\s*([\s\S]*?)\s*</SIR_JSON>", text, flags=re.IGNORECASE)
    sql_match = re.search(r"<FINAL_SQL>\s*([\s\S]*?)\s*</FINAL_SQL>", text, flags=re.IGNORECASE)

    sir_obj = None
    sir_error = None
    sql_text = None

    if sir_match:
        raw_sir = sir_match.group(1).strip()
        try:
            sir_obj = json.loads(raw_sir)
            if not isinstance(sir_obj, dict):
                sir_obj = None
                sir_error = "sir_not_json_object"
        except Exception as e:
            sir_error = str(e)
    else:
        sir_obj, sir_error = parse_sir_json(text)

    if sql_match:
        sql_text = sql_match.group(1).strip()
    else:
        m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
        if m:
            sql_text = m.group(0).strip()

    return sir_obj, sql_text, {
        "raw_text": text,
        "sir_found": sir_match is not None,
        "sql_found": sql_match is not None,
        "sir_error": sir_error,
    }


# =========================================================
# DATASET BUILD
# =========================================================

def build_rows(tokenizer, split: str, schemas: Dict[str, Dict[str, Any]], max_samples: Optional[int] = None) -> List[Dict[str, Any]]:
    ds = load_dataset("xlangai/spider", split=split, cache_dir=str(HF_DATASETS_CACHE))

    rows = []
    n = len(ds) if max_samples is None else min(len(ds), max_samples)

    for i in range(n):
        ex = ds[i]
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]

        schema = schemas[db_id]
        schema_text = schema_to_prompt(schema)
        db_path = SPIDER_DATABASE_DIR / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            continue

        gold_schema = classify_schema_adherence(gold_sql, schema)
        prompt = qwen_chat_prompt(tokenizer, schema_text, question)
        prompt_token_len = len(tokenizer(prompt, add_special_tokens=False).input_ids)

        rows.append({
            "prompt": prompt,
            "prompt_token_len": int(prompt_token_len),
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "db_path": str(db_path),
            "schema_text": schema_text,
            "gold_tables": gold_schema["tables"],
            "gold_columns": gold_schema["columns"],
        })

    return rows


def rows_to_dataset(rows: List[Dict[str, Any]]) -> Dataset:
    import pandas as pd
    return Dataset.from_pandas(pd.DataFrame(rows))


# =========================================================
# REWARD
# =========================================================

def spider_reward(
    prompts: List[str],
    completions,
    completions_ids=None,
    db_id: List[str] = None,
    question: List[str] = None,
    gold_sql: List[str] = None,
    db_path: List[str] = None,
    schema_text: List[str] = None,
    gold_tables: List[List[str]] = None,
    gold_columns: List[List[str]] = None,
    trainer_state=None,
    **kwargs
) -> List[float]:

    if completions and isinstance(completions[0], list):
        norm_completions = []
        for c in completions:
            if c and isinstance(c[0], dict):
                norm_completions.append(c[0].get("content", ""))
            else:
                norm_completions.append(str(c))
    else:
        norm_completions = [str(c) for c in completions]

    m = len(prompts)
    n = len(norm_completions)
    if m <= 0:
        return [0.0] * n

    k = max(1, n // m)

    def prompt_index(i: int) -> int:
        return min(m - 1, i // k)

    rewards: List[float] = []
    schemas = load_spider_tables_json(TABLES_JSON_PATH)

    for i, comp in enumerate(norm_completions):
        j = prompt_index(i)

        ex_db_id = db_id[j]
        ex_gold_sql = gold_sql[j]
        ex_db_path = db_path[j]
        ex_gold_tables = set(gold_tables[j] or [])
        ex_gold_columns = set(gold_columns[j] or [])

        schema = schemas[ex_db_id]

        sir_obj, pred_sql, aux = extract_sir_and_sql_from_completion(comp)

        reward = 0.0

        if sir_obj is not None:
            reward += W_SIR_PARSE
            repaired_sir = repair_sir(sir_obj, schema)
            sir_metrics = classify_sir_validity(repaired_sir, schema)
            if sir_metrics["sir_valid"]:
                reward += W_SIR_VALID

            sir_tables = set(sir_metrics["sir_tables"])
            sir_cols = set(sir_metrics["sir_columns"])
            sir_table_scores = prf1(sir_tables, ex_gold_tables)
            sir_col_scores = prf1(sir_cols, ex_gold_columns)

            reward += W_SIR_TABLE_F1 * sir_table_scores["f1"]
            reward += W_SIR_COL_F1 * sir_col_scores["f1"]
        else:
            repaired_sir = default_sir()
            sir_metrics = {
                "sir_valid": False,
                "sir_tables": [],
                "sir_columns": [],
                "sir_invalid_tables": [],
                "sir_invalid_columns": [],
            }

        execution_correct = False
        normalized_exact_match = False
        pred_exec_ok = False

        if pred_sql:
            pred_schema = classify_schema_adherence(pred_sql, schema)

            if pred_schema["parse_ok"]:
                reward += W_SQL_PARSE

            if pred_schema["schema_valid"]:
                reward += W_SQL_SCHEMA

            pred_tables = set(pred_schema["tables"])
            pred_cols = set(pred_schema["columns"])

            table_scores = prf1(pred_tables, ex_gold_tables)
            col_scores = prf1(pred_cols, ex_gold_columns)

            reward += W_SQL_TABLE_F1 * table_scores["f1"]
            reward += W_SQL_COL_F1 * col_scores["f1"]

            pred_exec = execute_sql(ex_db_path, pred_sql)
            gold_exec = execute_sql(ex_db_path, ex_gold_sql)

            pred_exec_ok = pred_exec["ok"]
            if pred_exec_ok:
                reward += W_SQL_EXEC_OK

            if pred_exec["ok"] and gold_exec["ok"]:
                order_sensitive = query_has_order_by(pred_sql) or query_has_order_by(ex_gold_sql)
                pred_res = result_to_canonical(pred_exec["rows"], order_sensitive)
                gold_res = result_to_canonical(gold_exec["rows"], order_sensitive)
                execution_correct = (pred_res == gold_res)

            normalized_exact_match = (normalize_sql_string(pred_sql) == normalize_sql_string(ex_gold_sql))

            if execution_correct:
                reward += W_EXEC_CORRECT
            if normalized_exact_match:
                reward += W_SQL_EXACT
        else:
            pred_schema = {
                "parse_ok": False,
                "schema_valid": False,
                "tables": [],
                "columns": [],
                "invalid_tables": [],
                "invalid_columns": [],
            }

        rewards.append(float(reward))

        if is_main_process():
            append_jsonl(LOG_JSONL, {
                "db_id": ex_db_id,
                "question": question[j],
                "gold_sql": ex_gold_sql,
                "completion_text": comp,
                "parsed_sir": repaired_sir,
                "sir_valid": sir_metrics["sir_valid"],
                "pred_sql": pred_sql,
                "pred_parse_ok": pred_schema["parse_ok"],
                "pred_schema_valid": pred_schema["schema_valid"],
                "pred_exec_ok": pred_exec_ok,
                "execution_correct": execution_correct,
                "normalized_exact_match": normalized_exact_match,
                "reward": reward,
                "timestamp": now_ts(),
            })

    return rewards


# =========================================================
# EVAL CALLBACK
# =========================================================

class EpochTrackerCallback(TrainerCallback):
    CURRENT_EPOCH = 0

    def on_epoch_begin(self, args, state, control, **kwargs):
        if state.epoch is not None:
            try:
                self.CURRENT_EPOCH = int(state.epoch)
            except Exception:
                pass


# =========================================================
# MAIN
# =========================================================

def main():
    validate_paths()

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(SEED)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Spider dir: %s", SPIDER_DIR)
    logger.info("Output dir: %s", OUTPUT_DIR)
    logger.info("Cache root: %s", CACHE_ROOT)

    schemas = load_spider_tables_json(TABLES_JSON_PATH)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, cache_dir=CACHE_DIR)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    logger.info("Building Spider training rows...")
    train_rows = build_rows(tokenizer, TRAIN_SPLIT, schemas, MAX_TRAIN_SAMPLES)
    eval_rows = build_rows(tokenizer, EVAL_SPLIT, schemas, MAX_EVAL_SAMPLES)

    train_ds = rows_to_dataset(train_rows)
    eval_ds = rows_to_dataset(eval_rows)

    logger.info("Train rows: %s | Eval rows: %s", len(train_rows), len(eval_rows))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        cache_dir=CACHE_DIR,
        device_map=None,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
    )

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False

    lora_cfg = LoraConfig(
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGET_MODULES,
        bias="none",
        task_type="CAUSAL_LM",
    )

    args = GRPOConfig(
        output_dir=str(OUTPUT_DIR),
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM_STEPS,
        num_train_epochs=EPOCHS,
        max_grad_norm=CLIP_GRAD_NORM,
        warmup_ratio=WARMUP_RATIO,
        num_generations=NUM_GENERATIONS,
        generation_batch_size=NUM_GENERATIONS,
        max_completion_length=MAX_COMPLETION_TOKENS,
        temperature=TEMPERATURE,
        top_p=TOP_P,
        logging_steps=LOG_EVERY,
        logging_dir=str(OUTPUT_DIR / "logs"),
        log_completions=False,
        loss_type="grpo",
        beta=BETA,
        scale_rewards="batch",
        fp16=torch.cuda.is_available(),
        bf16=False,
        remove_unused_columns=False,
        save_strategy="steps",
        save_steps=SAVE_EVERY_STEPS,
        save_total_limit=SAVE_TOTAL_LIMIT,
        ddp_find_unused_parameters=False,
        report_to=[],
    )

    for name, value in [("top_k", TOP_K)]:
        try:
            if hasattr(args, name):
                setattr(args, name, value)
        except Exception:
            pass

    def reward_with_tok(*a, **kw):
        kw["tokenizer"] = tokenizer
        return spider_reward(*a, **kw)

    trainer = GRPOTrainer(
        model=model,
        reward_funcs=reward_with_tok,
        args=args,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        processing_class=tokenizer,
        peft_config=lora_cfg,
        callbacks=[EpochTrackerCallback()],
    )

    resume_ckpt = get_last_checkpoint(str(OUTPUT_DIR))
    if resume_ckpt:
        logger.info("Resuming from checkpoint: %s", resume_ckpt)

    trainer.train(resume_from_checkpoint=resume_ckpt)

    trainer.model.save_pretrained(str(OUTPUT_DIR))
    trainer.save_model(str(OUTPUT_DIR))
    tokenizer.save_pretrained(str(OUTPUT_DIR))

    logger.info("Saved final model to %s", OUTPUT_DIR)


if __name__ == "__main__":
    main()