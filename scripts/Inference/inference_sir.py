#!/usr/bin/env python3
import os
os.environ.setdefault("TRANSFORMERS_NO_TORCHVISION", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import json
import csv
import time
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Set

import torch
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM

import sqlparse
from sqlglot import parse_one, exp
from peft import PeftModel


# =========================================================
# PATHS / CONFIG
# =========================================================

THIS_FILE = Path(__file__).resolve()
DEFAULT_PROJECT_ROOT = THIS_FILE.parent.parent if THIS_FILE.parent.name == "scripts" else THIS_FILE.parent
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", DEFAULT_PROJECT_ROOT)).resolve()

DATA_DIR = Path(os.environ.get("DATA_DIR", PROJECT_ROOT / "data")).resolve()
SPIDER_DIR_DEFAULT = Path(os.environ.get("SPIDER_DIR", DATA_DIR / "spider")).resolve()
OUTPUT_ROOT_DEFAULT = Path(os.environ.get("OUTPUT_DIR", PROJECT_ROOT / "outputs" / "inference" / "sir_sql_baseline")).resolve()
CACHE_ROOT = Path(os.environ.get("HF_HOME", PROJECT_ROOT / ".cache" / "hf")).resolve()
TRANSFORMERS_CACHE = Path(os.environ.get("TRANSFORMERS_CACHE", CACHE_ROOT / "transformers")).resolve()
HF_DATASETS_CACHE = Path(os.environ.get("HF_DATASETS_CACHE", CACHE_ROOT / "datasets")).resolve()
HUGGINGFACE_HUB_CACHE = Path(os.environ.get("HUGGINGFACE_HUB_CACHE", CACHE_ROOT / "hub")).resolve()

os.environ.setdefault("HF_HOME", str(CACHE_ROOT))
os.environ.setdefault("TRANSFORMERS_CACHE", str(TRANSFORMERS_CACHE))
os.environ.setdefault("HF_DATASETS_CACHE", str(HF_DATASETS_CACHE))
os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(HUGGINGFACE_HUB_CACHE))

for path in [PROJECT_ROOT, DATA_DIR, CACHE_ROOT, TRANSFORMERS_CACHE, HF_DATASETS_CACHE, HUGGINGFACE_HUB_CACHE]:
    path.mkdir(parents=True, exist_ok=True)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


# =========================================================
# Utilities
# =========================================================

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def now_ts() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")


def strip_think_blocks(text: str) -> str:
    if not text:
        return ""
    return re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE).strip()


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


def safe_json_dump(obj: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def append_jsonl(path: Path, obj: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


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


def validate_spider_dir(spider_dir: Path) -> Tuple[Path, Path]:
    tables_json_path = spider_dir / "tables.json"
    db_root = spider_dir / "database"

    if not tables_json_path.exists():
        raise FileNotFoundError(
            f"Missing Spider tables.json at: {tables_json_path}\n"
            f"Set --spider_dir or SPIDER_DIR to your local Spider folder."
        )
    if not db_root.exists():
        raise FileNotFoundError(
            f"Missing Spider database directory at: {db_root}\n"
            f"Set --spider_dir or SPIDER_DIR to your local Spider folder."
        )
    return tables_json_path, db_root


# =========================================================
# Spider schema loading
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
        primary_keys = db.get("primary_keys", [])
        foreign_keys = db.get("foreign_keys", [])

        tables_out = {}
        all_columns = []
        column_index_to_qualified = {}

        for col_idx, (table_idx, col_name) in enumerate(column_names_original):
            if table_idx == -1:
                continue
            table_name = table_names_original[table_idx]
            tables_out.setdefault(table_name, {"columns": []})
            tables_out[table_name]["columns"].append(col_name)
            all_columns.append((table_name, col_name, col_idx))
            column_index_to_qualified[col_idx] = f"{table_name}.{col_name}"

        fk_readable = []
        for c1, c2 in foreign_keys:
            if c1 in column_index_to_qualified and c2 in column_index_to_qualified:
                fk_readable.append((column_index_to_qualified[c1], column_index_to_qualified[c2]))

        out[db_id] = {
            "db_id": db_id,
            "table_names_original": table_names_original,
            "column_names_original": column_names_original,
            "column_types": column_types,
            "primary_keys": primary_keys,
            "foreign_keys": foreign_keys,
            "foreign_keys_readable": fk_readable,
            "tables": tables_out,
            "all_columns": all_columns,
            "schema_table_set": set(table_names_original),
            "schema_column_set": set(col for t in tables_out for col in tables_out[t]["columns"]),
            "qualified_column_set": set(f"{t}.{c}" for t in tables_out for c in tables_out[t]["columns"]),
            "qualified_to_type": {
                f"{table_names_original[tidx]}.{col_name}": column_types[idx]
                for idx, (tidx, col_name) in enumerate(column_names_original)
                if tidx >= 0 and idx < len(column_types)
            },
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
# SQL parsing / schema adherence
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
            "qualified_columns": [],
            "table_alias_map": {},
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
        "qualified_columns": sorted(pred_qualified_columns),
        "table_alias_map": info["table_alias_map"],
    }


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


# =========================================================
# SQLite execution
# =========================================================

def execute_sql(db_path: str, sql: str, timeout_sec: float = 30.0) -> Dict[str, Any]:
    conn = None
    try:
        conn = sqlite3.connect(db_path, timeout=timeout_sec)
        cur = conn.cursor()
        cur.execute(sql)
        rows = cur.fetchall()
        return {"ok": True, "rows": rows, "row_count": len(rows), "error": None}
    except Exception as e:
        return {"ok": False, "rows": None, "row_count": None, "error": str(e)}
    finally:
        if conn is not None:
            conn.close()


def classify_execution_error(err: Optional[str]) -> str:
    if not err:
        return "none"
    e = err.lower()
    if "no such table" in e:
        return "schema_table_error"
    if "no such column" in e:
        return "schema_column_error"
    if "ambiguous column name" in e:
        return "schema_ambiguous_column_error"
    if "cannot join" in e or "join clause" in e:
        return "schema_join_error"
    if "syntax error" in e or "near" in e:
        return "syntax_error"
    return "other_execution_error"


# =========================================================
# SIR definition and validation
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
    "notes": "",
}


def default_sir() -> Dict[str, Any]:
    return json.loads(json.dumps(SIR_SCHEMA_TEMPLATE))


def extract_json_object(text: str) -> Optional[str]:
    text = strip_think_blocks(text).strip()

    fenced = re.findall(r"```json\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced:
        return fenced[-1].strip()

    fenced_any = re.findall(r"```[\w]*\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if fenced_any:
        cand = fenced_any[-1].strip()
        if cand.startswith("{") and cand.endswith("}"):
            return cand

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
        return normalize_qualified_column(col) in {normalize_qualified_column(x) for x in schema["qualified_column_set"]}
    return col in schema["schema_column_set"]


def repair_sir(sir: Dict[str, Any], schema: Dict[str, Any]) -> Dict[str, Any]:
    repaired = default_sir()
    repaired.update({k: v for k, v in sir.items() if k in repaired})

    for key in ["tables", "joins", "select", "filters", "group_by", "having", "order_by"]:
        if not isinstance(repaired[key], list):
            repaired[key] = []

    repaired["tables"] = [t for t in repaired["tables"] if isinstance(t, str) and is_valid_table(t, schema)]

    clean_joins = []
    for j in repaired["joins"]:
        if not isinstance(j, dict):
            continue
        left = j.get("left")
        right = j.get("right")
        if is_valid_column_reference(left, schema) and is_valid_column_reference(right, schema):
            clean_joins.append({"left": left, "right": right})
    repaired["joins"] = clean_joins

    clean_select = []
    for s in repaired["select"]:
        if not isinstance(s, dict):
            continue
        expr_type = s.get("type", "column")
        column = s.get("column", "*")
        agg = s.get("agg")
        alias = s.get("alias")
        distinct = bool(s.get("distinct", False))

        if expr_type == "star":
            clean_select.append({"type": "star", "column": "*", "agg": agg, "alias": alias, "distinct": distinct})
        elif is_valid_column_reference(column, schema):
            clean_select.append({"type": expr_type, "column": column, "agg": agg, "alias": alias, "distinct": distinct})
    repaired["select"] = clean_select

    clean_filters = []
    for f in repaired["filters"]:
        if not isinstance(f, dict):
            continue
        column = f.get("column")
        op = f.get("op")
        value = f.get("value")
        value_type = f.get("value_type", "literal")
        if is_valid_column_reference(column, schema) and isinstance(op, str):
            clean_filters.append({
                "column": column,
                "op": op,
                "value": value,
                "value_type": value_type,
            })
    repaired["filters"] = clean_filters

    repaired["group_by"] = [c for c in repaired["group_by"] if isinstance(c, str) and is_valid_column_reference(c, schema)]

    clean_having = []
    for h in repaired["having"]:
        if not isinstance(h, dict):
            continue
        column = h.get("column")
        agg = h.get("agg")
        op = h.get("op")
        value = h.get("value")
        if is_valid_column_reference(column, schema) and isinstance(op, str):
            clean_having.append({
                "column": column,
                "agg": agg,
                "op": op,
                "value": value,
            })
    repaired["having"] = clean_having

    clean_order_by = []
    for o in repaired["order_by"]:
        if not isinstance(o, dict):
            continue
        column = o.get("column")
        direction = str(o.get("direction", "ASC")).upper()
        agg = o.get("agg")
        if direction not in {"ASC", "DESC"}:
            direction = "ASC"
        if is_valid_column_reference(column, schema):
            clean_order_by.append({
                "column": column,
                "direction": direction,
                "agg": agg,
            })
    repaired["order_by"] = clean_order_by

    if repaired["limit"] is not None:
        try:
            repaired["limit"] = int(repaired["limit"])
            if repaired["limit"] <= 0:
                repaired["limit"] = None
        except Exception:
            repaired["limit"] = None

    inferred_tables = set(repaired["tables"])
    for section in ["select", "filters", "having", "order_by"]:
        for item in repaired[section]:
            if isinstance(item, dict):
                col = item.get("column")
                if isinstance(col, str) and "." in col:
                    inferred_tables.add(col.split(".", 1)[0])

    for c in repaired["group_by"]:
        if "." in c:
            inferred_tables.add(c.split(".", 1)[0])

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

    valid = len(invalid_tables) == 0 and len(invalid_columns) == 0
    sir_tables = set(sir.get("tables", []))
    sir_cols = set()

    for s in sir.get("select", []):
        if isinstance(s, dict) and isinstance(s.get("column"), str):
            sir_cols.add(s["column"])
    for f in sir.get("filters", []):
        if isinstance(f, dict) and isinstance(f.get("column"), str):
            sir_cols.add(f["column"])
    for g in sir.get("group_by", []):
        if isinstance(g, str):
            sir_cols.add(g)
    for h in sir.get("having", []):
        if isinstance(h, dict) and isinstance(h.get("column"), str):
            sir_cols.add(h["column"])
    for o in sir.get("order_by", []):
        if isinstance(o, dict) and isinstance(o.get("column"), str):
            sir_cols.add(o["column"])

    return {
        "sir_valid": valid,
        "sir_invalid_tables": sorted(set(invalid_tables)),
        "sir_invalid_columns": sorted(set(invalid_columns)),
        "sir_tables": sorted(sir_tables),
        "sir_columns": sorted(sir_cols),
    }


# =========================================================
# Prompting
# =========================================================

SIR_SYSTEM_PROMPT = """You are an expert semantic parser for SQLite text-to-SQL.

Your task is NOT to directly write SQL first.
Your first task is to produce a Symbolic Intermediate Representation (SIR) as JSON.

The SIR must capture:
- which tables are needed
- how tables connect
- what columns are selected
- filters
- grouping / aggregation
- ordering / limit
- optional set operations or subquery requirement

Return EXACTLY one JSON object and nothing else.

SIR schema:
{
  "task_type": "select | count | aggregation | existence | comparison | set_op",
  "tables": ["table1", "table2"],
  "joins": [
    {"left": "table1.col_a", "right": "table2.col_b"}
  ],
  "select": [
    {"type": "column | star", "column": "table.col or *", "agg": null, "alias": null, "distinct": false}
  ],
  "filters": [
    {"column": "table.col", "op": "= | != | > | >= | < | <= | like | in | not in", "value": "literal or placeholder", "value_type": "literal"}
  ],
  "group_by": ["table.col"],
  "having": [
    {"column": "table.col", "agg": "COUNT | AVG | MIN | MAX | SUM", "op": ">", "value": 1}
  ],
  "order_by": [
    {"column": "table.col", "direction": "ASC | DESC", "agg": null}
  ],
  "limit": null,
  "set_op": null,
  "subquery": null,
  "notes": "brief reasoning summary"
}

Rules:
- Use only tables and columns from the schema.
- Prefer qualified columns like table.column.
- If an aggregation is needed, represent it explicitly in select or having.
- If the question asks for a count, represent that clearly.
- Return JSON only.
"""


SQL_FROM_SIR_SYSTEM_PROMPT = """You are an expert SQLite SQL writer.

You will be given:
1. a database schema
2. a validated symbolic intermediate representation (SIR)

Your job is to write ONE SQLite SQL query that faithfully implements the SIR.

Rules:
- Use only schema-valid tables and columns.
- Respect foreign keys and joins.
- Return only SQL.
- No markdown.
- No explanation.
"""


def build_sir_user_prompt(schema_text: str, question: str) -> str:
    return f"""{schema_text}

Question:
{question}

Produce the SIR JSON now.
""".strip()


def build_sql_from_sir_user_prompt(schema_text: str, question: str, sir: Dict[str, Any]) -> str:
    sir_text = json.dumps(sir, indent=2, ensure_ascii=False)
    return f"""{schema_text}

Question:
{question}

Validated SIR:
{sir_text}

Write the final SQLite SQL query that implements the SIR.
SQL:
""".strip()


def extract_sql_from_response(text: str) -> str:
    text = strip_think_blocks(text)

    m = re.findall(r"```sql\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()

    m = re.findall(r"```[\w]*\s*([\s\S]*?)```", text, flags=re.IGNORECASE)
    if m:
        return m[-1].strip()

    m = re.search(r"\b(SELECT|WITH)\b[\s\S]*", text, flags=re.IGNORECASE)
    if m:
        return m.group(0).strip()

    return text.strip()


@torch.no_grad()
def run_chat(model, tokenizer, messages: List[Dict[str, str]], max_new_tokens: int) -> str:
    inputs = tokenizer.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        enable_thinking=False,
    ).to(model.device)

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )

    gen_tokens = outputs[0][inputs["input_ids"].shape[-1]:]
    return tokenizer.decode(gen_tokens, skip_special_tokens=True)


def generate_sir(model, tokenizer, schema_text: str, question: str, max_new_tokens: int = 512) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SIR_SYSTEM_PROMPT},
        {"role": "user", "content": build_sir_user_prompt(schema_text, question)},
    ]
    raw_text = run_chat(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    sir_obj, parse_error = parse_sir_json(raw_text)
    return {
        "raw_model_text": raw_text,
        "sir_obj": sir_obj,
        "sir_parse_error": parse_error,
    }


def generate_sql_from_sir(model, tokenizer, schema_text: str, question: str, sir: Dict[str, Any], max_new_tokens: int = 256) -> Dict[str, Any]:
    messages = [
        {"role": "system", "content": SQL_FROM_SIR_SYSTEM_PROMPT},
        {"role": "user", "content": build_sql_from_sir_user_prompt(schema_text, question, sir)},
    ]
    raw_text = run_chat(model, tokenizer, messages, max_new_tokens=max_new_tokens)
    pred_sql = extract_sql_from_response(raw_text)
    return {
        "raw_model_text": raw_text,
        "pred_sql": pred_sql,
    }


# =========================================================
# Stats
# =========================================================

class RunningStats:
    def __init__(self):
        self.total = 0
        self.exec_correct = 0
        self.sql_parse_ok = 0
        self.sql_exec_ok = 0
        self.sql_schema_valid = 0
        self.sql_norm_exact = 0
        self.sir_parse_ok = 0
        self.sir_valid = 0
        self.sir_table_match = 0
        self.sir_schema_valid_but_sql_wrong = 0
        self.schema_oriented_table_match = 0
        self.schema_oriented_column_exact = 0

    def update(self, row: Dict[str, Any]):
        self.total += 1
        self.exec_correct += int(row["execution_correct"])
        self.sql_parse_ok += int(row["pred_parse_ok"])
        self.sql_exec_ok += int(row["pred_exec_ok"])
        self.sql_schema_valid += int(row["pred_schema_valid"])
        self.sql_norm_exact += int(row["normalized_exact_match"])
        self.sir_parse_ok += int(row["sir_parse_ok"])
        self.sir_valid += int(row["sir_valid"])
        self.sir_table_match += int(row["sir_table_match"])
        self.sir_schema_valid_but_sql_wrong += int(row["sir_valid_but_sql_wrong"])
        self.schema_oriented_table_match += int(row["schema_oriented_table_match"])
        self.schema_oriented_column_exact += int(row["schema_oriented_column_exact"])

    def summary(self) -> Dict[str, Any]:
        n = max(1, self.total)
        return {
            "n_examples": self.total,
            "execution_accuracy": self.exec_correct / n,
            "normalized_exact_match": self.sql_norm_exact / n,
            "pred_parse_ok_rate": self.sql_parse_ok / n,
            "pred_execution_success_rate": self.sql_exec_ok / n,
            "pred_schema_valid_rate": self.sql_schema_valid / n,
            "schema_oriented_table_match_rate": self.schema_oriented_table_match / n,
            "schema_oriented_column_exact_rate": self.schema_oriented_column_exact / n,
            "sir_parse_ok_rate": self.sir_parse_ok / n,
            "sir_valid_rate": self.sir_valid / n,
            "sir_table_match_rate": self.sir_table_match / n,
            "sir_valid_but_sql_wrong_rate": self.sir_schema_valid_but_sql_wrong / n,
        }


# =========================================================
# Main
# =========================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_model_id", type=str, default=os.environ.get("MODEL_ID", "Qwen/Qwen3-4B"))
    parser.add_argument("--checkpoint_path", type=str, default=os.environ.get("CHECKPOINT_PATH", None), help="Optional path to trained checkpoint. Supports PEFT adapter or merged model.")
    parser.add_argument("--hf_dataset", type=str, default=os.environ.get("HF_DATASET", "xlangai/spider"))
    parser.add_argument("--split", type=str, default=os.environ.get("SPLIT", "validation"))
    parser.add_argument("--spider_dir", type=str, default=str(SPIDER_DIR_DEFAULT))
    parser.add_argument("--output_dir", type=str, default=str(OUTPUT_ROOT_DEFAULT))
    parser.add_argument("--sir_max_new_tokens", type=int, default=int(os.environ.get("SIR_MAX_NEW_TOKENS", 512)))
    parser.add_argument("--sql_max_new_tokens", type=int, default=int(os.environ.get("SQL_MAX_NEW_TOKENS", 256)))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--start_idx", type=int, default=0)
    parser.add_argument("--device_map", type=str, default=os.environ.get("DEVICE_MAP", "auto"))
    parser.add_argument("--torch_dtype", type=str, default=os.environ.get("TORCH_DTYPE", "auto"), choices=["auto", "float16", "bfloat16", "float32"])
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    spider_dir = Path(args.spider_dir).resolve()
    output_dir = Path(args.output_dir).resolve()
    ensure_dir(output_dir)

    predictions_jsonl = output_dir / "predictions.jsonl"
    predictions_csv = output_dir / "predictions.csv"
    summary_json = output_dir / "summary.json"

    if args.overwrite:
        for p in [predictions_jsonl, predictions_csv, summary_json]:
            if p.exists():
                p.unlink()

    if args.torch_dtype == "auto":
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
    elif args.torch_dtype == "float16":
        dtype = torch.float16
    elif args.torch_dtype == "bfloat16":
        dtype = torch.bfloat16
    else:
        dtype = torch.float32

    tables_json_path, db_root = validate_spider_dir(spider_dir)

    load_path_for_tokenizer = args.checkpoint_path if args.checkpoint_path else args.base_model_id

    logger.info("Project root: %s", PROJECT_ROOT)
    logger.info("Spider dir: %s", spider_dir)
    logger.info("Output dir: %s", output_dir)
    logger.info("Loading tokenizer from: %s", load_path_for_tokenizer)

    try:
        tokenizer = AutoTokenizer.from_pretrained(load_path_for_tokenizer, cache_dir=str(TRANSFORMERS_CACHE))
    except Exception:
        tokenizer = AutoTokenizer.from_pretrained(args.base_model_id, cache_dir=str(TRANSFORMERS_CACHE))

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if args.checkpoint_path is None:
        logger.info("Loading base model: %s", args.base_model_id)
        model = AutoModelForCausalLM.from_pretrained(
            args.base_model_id,
            cache_dir=str(TRANSFORMERS_CACHE),
            device_map=args.device_map,
            torch_dtype=dtype,
        )
    else:
        checkpoint_path = Path(args.checkpoint_path).resolve()
        adapter_config_path = checkpoint_path / "adapter_config.json"

        if adapter_config_path.exists():
            logger.info("Loading base model: %s", args.base_model_id)
            base_model = AutoModelForCausalLM.from_pretrained(
                args.base_model_id,
                cache_dir=str(TRANSFORMERS_CACHE),
                device_map=args.device_map,
                torch_dtype=dtype,
            )

            logger.info("Loading PEFT adapter checkpoint: %s", checkpoint_path)
            model = PeftModel.from_pretrained(base_model, str(checkpoint_path))
        else:
            logger.info("Loading full checkpoint as model: %s", checkpoint_path)
            model = AutoModelForCausalLM.from_pretrained(
                str(checkpoint_path),
                cache_dir=str(TRANSFORMERS_CACHE),
                device_map=args.device_map,
                torch_dtype=dtype,
            )

    model.eval()

    logger.info("Loading dataset split: %s/%s", args.hf_dataset, args.split)
    ds = load_dataset(args.hf_dataset, split=args.split, cache_dir=str(HF_DATASETS_CACHE))

    schemas = load_spider_tables_json(tables_json_path)

    total_len = len(ds)
    start = max(0, args.start_idx)
    end = total_len if args.limit is None else min(total_len, start + args.limit)

    logger.info("Running examples %s..%s out of %s", start, end - 1, total_len)

    stats = RunningStats()
    csv_rows = []

    for idx in tqdm(range(start, end), desc="Spider SIR inference"):
        ex = ds[idx]
        db_id = ex["db_id"]
        question = ex["question"]
        gold_sql = ex["query"]

        schema = schemas[db_id]
        schema_text = schema_to_prompt(schema)
        db_path = db_root / db_id / f"{db_id}.sqlite"
        if not db_path.exists():
            raise FileNotFoundError(f"Missing sqlite db: {db_path}")

        gold_schema = classify_schema_adherence(gold_sql, schema)
        gold_tables = set(gold_schema["tables"])
        gold_cols = set(gold_schema["columns"])

        sir_gen = generate_sir(
            model=model,
            tokenizer=tokenizer,
            schema_text=schema_text,
            question=question,
            max_new_tokens=args.sir_max_new_tokens,
        )
        sir_parse_ok = sir_gen["sir_obj"] is not None

        repaired_sir = repair_sir(sir_gen["sir_obj"], schema) if sir_parse_ok else default_sir()
        sir_metrics = classify_sir_validity(repaired_sir, schema)
        sir_tables = set(sir_metrics["sir_tables"])
        sir_cols = set(sir_metrics["sir_columns"])

        sir_table_scores = prf1(sir_tables, gold_tables)
        sir_col_scores = prf1(sir_cols, gold_cols)
        sir_table_match = sir_metrics["sir_valid"] and sir_tables == gold_tables

        sql_gen = generate_sql_from_sir(
            model=model,
            tokenizer=tokenizer,
            schema_text=schema_text,
            question=question,
            sir=repaired_sir,
            max_new_tokens=args.sql_max_new_tokens,
        )
        pred_sql = sql_gen["pred_sql"]

        pred_schema = classify_schema_adherence(pred_sql, schema)
        pred_tables = set(pred_schema["tables"])
        pred_cols = set(pred_schema["columns"])

        table_scores = prf1(pred_tables, gold_tables)
        col_scores = prf1(pred_cols, gold_cols)

        pred_exec = execute_sql(str(db_path), pred_sql)
        gold_exec = execute_sql(str(db_path), gold_sql)

        pred_exec_ok = pred_exec["ok"]
        gold_exec_ok = gold_exec["ok"]
        pred_exec_error_type = classify_execution_error(pred_exec["error"])

        execution_correct = False
        if pred_exec_ok and gold_exec_ok:
            order_sensitive = query_has_order_by(gold_sql) or query_has_order_by(pred_sql)
            pred_res = result_to_canonical(pred_exec["rows"], order_sensitive=order_sensitive)
            gold_res = result_to_canonical(gold_exec["rows"], order_sensitive=order_sensitive)
            execution_correct = pred_res == gold_res
        else:
            pred_res = None
            gold_res = None

        normalized_exact_match = normalize_sql_string(pred_sql) == normalize_sql_string(gold_sql)
        schema_oriented_table_match = pred_schema["schema_valid"] and pred_tables == gold_tables
        schema_oriented_column_exact = pred_schema["schema_valid"] and pred_cols == gold_cols
        sir_valid_but_sql_wrong = sir_metrics["sir_valid"] and not execution_correct

        row = {
            "idx": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "sir_raw_model_text": sir_gen["raw_model_text"],
            "sir_parse_ok": sir_parse_ok,
            "sir_parse_error": sir_gen["sir_parse_error"],
            "sir_raw_obj": sir_gen["sir_obj"],
            "sir_repaired_obj": repaired_sir,
            "sir_valid": sir_metrics["sir_valid"],
            "sir_invalid_tables": sir_metrics["sir_invalid_tables"],
            "sir_invalid_columns": sir_metrics["sir_invalid_columns"],
            "sir_tables": sir_metrics["sir_tables"],
            "sir_columns": sir_metrics["sir_columns"],
            "sir_table_precision": sir_table_scores["precision"],
            "sir_table_recall": sir_table_scores["recall"],
            "sir_table_f1": sir_table_scores["f1"],
            "sir_column_precision": sir_col_scores["precision"],
            "sir_column_recall": sir_col_scores["recall"],
            "sir_column_f1": sir_col_scores["f1"],
            "sir_table_match": sir_table_match,
            "sir_valid_but_sql_wrong": sir_valid_but_sql_wrong,
            "sql_raw_model_text": sql_gen["raw_model_text"],
            "pred_sql": pred_sql,
            "pred_parse_ok": pred_schema["parse_ok"],
            "pred_exec_ok": pred_exec_ok,
            "pred_exec_error": pred_exec["error"],
            "pred_exec_error_type": pred_exec_error_type,
            "pred_schema_valid": pred_schema["schema_valid"],
            "pred_invalid_tables": pred_schema["invalid_tables"],
            "pred_invalid_columns": pred_schema["invalid_columns"],
            "gold_tables": sorted(gold_tables),
            "pred_tables": sorted(pred_tables),
            "gold_columns": sorted(gold_cols),
            "pred_columns": sorted(pred_cols),
            "table_precision": table_scores["precision"],
            "table_recall": table_scores["recall"],
            "table_f1": table_scores["f1"],
            "table_jaccard": table_scores["jaccard"],
            "column_precision": col_scores["precision"],
            "column_recall": col_scores["recall"],
            "column_f1": col_scores["f1"],
            "column_jaccard": col_scores["jaccard"],
            "schema_oriented_table_match": schema_oriented_table_match,
            "schema_oriented_column_exact": schema_oriented_column_exact,
            "normalized_exact_match": normalized_exact_match,
            "execution_correct": execution_correct,
            "gold_result": gold_res,
            "pred_result": pred_res,
        }

        stats.update(row)
        append_jsonl(predictions_jsonl, row)

        csv_rows.append({
            "idx": idx,
            "db_id": db_id,
            "question": question,
            "gold_sql": gold_sql,
            "pred_sql": pred_sql,
            "sir_parse_ok": sir_parse_ok,
            "sir_valid": sir_metrics["sir_valid"],
            "sir_table_match": sir_table_match,
            "pred_parse_ok": pred_schema["parse_ok"],
            "pred_exec_ok": pred_exec_ok,
            "pred_exec_error_type": pred_exec_error_type,
            "pred_schema_valid": pred_schema["schema_valid"],
            "schema_oriented_table_match": schema_oriented_table_match,
            "schema_oriented_column_exact": schema_oriented_column_exact,
            "normalized_exact_match": normalized_exact_match,
            "execution_correct": execution_correct,
            "sir_table_f1": sir_table_scores["f1"],
            "sir_column_f1": sir_col_scores["f1"],
            "table_f1": table_scores["f1"],
            "column_f1": col_scores["f1"],
            "pred_invalid_tables": ";".join(pred_schema["invalid_tables"]),
            "pred_invalid_columns": ";".join(pred_schema["invalid_columns"]),
            "sir_invalid_tables": ";".join(sir_metrics["sir_invalid_tables"]),
            "sir_invalid_columns": ";".join(sir_metrics["sir_invalid_columns"]),
        })

        if (idx - start + 1) % 25 == 0:
            safe_json_dump(stats.summary(), summary_json)

    with predictions_csv.open("w", encoding="utf-8", newline="") as f:
        if csv_rows:
            writer = csv.DictWriter(f, fieldnames=list(csv_rows[0].keys()))
            writer.writeheader()
            writer.writerows(csv_rows)

    final_summary = {
        "base_model_id": args.base_model_id,
        "checkpoint_path": args.checkpoint_path,
        "hf_dataset": args.hf_dataset,
        "split": args.split,
        "spider_dir": str(spider_dir),
        "output_dir": str(output_dir),
        "start_idx": start,
        "end_idx": end,
        "sir_max_new_tokens": args.sir_max_new_tokens,
        "sql_max_new_tokens": args.sql_max_new_tokens,
        "timestamp": now_ts(),
        "metrics": stats.summary(),
    }
    safe_json_dump(final_summary, summary_json)

    print("\nDone.")
    print(json.dumps(final_summary, indent=2))


if __name__ == "__main__":
    main()
