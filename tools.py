"""
Tool implementations (pandas-backed) + a pre-built library the meta-agent
can unlock by name. All tools receive data_dir at call time — they are
stateless and safe to call concurrently.

Built-in starting tools: list_files, read_sample
Pre-built library (unlockable): filter_count, column_sum, column_stats,
  get_distinct, group_aggregate, sort_top_k, join_tables
"""
from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import textwrap
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd

# ── file cache ────────────────────────────────────────────────────────────────

@lru_cache(maxsize=16)
def _load(path_str: str) -> pd.DataFrame:
    return pd.read_csv(path_str)


def _df(file: str, data_dir: Path) -> pd.DataFrame:
    # Also check the in-process join cache
    if file in _JOIN_CACHE:
        return _JOIN_CACHE[file]
    return _load(str(data_dir / file))


_JOIN_CACHE: Dict[str, pd.DataFrame] = {}


def _apply_filters(df: pd.DataFrame, filters: List[Dict]) -> pd.DataFrame:
    for f in filters:
        col, op, val = f["column"], f["op"], f["value"]
        if op == "=":
            df = df[df[col] == val]
        elif op == "!=":
            df = df[df[col] != val]
        elif op == ">":
            df = df[df[col] > val]
        elif op == "<":
            df = df[df[col] < val]
        elif op == ">=":
            df = df[df[col] >= val]
        elif op == "<=":
            df = df[df[col] <= val]
        elif op == "contains":
            df = df[df[col].astype(str).str.contains(str(val), na=False)]
    return df


# ── starting tools (always in scaffold) ──────────────────────────────────────

def list_files(data_dir: Path) -> str:
    files = sorted(data_dir.glob("*.csv"))
    if not files:
        return "No data files found."
    lines = []
    for f in files:
        n = sum(1 for _ in open(f)) - 1
        df = pd.read_csv(f, nrows=0)
        lines.append(f"{f.name}  ({n:,} rows)  columns: {', '.join(df.columns)}")
    return "\n".join(lines)


def read_sample(file: str, n_rows: int = 10, *, data_dir: Path) -> str:
    df = _df(file, data_dir).head(n_rows)
    return df.to_string(index=False, max_cols=20)


# ── pre-built library tools ───────────────────────────────────────────────────

def filter_count(file: str, filters: List[Dict], *, data_dir: Path) -> str:
    """Count rows matching all filter conditions."""
    df = _apply_filters(_df(file, data_dir), filters)
    return str(len(df))


def column_sum(file: str, column: str, filters: Optional[List[Dict]] = None, *,
               data_dir: Path) -> str:
    """Sum a numeric column, optionally after filtering."""
    df = _df(file, data_dir)
    if filters:
        df = _apply_filters(df, filters)
    return str(round(float(df[column].sum()), 2))


def column_stats(file: str, column: str, *, data_dir: Path) -> str:
    """Return count, min, max, mean, and null count for a column."""
    s = _df(file, data_dir)[column]
    return json.dumps({
        "count":  int(s.count()),
        "min":    round(float(s.min()), 4) if s.dtype != object else str(s.min()),
        "max":    round(float(s.max()), 4) if s.dtype != object else str(s.max()),
        "mean":   round(float(s.mean()), 4) if s.dtype != object else "n/a",
        "nulls":  int(s.isna().sum()),
    })


def get_distinct(file: str, column: str, *, data_dir: Path) -> str:
    """Return all unique values of a column (up to 50)."""
    vals = sorted(_df(file, data_dir)[column].dropna().unique().tolist())
    if len(vals) > 50:
        return f"{vals[:50]} ... ({len(vals)} total)"
    return str(vals)


def group_aggregate(file: str, group_col: str, agg_col: str,
                    agg_fn: str = "sum",
                    filters: Optional[List[Dict]] = None, *,
                    data_dir: Path) -> str:
    """Group by a column and aggregate another. agg_fn: sum|count|mean|min|max."""
    df = _df(file, data_dir)
    if filters:
        df = _apply_filters(df, filters)
    result = df.groupby(group_col)[agg_col].agg(agg_fn).reset_index()
    result.columns = [group_col, f"{agg_fn}_{agg_col}"]
    result = result.sort_values(f"{agg_fn}_{agg_col}", ascending=False)
    return result.to_string(index=False)


def sort_top_k(file: str, sort_col: str, k: int = 5,
               ascending: bool = False,
               filters: Optional[List[Dict]] = None, *,
               data_dir: Path) -> str:
    """Return top-k rows sorted by a column."""
    df = _df(file, data_dir)
    if filters:
        df = _apply_filters(df, filters)
    return df.sort_values(sort_col, ascending=ascending).head(k).to_string(index=False)


def join_tables(left_file: str, right_file: str, on: str,
                how: str = "inner",
                result_name: Optional[str] = None, *,
                data_dir: Path) -> str:
    """Join two tables. Optionally cache result as result_name for further ops."""
    left  = _df(left_file,  data_dir)
    right = _df(right_file, data_dir)
    joined = left.merge(right, on=on, how=how, suffixes=("", "_r"))
    if result_name:
        _JOIN_CACHE[result_name] = joined
        return (f"Joined {left_file} ✕ {right_file} on '{on}' ({how}). "
                f"Result '{result_name}': {len(joined):,} rows, "
                f"columns: {', '.join(joined.columns)}")
    return joined.head(5).to_string(index=False) + f"\n... ({len(joined):,} rows total)"


# ── dynamic tool runner (for meta-agent generated code) ──────────────────────

_TIMEOUT = 10


def run_dynamic_tool(implementation: str, func_name: str,
                     kwargs: Dict[str, Any], data_dir: Path) -> str:
    args = json.dumps({"data_dir": str(data_dir), **kwargs})
    script = textwrap.dedent(f"""\
import sys, json
from pathlib import Path

{implementation}

raw = json.loads(sys.argv[1])
data_dir = Path(raw.pop("data_dir"))
try:
    result = {func_name}(**raw, data_dir=data_dir)
    print(json.dumps({{"result": str(result)}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        r = subprocess.run([sys.executable, tmp, args],
                           capture_output=True, text=True, timeout=_TIMEOUT)
        if r.returncode != 0:
            return f"ToolError:\n{r.stderr[:400]}"
        payload = json.loads(r.stdout.strip())
        return payload.get("result", f"Error: {payload.get('error')}")
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── pre-built library manifest (shown to meta-agent) ─────────────────────────

TOOL_LIBRARY: Dict[str, Dict] = {
    "filter_count": {
        "fn": filter_count,
        "description": (
            "Count rows in a CSV file that match ALL given filter conditions. "
            "Use this whenever you need an exact count with criteria."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file":    {"type": "string", "description": "CSV filename"},
                "filters": {
                    "type": "array",
                    "description": "List of filter conditions (all must match)",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op":     {"type": "string",
                                       "enum": ["=", "!=", ">", "<", ">=", "<=", "contains"]},
                            "value":  {"description": "string or number"},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
            },
            "required": ["file", "filters"],
        },
    },
    "column_sum": {
        "fn": column_sum,
        "description": (
            "Sum a numeric column in a CSV file, with optional filter conditions. "
            "Returns the rounded total."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file":    {"type": "string"},
                "column":  {"type": "string", "description": "Column to sum"},
                "filters": {
                    "type": "array",
                    "description": "Optional filter conditions",
                    "items": {
                        "type": "object",
                        "properties": {
                            "column": {"type": "string"},
                            "op":     {"type": "string",
                                       "enum": ["=", "!=", ">", "<", ">=", "<="]},
                            "value":  {},
                        },
                        "required": ["column", "op", "value"],
                    },
                },
            },
            "required": ["file", "column"],
        },
    },
    "column_stats": {
        "fn": column_stats,
        "description": "Return count, min, max, mean, and null count for a column.",
        "parameters": {
            "type": "object",
            "properties": {
                "file":   {"type": "string"},
                "column": {"type": "string"},
            },
            "required": ["file", "column"],
        },
    },
    "get_distinct": {
        "fn": get_distinct,
        "description": "Return all unique values of a column (up to 50).",
        "parameters": {
            "type": "object",
            "properties": {
                "file":   {"type": "string"},
                "column": {"type": "string"},
            },
            "required": ["file", "column"],
        },
    },
    "group_aggregate": {
        "fn": group_aggregate,
        "description": (
            "Group a CSV by one column and aggregate another. "
            "agg_fn: sum | count | mean | min | max. "
            "Results are sorted descending by the aggregated value."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "group_col": {"type": "string"},
                "agg_col":   {"type": "string"},
                "agg_fn":    {"type": "string",
                               "enum": ["sum", "count", "mean", "min", "max"]},
                "filters":   {"type": "array", "items": {"type": "object"}},
            },
            "required": ["file", "group_col", "agg_col", "agg_fn"],
        },
    },
    "sort_top_k": {
        "fn": sort_top_k,
        "description": "Return the top-k rows sorted by a column.",
        "parameters": {
            "type": "object",
            "properties": {
                "file":      {"type": "string"},
                "sort_col":  {"type": "string"},
                "k":         {"type": "integer", "default": 5},
                "ascending": {"type": "boolean", "default": False},
                "filters":   {"type": "array", "items": {"type": "object"}},
            },
            "required": ["file", "sort_col"],
        },
    },
    "join_tables": {
        "fn": join_tables,
        "description": (
            "Join two CSV files on a shared column. "
            "Provide result_name to cache for further operations."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "left_file":   {"type": "string"},
                "right_file":  {"type": "string"},
                "on":          {"type": "string"},
                "how":         {"type": "string",
                                "enum": ["inner", "left", "right", "outer"],
                                "default": "inner"},
                "result_name": {"type": "string",
                                "description": "Name to cache the result under"},
            },
            "required": ["left_file", "right_file", "on"],
        },
    },
}

STARTING_TOOLS = {"list_files", "read_sample"}
