"""
Built-in tool implementations + dynamic tool runner.
All code that executes arbitrary strings runs in a subprocess with timeout.
"""
from __future__ import annotations

import ast
import json
import math
import subprocess
import sys
import tempfile
import textwrap
from pathlib import Path
from typing import Any, Dict

# ── calculator ────────────────────────────────────────────────────────────────

_MATH_NAMES: Dict[str, Any] = {k: v for k, v in math.__dict__.items() if not k.startswith("_")}
_MATH_NAMES.update({"abs": abs, "round": round, "min": min, "max": max, "sum": sum, "pow": pow})


def calculator(expression: str) -> str:
    try:
        tree = ast.parse(expression.strip(), mode="eval")
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                if node.func.id not in _MATH_NAMES:
                    return f"Error: function '{node.func.id}' not allowed"
        result = eval(compile(tree, "<expr>", "eval"), {"__builtins__": {}}, _MATH_NAMES)
        if isinstance(result, float) and result == int(result) and abs(result) < 1e15:
            result = round(result, 10)  # trim float noise
        return str(result)
    except Exception as e:
        return f"Error: {e}"


# ── web search (stub knowledge base) ─────────────────────────────────────────

_KB: Dict[str, str] = {
    "leap year days": "366",
    "days in a leap year": "366",
    "boiling point of water fahrenheit": "212",
    "boiling point water fahrenheit": "212",
    "boiling point of water in fahrenheit": "212 degrees Fahrenheit",
    "chemical symbol for gold": "Au",
    "gold chemical symbol": "Au",
    "bytes in a kilobyte": "1024",
    "kilobyte bytes": "1024",
    "largest planet solar system": "Jupiter",
    "largest planet": "Jupiter",
    "speed of light": "299792458 meters per second",
    "speed of light meters per second": "299792458",
    "speed of light m/s": "299792458",
    "fibonacci sequence": "0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144...",
    "first 8 fibonacci numbers": "0, 1, 1, 2, 3, 5, 8, 13",
    "capital of france": "Paris",
    "eiffel tower location": "Paris, France",
    "value of pi": "3.14159265358979",
    "pi": "3.14159265358979",
    "euler number e": "2.71828182845905",
    "day of week january 1 2025": "Wednesday",
    "january 1 2025 day": "Wednesday",
    "what day is january 1 2025": "Wednesday",
    "pentagon sides": "5",
    "sides of a pentagon": "5",
    "base64 encoding": "A method to encode binary data as ASCII text using 64 characters",
    "hexadecimal ff decimal": "255",
    "ff hexadecimal to decimal": "255",
    "levenshtein distance": "A measure of difference between two strings (edit distance)",
    "collatz conjecture": "A sequence where n is halved if even, tripled+1 if odd, always reaches 1",
}


def web_search(query: str) -> str:
    q = query.lower().strip().rstrip("?.")
    if q in _KB:
        return _KB[q]
    for key, val in _KB.items():
        if key in q or q in key:
            return f"{val}"
    return f"No results found for: '{query}'. Try rephrasing or use python_exec to compute it."


# ── python executor ───────────────────────────────────────────────────────────

_TIMEOUT = 10  # seconds


def python_exec(code: str) -> str:
    wrapped = textwrap.dedent(f"""\
import sys, io as _io
_buf = _io.StringIO()
_orig = sys.stdout
sys.stdout = _buf
try:
{textwrap.indent(code, '    ')}
finally:
    sys.stdout = _orig
print(_buf.getvalue(), end='')
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(wrapped)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        out = r.stdout.strip()
        err = r.stderr.strip()
        if r.returncode != 0:
            return f"RuntimeError:\n{err[:600]}"
        return out if out else "(no output)"
    except subprocess.TimeoutExpired:
        return f"Error: timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── memory ────────────────────────────────────────────────────────────────────

_MEMORY: Dict[str, str] = {}


def memory_write(key: str, value: str) -> str:
    _MEMORY[key] = value
    return f"Stored '{key}'"


def memory_read(key: str) -> str:
    return _MEMORY.get(key, f"Key '{key}' not found in memory")


def memory_clear() -> None:
    _MEMORY.clear()


# ── dynamic tool runner ───────────────────────────────────────────────────────

def run_dynamic_tool(implementation: str, func_name: str, kwargs: Dict[str, Any]) -> str:
    """Execute a dynamically-defined tool in a sandboxed subprocess."""
    args_json = json.dumps(kwargs)
    script = textwrap.dedent(f"""\
import sys, json

{implementation}

args = json.loads(sys.argv[1])
try:
    result = {func_name}(**args)
    print(json.dumps({{"result": str(result)}}))
except Exception as e:
    print(json.dumps({{"error": str(e)}}))
""")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(script)
        tmp = f.name
    try:
        r = subprocess.run(
            [sys.executable, tmp, args_json],
            capture_output=True, text=True, timeout=_TIMEOUT,
        )
        if r.returncode != 0:
            return f"DynamicToolError:\n{r.stderr[:400]}"
        payload = json.loads(r.stdout.strip())
        if "error" in payload:
            return f"ToolError: {payload['error']}"
        return payload["result"]
    except subprocess.TimeoutExpired:
        return f"Error: dynamic tool timed out after {_TIMEOUT}s"
    except Exception as e:
        return f"Error: {e}"
    finally:
        Path(tmp).unlink(missing_ok=True)


# ── dispatch table ────────────────────────────────────────────────────────────

BUILTIN_DISPATCH = {
    "calculator": calculator,
    "web_search": web_search,
    "python_exec": python_exec,
    "memory_write": memory_write,
    "memory_read": memory_read,
}
