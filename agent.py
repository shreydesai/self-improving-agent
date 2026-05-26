"""
Haiku agent loop with limited context.
max_tokens_per_step and max_steps are kept small so the agent is
forced to use tools rather than reasoning from a large context.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import anthropic

import tools as tool_module
from scaffold import ScaffoldVersion, ToolSpec

AGENT_MODEL = "claude-haiku-4-5-20251001"


@dataclass
class Step:
    step_num: int
    tool_name: Optional[str]
    tool_input: Optional[Dict[str, Any]]
    tool_output: Optional[str]
    assistant_text: str
    latency_ms: float


@dataclass
class Trajectory:
    task_id: str
    question: str
    correct_answer: str
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    tokens_used: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None


# ── tool execution ────────────────────────────────────────────────────────────

def _execute(spec: ToolSpec, inputs: Dict[str, Any], data_dir: Path) -> str:
    if spec.source == "builtin":
        fn = getattr(tool_module, spec.name, None)
        if fn is None:
            return f"Error: built-in '{spec.name}' not found"
        try:
            # built-ins that need data_dir accept it as kwarg
            import inspect
            sig = inspect.signature(fn)
            if "data_dir" in sig.parameters:
                return str(fn(**inputs, data_dir=data_dir))
            return str(fn(**inputs))
        except Exception as e:
            return f"ToolError({spec.name}): {e}"

    if spec.source == "library":
        entry = tool_module.TOOL_LIBRARY.get(spec.name)
        if not entry:
            return f"Error: library tool '{spec.name}' not found"
        try:
            return str(entry["fn"](**inputs, data_dir=data_dir))
        except Exception as e:
            return f"ToolError({spec.name}): {e}"

    # dynamic
    return tool_module.run_dynamic_tool(spec.implementation, spec.name, inputs, data_dir)


# ── schema context injected into system prompt ────────────────────────────────

def _schema_context(data_dir: Path) -> str:
    import pandas as pd
    lines = ["\nAvailable data files:"]
    for f in sorted(data_dir.glob("*.csv")):
        n = sum(1 for _ in open(f)) - 1
        df = pd.read_csv(f, nrows=3)
        col_info = ", ".join(
            f"{c}({df[c].dtype})" for c in df.columns
        )
        lines.append(f"  {f.name}  ({n:,} rows)  [{col_info}]")
    return "\n".join(lines)


# ── evaluator ─────────────────────────────────────────────────────────────────

import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().strip().rstrip("."))


def evaluate(predicted: str, correct: str) -> bool:
    pn, cn = _norm(predicted), _norm(correct)
    if cn == pn or cn in pn:
        return True
    # strip commas and recheck
    if cn.replace(",", "") in pn.replace(",", ""):
        return True
    # numeric
    try:
        c = float(correct.replace(",", ""))
        for m in re.finditer(r"-?\d[\d,]*\.?\d*", predicted):
            if abs(float(m.group().replace(",", "")) - c) <= max(1e-4, abs(c) * 1e-4):
                return True
    except ValueError:
        pass
    return False


# ── episode ───────────────────────────────────────────────────────────────────

def run_episode(
    task_id: str,
    question: str,
    correct_answer: str,
    scaffold: ScaffoldVersion,
    data_dir: Path,
    client: anthropic.Anthropic,
) -> Trajectory:
    traj = Trajectory(task_id=task_id, question=question, correct_answer=correct_answer)
    t0 = time.time()

    policy  = scaffold.planning_policy
    schema  = _schema_context(data_dir)
    system  = policy.system_prompt + schema

    tool_defs = [
        {"name": t.name, "description": t.description, "input_schema": t.parameters}
        for t in scaffold.tools
    ]

    messages = [{"role": "user", "content": question}]

    try:
        for step_num in range(policy.max_steps):
            st0 = time.time()
            resp = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=policy.max_tokens_per_step,
                system=system,
                tools=tool_defs,
                messages=messages,
            )
            traj.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
            latency = (time.time() - st0) * 1000

            text = " ".join(
                b.text for b in resp.content
                if hasattr(b, "type") and b.type == "text"
            ).strip()

            if resp.stop_reason == "end_turn":
                traj.steps.append(Step(step_num, None, None, None, text, latency))
                traj.final_answer = text
                break

            if resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                results = []
                for b in resp.content:
                    if not (hasattr(b, "type") and b.type == "tool_use"):
                        continue
                    spec = next((t for t in scaffold.tools if t.name == b.name), None)
                    if spec is None:
                        out = f"Error: unknown tool '{b.name}'"
                    else:
                        out = _execute(spec, b.input, data_dir)
                    # truncate long outputs so context stays small
                    out_trimmed = out[:400] if len(out) > 400 else out
                    results.append({
                        "type": "tool_result",
                        "tool_use_id": b.id,
                        "content": out_trimmed,
                    })
                    traj.steps.append(Step(step_num, b.name, b.input, out_trimmed, text, latency))
                    latency = 0
                messages.append({"role": "user", "content": results})
        else:
            # force final answer
            messages.append({"role": "user", "content": "Provide your final answer now."})
            resp = client.messages.create(
                model=AGENT_MODEL, max_tokens=128, system=system, messages=messages
            )
            traj.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
            traj.final_answer = " ".join(
                b.text for b in resp.content if hasattr(b, "type") and b.type == "text"
            ).strip()

    except anthropic.AuthenticationError as e:
        raise RuntimeError("Set ANTHROPIC_API_KEY before running.") from e
    except Exception as e:
        traj.error = str(e)
        print(f"    [agent error] {type(e).__name__}: {e}")

    traj.elapsed_s = time.time() - t0
    traj.success = evaluate(traj.final_answer, correct_answer)
    return traj
