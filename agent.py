"""
Base agent loop: plan → select tool → execute → observe → repeat.
Returns a Trajectory for every task episode.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import anthropic

from scaffold import ScaffoldVersion, ToolSpec
from tools import BUILTIN_DISPATCH, run_dynamic_tool, memory_clear

AGENT_MODEL = "claude-sonnet-4-6"
MAX_TOKENS_PER_STEP = 2048


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
    task_text: str
    correct_answer: str
    steps: List[Step] = field(default_factory=list)
    final_answer: str = ""
    success: bool = False
    tokens_used: int = 0
    elapsed_s: float = 0.0
    error: Optional[str] = None


def _tool_definitions(scaffold: ScaffoldVersion) -> List[Dict[str, Any]]:
    defs = []
    for t in scaffold.tools:
        if t.enabled:
            defs.append({
                "name": t.name,
                "description": t.description,
                "input_schema": t.parameters,
            })
    return defs


def _execute_tool(name: str, inputs: Dict[str, Any], scaffold: ScaffoldVersion) -> str:
    # Built-in tools
    if name in BUILTIN_DISPATCH:
        try:
            return str(BUILTIN_DISPATCH[name](**inputs))
        except Exception as e:
            return f"ToolError({name}): {e}"

    # Dynamically-added tools
    for spec in scaffold.tools:
        if spec.name == name and spec.implementation != "builtin":
            return run_dynamic_tool(spec.implementation, name, inputs)

    return f"Error: tool '{name}' not found"


def _extract_text(content: List[Any]) -> str:
    parts = []
    for block in content:
        if hasattr(block, "type") and block.type == "text":
            parts.append(block.text)
    return " ".join(parts).strip()


def run_episode(
    task_id: str,
    task_text: str,
    correct_answer: str,
    scaffold: ScaffoldVersion,
    client: anthropic.Anthropic,
    evaluator,
) -> Trajectory:
    """Run one agent episode on a single task."""
    memory_clear()
    traj = Trajectory(task_id=task_id, task_text=task_text, correct_answer=correct_answer)
    t0 = time.time()

    messages = [{"role": "user", "content": task_text}]
    tool_defs = _tool_definitions(scaffold)
    max_steps = scaffold.planning_policy.max_steps

    try:
        for step_num in range(max_steps):
            step_t0 = time.time()
            resp = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=MAX_TOKENS_PER_STEP,
                system=scaffold.planning_policy.system_prompt,
                tools=tool_defs,
                messages=messages,
                temperature=scaffold.planning_policy.temperature,
            )
            traj.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
            latency = (time.time() - step_t0) * 1000

            assistant_text = _extract_text(resp.content)

            if resp.stop_reason == "end_turn":
                traj.steps.append(Step(
                    step_num=step_num,
                    tool_name=None,
                    tool_input=None,
                    tool_output=None,
                    assistant_text=assistant_text,
                    latency_ms=latency,
                ))
                traj.final_answer = assistant_text
                break

            elif resp.stop_reason == "tool_use":
                messages.append({"role": "assistant", "content": resp.content})
                tool_results = []
                step_tools = []

                for block in resp.content:
                    if not (hasattr(block, "type") and block.type == "tool_use"):
                        continue
                    tool_out = _execute_tool(block.name, block.input, scaffold)
                    tool_results.append({
                        "type": "tool_result",
                        "tool_use_id": block.id,
                        "content": str(tool_out),
                    })
                    step_tools.append((block.name, block.input, tool_out))

                messages.append({"role": "user", "content": tool_results})

                # Record one Step per tool call (first call captures them all)
                for i, (tname, tinput, tout) in enumerate(step_tools):
                    traj.steps.append(Step(
                        step_num=step_num,
                        tool_name=tname,
                        tool_input=tinput,
                        tool_output=str(tout),
                        assistant_text=assistant_text if i == 0 else "",
                        latency_ms=latency if i == 0 else 0,
                    ))
            else:
                traj.final_answer = assistant_text
                break
        else:
            # Exhausted steps — do one final no-tool call
            messages.append({"role": "user", "content": "Please provide your final answer now."})
            resp = client.messages.create(
                model=AGENT_MODEL,
                max_tokens=512,
                system=scaffold.planning_policy.system_prompt,
                messages=messages,
            )
            traj.tokens_used += resp.usage.input_tokens + resp.usage.output_tokens
            traj.final_answer = _extract_text(resp.content)

    except anthropic.AuthenticationError as e:
        raise RuntimeError(
            "Anthropic API key not set. Export ANTHROPIC_API_KEY before running."
        ) from e
    except Exception as e:
        traj.error = str(e)
        traj.final_answer = ""
        print(f"    [agent error] {type(e).__name__}: {e}")

    traj.elapsed_s = time.time() - t0
    traj.success = evaluator(traj.final_answer, correct_answer)
    return traj
