"""
Meta-agent: reads failure trajectories → proposes scaffold mutations.
Knows about the pre-built tool library so it can unlock tools by name
rather than having to write pandas code from scratch.
"""
from __future__ import annotations

import copy
import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import anthropic

from agent import Trajectory
from scaffold import ScaffoldVersion, ToolSpec, PlanningPolicy, _vid

META_MODEL = "claude-opus-4-7"
MutationType = Literal["add_tool", "modify_tool", "rewrite_planning_policy", "remove_tool"]


@dataclass
class Mutation:
    mutation_type: MutationType
    rationale: str
    details: Dict[str, Any]


# ── tool library summary shown to meta-agent ──────────────────────────────────

def _library_summary() -> str:
    from tools import TOOL_LIBRARY
    lines = ["Pre-built tools you can add (use exact names):"]
    for name, spec in TOOL_LIBRARY.items():
        lines.append(f"  {name}: {spec['description'][:100]}")
    return "\n".join(lines)


# ── meta-agent system prompt ──────────────────────────────────────────────────

_META_SYSTEM = """\
You are a meta-agent that improves a data-analysis agent's tool scaffold.

The agent answers questions about CSV files. It can ONLY access data through
tool calls — it cannot load files directly. When the agent lacks the right tool,
it fails or guesses incorrectly.

{library}

MUTATION TYPES:
- add_tool: unlock a tool from the pre-built library OR provide custom Python code
- modify_tool: update an existing tool's description
- rewrite_planning_policy: rewrite the system prompt (max 600 chars)
- remove_tool: remove a tool that causes confusion or errors

OUTPUT FORMAT — respond ONLY with valid JSON:
{{
  "analysis": "1-2 sentences on the dominant failure pattern",
  "proposals": [
    {{
      "mutation_type": "add_tool",
      "rationale": "why this tool is needed",
      "details": {{
        "name": "filter_count",
        "source": "library"
      }}
    }},
    {{
      "mutation_type": "add_tool",
      "rationale": "custom tool not in library",
      "details": {{
        "name": "my_tool",
        "source": "dynamic",
        "description": "what it does",
        "parameters": {{"type": "object", "properties": {{"x": {{"type": "string"}}}}, "required": ["x"]}},
        "implementation": "def my_tool(x, *, data_dir):\\n    return x.upper()"
      }}
    }},
    {{
      "mutation_type": "rewrite_planning_policy",
      "rationale": "...",
      "details": {{
        "new_system_prompt": "full replacement prompt (max 600 chars)"
      }}
    }}
  ]
}}

RULES:
- Propose at most 2 mutations per round.
- For add_tool with source=library: only specify name + source.
  The description and parameters are pulled from the library automatically.
- Only propose mutations that directly address observed failures.
- Do not propose tools the scaffold already has.
"""


# ── failure summary ───────────────────────────────────────────────────────────

def _failure_summary(trajectories: List[Trajectory]) -> str:
    failures = [t for t in trajectories if not t.success]
    successes = [t for t in trajectories if t.success]
    lines = [
        f"Tasks: {len(trajectories)}  passed: {len(successes)}  failed: {len(failures)}",
        "",
        "=== FAILURES ===",
    ]
    for t in failures[:10]:
        lines.append(f"\nTask [{t.task_id}]: {t.question[:120]}")
        lines.append(f"  Expected: {t.correct_answer}")
        lines.append(f"  Got:      {t.final_answer[:120]}")
        tools_used = [s.tool_name for s in t.steps if s.tool_name]
        lines.append(f"  Tools called: {tools_used}")
        if t.steps:
            last = t.steps[-1]
            if last.tool_output:
                lines.append(f"  Last output: {last.tool_output[:150]}")
        if t.error:
            lines.append(f"  Error: {t.error}")
    return "\n".join(lines)


def _scaffold_summary(s: ScaffoldVersion) -> str:
    lines = [f"Current tools ({len(s.tools)}):"]
    for t in s.tools:
        lines.append(f"  {t.name} [{t.source}]: {t.description[:80]}")
    return "\n".join(lines)


# ── propose ───────────────────────────────────────────────────────────────────

def propose(
    scaffold: ScaffoldVersion,
    trajectories: List[Trajectory],
    client: anthropic.Anthropic,
    max_mutations: int = 2,
) -> List[Mutation]:
    failures = [t for t in trajectories if not t.success]
    if not failures:
        print("  [meta] no failures — skipping")
        return []

    system = _META_SYSTEM.format(library=_library_summary())
    user_msg = "\n\n".join([
        _scaffold_summary(scaffold),
        _failure_summary(trajectories),
    ])

    resp = client.messages.create(
        model=META_MODEL,
        max_tokens=2048,
        system=system,
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = resp.content[0].text if resp.content else ""
    mutations = _parse(raw)[:max_mutations]
    print(f"  [meta] proposed {len(mutations)} mutation(s)")
    return mutations


def _parse(raw: str) -> List[Mutation]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return []
    try:
        data = json.loads(m.group())
    except json.JSONDecodeError:
        return []
    out = []
    for p in data.get("proposals", []):
        try:
            out.append(Mutation(
                mutation_type=p["mutation_type"],
                rationale=p.get("rationale", ""),
                details=p.get("details", {}),
            ))
        except (KeyError, TypeError):
            continue
    return out


# ── apply ─────────────────────────────────────────────────────────────────────

def apply(scaffold: ScaffoldVersion, mutation: Mutation) -> ScaffoldVersion:
    from tools import TOOL_LIBRARY

    new = ScaffoldVersion.model_validate(copy.deepcopy(scaffold.model_dump()))
    new.parent_id = scaffold.version_id
    new.mutation_rationale = mutation.rationale
    d = mutation.details
    existing_names = {t.name for t in new.tools}

    if mutation.mutation_type == "add_tool":
        name = d.get("name", "")
        if name in existing_names:
            return scaffold  # no-op

        source = d.get("source", "library")
        if source == "library":
            entry = TOOL_LIBRARY.get(name)
            if not entry:
                print(f"    [apply] library tool '{name}' not found — skipping")
                return scaffold
            new.tools.append(ToolSpec(
                name=name,
                source="library",
                description=entry["description"],
                parameters=entry["parameters"],
            ))
        else:
            # dynamic
            new.tools.append(ToolSpec(
                name=name,
                source="dynamic",
                description=d.get("description", ""),
                parameters=d.get("parameters", {"type": "object", "properties": {}, "required": []}),
                implementation=d.get("implementation", ""),
            ))

    elif mutation.mutation_type == "modify_tool":
        for t in new.tools:
            if t.name == d.get("name"):
                if "new_description" in d:
                    t.description = d["new_description"]
                break

    elif mutation.mutation_type == "rewrite_planning_policy":
        prompt = d.get("new_system_prompt", "")
        if prompt:
            new.planning_policy = PlanningPolicy(
                system_prompt=prompt,
                max_steps=scaffold.planning_policy.max_steps,
                max_tokens_per_step=scaffold.planning_policy.max_tokens_per_step,
            )

    elif mutation.mutation_type == "remove_tool":
        protected = {"list_files", "read_sample"}
        name = d.get("name", "")
        if name and name not in protected:
            new.tools = [t for t in new.tools if t.name != name]

    new.version_id = _vid(new)
    return new
