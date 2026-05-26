"""
Meta-agent: reads failure trajectories, proposes structured scaffold mutations.
Uses temperature=0 for deterministic proposals.
Proposals are structured diffs, not free-form text.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

import anthropic

from agent import Trajectory
from scaffold import ScaffoldVersion, ToolSpec, RouterRule, PlanningPolicy

META_MODEL = "claude-opus-4-7"
MAX_MUTATIONS_PER_ROUND = 3

MutationType = Literal[
    "add_tool",
    "modify_tool",
    "rewrite_planning_policy",
    "add_routing_rule",
    "remove_tool",
]


@dataclass
class MutationProposal:
    mutation_type: MutationType
    rationale: str
    details: Dict[str, Any]


# ── system prompt for meta-agent ──────────────────────────────────────────────

_META_SYSTEM = """\
You are a meta-agent that improves an AI agent's scaffolding (tools, planning policy, \
routing rules) by analyzing its failure trajectories.

You will receive:
1. The current scaffold (tools, planning policy, routing rules)
2. A summary of recent task failures with trajectories

Your job: propose at most {max_mutations} concrete scaffold mutations that would \
fix the observed failure patterns.

MUTATION TYPES you can propose:
- add_tool: Add a new Python tool the agent can call
- modify_tool: Edit an existing tool's description or implementation
- rewrite_planning_policy: Rewrite the system prompt governing task decomposition
- add_routing_rule: Add a rule like "if task mentions X, prefer tool Y"
- remove_tool: Remove a harmful or redundant tool

OUTPUT FORMAT (respond ONLY with valid JSON, no prose):
{{
  "analysis": "1-2 sentences describing the dominant failure pattern",
  "proposals": [
    {{
      "mutation_type": "add_tool",
      "rationale": "why this mutation addresses the failures",
      "details": {{
        "name": "tool_name",
        "description": "what the tool does",
        "parameters": {{
          "type": "object",
          "properties": {{
            "param1": {{"type": "string", "description": "..."}}
          }},
          "required": ["param1"]
        }},
        "implementation": "def tool_name(param1):\\n    # Python code\\n    return result"
      }}
    }},
    {{
      "mutation_type": "rewrite_planning_policy",
      "rationale": "...",
      "details": {{
        "new_system_prompt": "full replacement system prompt"
      }}
    }},
    {{
      "mutation_type": "add_routing_rule",
      "rationale": "...",
      "details": {{
        "condition": "if task involves computing edit distance or string similarity",
        "preferred_tools": ["levenshtein_distance"],
        "priority": 10
      }}
    }},
    {{
      "mutation_type": "modify_tool",
      "rationale": "...",
      "details": {{
        "name": "existing_tool_name",
        "new_description": "updated description",
        "new_implementation": "def existing_tool_name(...):\\n    ..."
      }}
    }},
    {{
      "mutation_type": "remove_tool",
      "rationale": "...",
      "details": {{
        "name": "tool_to_remove"
      }}
    }}
  ]
}}

RULES:
- Only propose mutations that directly address observed failures
- For add_tool: the implementation must be a complete, valid Python function
  that returns a string. Allowed imports: math, statistics, re, json, base64,
  hashlib, datetime, calendar, itertools, functools, collections, decimal.
- For rewrite_planning_policy: keep the prompt focused on tool selection strategy
  and task decomposition. Do not make it verbose (max 800 chars).
- Propose at most {max_mutations} mutations. Quality over quantity.
- Do not propose mutations for tasks the agent already solves correctly.
"""


def _summarize_failures(trajectories: List[Trajectory]) -> str:
    failures = [t for t in trajectories if not t.success]
    successes = [t for t in trajectories if t.success]
    lines = [
        f"Total tasks: {len(trajectories)}",
        f"Successes: {len(successes)} ({100*len(successes)//max(1,len(trajectories))}%)",
        f"Failures: {len(failures)}",
        "",
        "=== FAILURE DETAILS ===",
    ]
    for t in failures[:15]:  # cap to avoid token overflow
        lines.append(f"\nTask [{t.task_id}]: {t.task_text[:120]}")
        lines.append(f"  Correct answer: {t.correct_answer}")
        lines.append(f"  Agent answer: {t.final_answer[:150]}")
        tools_used = [s.tool_name for s in t.steps if s.tool_name]
        lines.append(f"  Tools used: {tools_used}")
        if t.steps:
            last = t.steps[-1]
            if last.tool_output:
                lines.append(f"  Last tool output: {last.tool_output[:200]}")
        if t.error:
            lines.append(f"  Error: {t.error}")
    return "\n".join(lines)


def _scaffold_summary(scaffold: ScaffoldVersion) -> str:
    lines = ["=== CURRENT SCAFFOLD ==="]
    lines.append(f"\nTools ({len(scaffold.tools)}):")
    for t in scaffold.tools:
        status = "enabled" if t.enabled else "disabled"
        is_builtin = t.implementation == "builtin"
        lines.append(f"  - {t.name} [{status}] {'(builtin)' if is_builtin else '(dynamic)'}")
        lines.append(f"    {t.description[:150]}")
    lines.append(f"\nPlanning policy ({len(scaffold.planning_policy.system_prompt)} chars):")
    lines.append(scaffold.planning_policy.system_prompt[:600])
    if scaffold.router_rules:
        lines.append(f"\nRouting rules ({len(scaffold.router_rules)}):")
        for r in scaffold.router_rules:
            lines.append(f"  - [{r.priority}] {r.condition} → {r.preferred_tools}")
    return "\n".join(lines)


def _parse_proposals(raw: str) -> List[MutationProposal]:
    # Extract JSON from response (strip any markdown fences)
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return []
    try:
        data = json.loads(match.group())
    except json.JSONDecodeError:
        return []

    proposals = []
    for p in data.get("proposals", []):
        try:
            proposals.append(MutationProposal(
                mutation_type=p["mutation_type"],
                rationale=p.get("rationale", ""),
                details=p.get("details", {}),
            ))
        except (KeyError, TypeError):
            continue
    return proposals


def propose_mutations(
    scaffold: ScaffoldVersion,
    trajectories: List[Trajectory],
    client: anthropic.Anthropic,
    max_mutations: int = MAX_MUTATIONS_PER_ROUND,
) -> List[MutationProposal]:
    """Ask the meta-agent to propose scaffold mutations based on failures."""
    failures = [t for t in trajectories if not t.success]
    if not failures:
        print("  [meta] No failures — skipping mutation round")
        return []

    system = _META_SYSTEM.format(max_mutations=max_mutations)
    user_content = "\n\n".join([
        _scaffold_summary(scaffold),
        _summarize_failures(trajectories),
    ])

    resp = client.messages.create(
        model=META_MODEL,
        max_tokens=4096,
        system=system,
        messages=[{"role": "user", "content": user_content}],
        temperature=0,
    )

    raw = resp.content[0].text if resp.content else ""
    proposals = _parse_proposals(raw)
    print(f"  [meta] proposed {len(proposals)} mutations")
    return proposals


# ── mutation applicator ───────────────────────────────────────────────────────

def apply_mutation(scaffold: ScaffoldVersion, proposal: MutationProposal) -> ScaffoldVersion:
    """Return a new scaffold with the mutation applied (does not mutate in place)."""
    import copy
    new = ScaffoldVersion.model_validate(copy.deepcopy(scaffold.model_dump()))
    new.parent_id = scaffold.version_id
    new.mutation_rationale = proposal.rationale
    d = proposal.details

    if proposal.mutation_type == "add_tool":
        # Reject if tool already exists
        existing = {t.name for t in new.tools}
        if d.get("name") in existing:
            return scaffold  # no-op
        new.tools.append(ToolSpec(
            name=d["name"],
            description=d.get("description", ""),
            parameters=d.get("parameters", {"type": "object", "properties": {}, "required": []}),
            implementation=d.get("implementation", ""),
        ))

    elif proposal.mutation_type == "modify_tool":
        for t in new.tools:
            if t.name == d.get("name"):
                if "new_description" in d:
                    t.description = d["new_description"]
                if "new_implementation" in d and d["new_implementation"]:
                    t.implementation = d["new_implementation"]
                break

    elif proposal.mutation_type == "rewrite_planning_policy":
        prompt = d.get("new_system_prompt", "")
        if prompt:
            new.planning_policy = PlanningPolicy(
                system_prompt=prompt,
                max_steps=scaffold.planning_policy.max_steps,
                temperature=scaffold.planning_policy.temperature,
            )

    elif proposal.mutation_type == "add_routing_rule":
        condition = d.get("condition", "")
        tools = d.get("preferred_tools", [])
        priority = d.get("priority", 0)
        if condition and tools:
            new.router_rules.append(RouterRule(
                condition=condition,
                preferred_tools=tools,
                priority=priority,
            ))

    elif proposal.mutation_type == "remove_tool":
        name = d.get("name")
        # Never remove built-in core tools
        protected = {"calculator", "web_search", "python_exec", "memory_read", "memory_write"}
        if name and name not in protected:
            new.tools = [t for t in new.tools if t.name != name]

    return new
