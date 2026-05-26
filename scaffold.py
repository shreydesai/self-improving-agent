"""
Mutable scaffold state: tools, planner prompt, router config.
Every version is saved to disk with a parent pointer for full replay.
"""
from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]          # JSON schema
    implementation: str                  # Python source (built-ins use sentinel "builtin")
    enabled: bool = True
    created_at: float = Field(default_factory=time.time)
    source_generation: int = 0           # which generation added this tool


class RouterRule(BaseModel):
    condition: str                       # natural language condition
    preferred_tools: List[str]
    priority: int = 0


class PlanningPolicy(BaseModel):
    system_prompt: str
    max_steps: int = 8
    temperature: float = 0.7


class ScaffoldVersion(BaseModel):
    version_id: str = "init"
    parent_id: Optional[str] = None
    generation: int = 0
    created_at: float = Field(default_factory=time.time)
    tools: List[ToolSpec]
    planning_policy: PlanningPolicy
    router_rules: List[RouterRule] = Field(default_factory=list)
    mutation_rationale: Optional[str] = None


# ── persistence ──────────────────────────────────────────────────────────────

LOGS_DIR = Path("logs")
SCAFFOLDS_DIR = LOGS_DIR / "scaffolds"


def _compute_vid(scaffold: ScaffoldVersion) -> str:
    data = scaffold.model_dump_json(exclude={"version_id", "created_at", "parent_id"})
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def save_scaffold(scaffold: ScaffoldVersion) -> Path:
    SCAFFOLDS_DIR.mkdir(parents=True, exist_ok=True)
    path = SCAFFOLDS_DIR / f"{scaffold.version_id}.json"
    path.write_text(scaffold.model_dump_json(indent=2))
    return path


def load_scaffold(version_id: str) -> ScaffoldVersion:
    path = SCAFFOLDS_DIR / f"{version_id}.json"
    return ScaffoldVersion.model_validate_json(path.read_text())


def snapshot(scaffold: ScaffoldVersion, generation: int, rationale: str = "") -> ScaffoldVersion:
    scaffold.generation = generation
    scaffold.version_id = _compute_vid(scaffold)
    if rationale:
        scaffold.mutation_rationale = rationale
    save_scaffold(scaffold)
    print(f"  [scaffold] gen={generation} id={scaffold.version_id} tools={len(scaffold.tools)}")
    return scaffold


# ── initial scaffold factory ──────────────────────────────────────────────────

INITIAL_SYSTEM_PROMPT = """\
You are a problem-solving agent with access to a set of tools. Your goal is to \
answer the user's question as accurately as possible.

Guidelines:
- Think step by step before using any tool.
- Use the calculator tool for arithmetic.
- Use web_search to look up facts.
- Use python_exec to run code when needed.
- Use memory_write/memory_read to store intermediate results across steps.
- After gathering all necessary information, provide a clear, concise final answer.
- Your final answer should directly address what was asked.
"""


def make_initial_scaffold() -> ScaffoldVersion:
    tools = [
        ToolSpec(
            name="calculator",
            description="Evaluate a mathematical expression and return the numeric result. "
                        "Supports standard arithmetic, math functions (sqrt, log, sin, cos, etc.), "
                        "and constants (pi, e).",
            parameters={
                "type": "object",
                "properties": {
                    "expression": {
                        "type": "string",
                        "description": "A Python-evaluable math expression, e.g. '15 * 0.10 + 30' or 'sqrt(144)'",
                    }
                },
                "required": ["expression"],
            },
            implementation="builtin",
        ),
        ToolSpec(
            name="web_search",
            description="Search for factual information. Returns relevant facts from a knowledge base. "
                        "Use for geography, science, history, and other factual questions.",
            parameters={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    }
                },
                "required": ["query"],
            },
            implementation="builtin",
        ),
        ToolSpec(
            name="python_exec",
            description="Execute Python code and return the printed output. "
                        "Useful for algorithms, data processing, and complex computations. "
                        "Safe imports allowed: math, statistics, random, itertools, "
                        "functools, collections, string, json, re, base64, hashlib, datetime, decimal.",
            parameters={
                "type": "object",
                "properties": {
                    "code": {
                        "type": "string",
                        "description": "Python code to execute. Use print() to output results.",
                    }
                },
                "required": ["code"],
            },
            implementation="builtin",
        ),
        ToolSpec(
            name="memory_write",
            description="Store a value in memory for later retrieval.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Storage key"},
                    "value": {"type": "string", "description": "Value to store"},
                },
                "required": ["key", "value"],
            },
            implementation="builtin",
        ),
        ToolSpec(
            name="memory_read",
            description="Retrieve a previously stored value from memory.",
            parameters={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "description": "Storage key"},
                },
                "required": ["key"],
            },
            implementation="builtin",
        ),
    ]

    return ScaffoldVersion(
        version_id="initial",
        tools=tools,
        planning_policy=PlanningPolicy(system_prompt=INITIAL_SYSTEM_PROMPT),
    )
