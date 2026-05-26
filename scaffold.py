"""
Scaffold: versioned set of tools + planning policy.
Tools are either 'library' (name maps to tools.TOOL_LIBRARY) or
'dynamic' (implementation stored as Python source).
"""
from __future__ import annotations

import hashlib
import time
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

ToolSource = Literal["builtin", "library", "dynamic"]


class ToolSpec(BaseModel):
    name: str
    description: str
    parameters: Dict[str, Any]
    source: ToolSource = "library"
    implementation: str = ""          # populated for dynamic tools
    added_generation: int = 0


class PlanningPolicy(BaseModel):
    system_prompt: str
    max_steps: int = 5
    max_tokens_per_step: int = 512    # kept low to force tool use over reasoning


class ScaffoldVersion(BaseModel):
    version_id: str = "init"
    parent_id: Optional[str] = None
    generation: int = 0
    created_at: float = Field(default_factory=time.time)
    tools: List[ToolSpec]
    planning_policy: PlanningPolicy
    mutation_rationale: Optional[str] = None


# ── persistence ───────────────────────────────────────────────────────────────

LOGS_DIR    = Path("logs")
SCAFFOLDS_DIR = LOGS_DIR / "scaffolds"


def _vid(s: ScaffoldVersion) -> str:
    data = s.model_dump_json(exclude={"version_id", "created_at", "parent_id"})
    return hashlib.sha256(data.encode()).hexdigest()[:12]


def save(s: ScaffoldVersion) -> None:
    SCAFFOLDS_DIR.mkdir(parents=True, exist_ok=True)
    (SCAFFOLDS_DIR / f"{s.version_id}.json").write_text(s.model_dump_json(indent=2))


def load(version_id: str) -> ScaffoldVersion:
    return ScaffoldVersion.model_validate_json(
        (SCAFFOLDS_DIR / f"{version_id}.json").read_text()
    )


def snapshot(s: ScaffoldVersion, generation: int, rationale: str = "") -> ScaffoldVersion:
    s.generation = generation
    s.version_id = _vid(s)
    if rationale:
        s.mutation_rationale = rationale
    save(s)
    tool_names = [t.name for t in s.tools]
    print(f"  [scaffold] gen={generation} id={s.version_id} tools={tool_names}")
    return s


# ── initial scaffold factory ──────────────────────────────────────────────────

INITIAL_PROMPT = """\
You are a precise data analyst. You answer questions about CSV data files.

Rules:
- You CANNOT read entire files — they are too large. Use tools to query.
- Always call list_files first if you don't know what data is available.
- Call read_sample to understand a file's columns and value formats.
- Use the most specific tool available for each operation.
- Give a short, direct final answer with just the requested value.
- Do not guess or estimate — compute exact answers using tools.
"""


def make_initial_scaffold() -> ScaffoldVersion:
    from tools import TOOL_LIBRARY

    tools = [
        ToolSpec(
            name="list_files",
            source="builtin",
            description="List all available CSV data files with their row counts and columns.",
            parameters={"type": "object", "properties": {}, "required": []},
        ),
        ToolSpec(
            name="read_sample",
            source="builtin",
            description=(
                "Read a small sample of rows from a CSV file to understand its "
                "structure and value formats. Default 10 rows."
            ),
            parameters={
                "type": "object",
                "properties": {
                    "file":   {"type": "string", "description": "CSV filename"},
                    "n_rows": {"type": "integer", "default": 10,
                               "description": "Number of rows to sample"},
                },
                "required": ["file"],
            },
        ),
    ]

    return ScaffoldVersion(
        version_id="initial",
        tools=tools,
        planning_policy=PlanningPolicy(system_prompt=INITIAL_PROMPT),
    )
