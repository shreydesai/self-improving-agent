"""
Scenario + Task dataclasses and the global scenario registry.
Adding a new scenario = dropping a new .py file in scenarios/.
"""
from __future__ import annotations

import importlib
import pkgutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal

Split = Literal["train", "val", "test"]


@dataclass
class Task:
    id: str
    question: str
    answer: Any                   # pre-computed ground truth
    split: Split
    notes: str = ""               # optional hint (not shown to agent)


@dataclass
class Scenario:
    id: str
    name: str
    description: str
    data_dir: Path                # where CSV files live
    tasks: List[Task]
    generate_data: callable       # fn(data_dir) -> None

    @property
    def train_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.split == "train"]

    @property
    def val_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.split == "val"]

    @property
    def test_tasks(self) -> List[Task]:
        return [t for t in self.tasks if t.split == "test"]


# ── registry ──────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Scenario] = {}


def register(scenario: Scenario) -> None:
    _REGISTRY[scenario.id] = scenario


def get(scenario_id: str) -> Scenario:
    if scenario_id not in _REGISTRY:
        raise KeyError(f"Unknown scenario '{scenario_id}'. Available: {list(_REGISTRY)}")
    return _REGISTRY[scenario_id]


def list_all() -> List[Scenario]:
    return list(_REGISTRY.values())


def load_all() -> None:
    """Auto-import every module in the scenarios/ package (except base)."""
    pkg_dir = Path(__file__).parent
    for finder, name, _ in pkgutil.iter_modules([str(pkg_dir)]):
        if name not in ("base",):
            importlib.import_module(f"scenarios.{name}")
