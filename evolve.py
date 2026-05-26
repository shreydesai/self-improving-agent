"""
Evolution loop: run → reflect → mutate → A/B gate → snapshot.
Scenario-agnostic: pass any registered scenario id.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List, Optional

import anthropic

import scenarios as sc
from agent import Trajectory, run_episode, evaluate
from meta_agent import Mutation, apply, propose
from scaffold import ScaffoldVersion, make_initial_scaffold, snapshot
from scenarios.base import Task

EPSILON        = 0.15   # higher bar since val set is small (1 task = 100% swings)
N_GENERATIONS  = 4
MAX_MUTATIONS  = 2
LOGS_DIR       = Path("logs")


# ── helpers ───────────────────────────────────────────────────────────────────

def _run_tasks(tasks: List[Task], scaffold: ScaffoldVersion,
               data_dir: Path, client: anthropic.Anthropic,
               label: str) -> List[Trajectory]:
    trajs = []
    for i, task in enumerate(tasks):
        print(f"    [{label}] {i+1}/{len(tasks)} {task.id} ...", end=" ", flush=True)
        t = run_episode(task.id, task.question, task.answer,
                        scaffold, data_dir, client)
        mark = "✓" if t.success else "✗"
        print(f"{mark}  ({t.final_answer[:60].strip()!r})")
        trajs.append(t)
    return trajs


def _score(trajs: List[Trajectory]) -> float:
    return sum(t.success for t in trajs) / len(trajs) if trajs else 0.0


def _save(trajs: List[Trajectory], tag: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    (LOGS_DIR / f"{tag}.json").write_text(json.dumps([
        {"task_id": t.task_id, "question": t.question,
         "correct": t.correct_answer, "predicted": t.final_answer,
         "success": t.success, "tokens": t.tokens_used,
         "steps": [{"tool": s.tool_name, "input": s.tool_input,
                    "output": s.tool_output} for s in t.steps]}
        for t in trajs
    ], indent=2))


def _val_score(scaffold: ScaffoldVersion, scenario, client) -> float:
    trajs = _run_tasks(scenario.val_tasks, scaffold, scenario.data_dir, client, "val")
    return _score(trajs)


# ── main loop ─────────────────────────────────────────────────────────────────

def evolve(scenario_id: str, api_key: Optional[str] = None,
           n_generations: int = N_GENERATIONS) -> None:
    sc.load_all()
    scenario = sc.get(scenario_id)
    client   = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    # ensure data exists
    if not any(scenario.data_dir.glob("*.csv")):
        print("  [data] generating …")
        scenario.generate_data(scenario.data_dir)

    scaffold = make_initial_scaffold()
    scaffold = snapshot(scaffold, 0)

    history = []

    # ── gen 0 baseline ────────────────────────────────────────────────────────
    print(f"\n{'='*52}\nBASELINE — scenario: {scenario.name}\n{'='*52}")
    train_trajs = _run_tasks(scenario.train_tasks, scaffold, scenario.data_dir, client, "train")
    _save(train_trajs, "gen0_train")
    val_trajs = _run_tasks(scenario.val_tasks, scaffold, scenario.data_dir, client, "val")
    _save(val_trajs, "gen0_val")

    train_score = _score(train_trajs)
    val_score   = _score(val_trajs)
    print(f"\n  Gen-0  train={train_score:.0%}  val={val_score:.0%}  tools={len(scaffold.tools)}")
    history.append(_record(0, train_score, val_score, scaffold, [], []))

    # ── evolution ─────────────────────────────────────────────────────────────
    for gen in range(1, n_generations + 1):
        print(f"\n{'='*52}\nGENERATION {gen}\n{'='*52}")

        mutations = propose(scaffold, train_trajs, client, MAX_MUTATIONS)
        accepted, rejected = [], []

        for mut in mutations:
            print(f"  [proposal] {mut.mutation_type}: {mut.rationale[:80]}")
            candidate = apply(scaffold, mut)
            if candidate.version_id == scaffold.version_id:
                print("    → no-op")
                rejected.append(f"{mut.mutation_type}(no-op)")
                continue

            cand_val = _val_score(candidate, scenario, client)
            delta = cand_val - val_score
            print(f"    val: {val_score:.0%} → {cand_val:.0%}  (Δ={delta:+.0%})")

            if delta > EPSILON:
                print("    ✓ ACCEPTED")
                scaffold  = candidate
                scaffold  = snapshot(scaffold, gen, mut.rationale)
                val_score = cand_val
                accepted.append(f"{mut.mutation_type}: {mut.rationale[:60]}")
            else:
                print(f"    ✗ REJECTED")
                rejected.append(f"{mut.mutation_type}: {mut.rationale[:60]}")

        print(f"\n  Running train set on updated scaffold …")
        train_trajs = _run_tasks(scenario.train_tasks, scaffold,
                                  scenario.data_dir, client, "train")
        _save(train_trajs, f"gen{gen}_train")
        train_score = _score(train_trajs)
        print(f"\n  Gen-{gen}  train={train_score:.0%}  val={val_score:.0%}  "
              f"tools={len(scaffold.tools)}  accepted={len(accepted)}")
        history.append(_record(gen, train_score, val_score, scaffold, accepted, rejected))

    # ── test ──────────────────────────────────────────────────────────────────
    print(f"\n{'='*52}\nFINAL TEST\n{'='*52}")
    test_trajs = _run_tasks(scenario.test_tasks, scaffold, scenario.data_dir, client, "test")
    _save(test_trajs, "final_test")
    test_score = _score(test_trajs)
    print(f"\n  TEST SCORE: {test_score:.0%}  ({sum(t.success for t in test_trajs)}/{len(test_trajs)})")

    report = {"scenario": scenario_id, "generations": history,
              "final_test_score": round(test_score, 4),
              "final_tools": [t.name for t in scaffold.tools]}
    (LOGS_DIR / "report.json").write_text(json.dumps(report, indent=2))
    _print_summary(history, test_score)


def _record(gen, train, val, scaffold, accepted, rejected):
    return {"generation": gen, "train_score": round(train, 4), "val_score": round(val, 4),
            "n_tools": len(scaffold.tools),
            "tool_names": [t.name for t in scaffold.tools],
            "accepted": accepted, "rejected": rejected,
            "scaffold_id": scaffold.version_id}


def _print_summary(history, test_score):
    print(f"\n{'='*60}\nEVOLUTION SUMMARY\n{'='*60}")
    print(f"{'Gen':>4}  {'Train':>7}  {'Val':>7}  {'Tools':>6}  Accepted")
    print("-" * 60)
    for h in history:
        acc = "; ".join(h["accepted"]) or "—"
        print(f"{h['generation']:>4}  {h['train_score']:>7.0%}  "
              f"{h['val_score']:>7.0%}  {h['n_tools']:>6}  {acc[:45]}")
    print("-" * 60)
    print(f"Final test: {test_score:.0%}")
    print("=" * 60)


if __name__ == "__main__":
    import sys
    sid = sys.argv[1] if len(sys.argv) > 1 else "orders_basic"
    evolve(sid)
