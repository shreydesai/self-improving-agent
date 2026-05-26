"""
Main evolution loop:
  for each generation:
    1. Run agent on train set → trajectories
    2. Meta-agent proposes mutations
    3. A/B gate: accept mutation only if val score improves by > epsilon
    4. Snapshot the scaffold
  Final: evaluate on held-out test set
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import List

import anthropic

from agent import Trajectory, run_episode
from meta_agent import apply_mutation, propose_mutations
from scaffold import ScaffoldVersion, make_initial_scaffold, save_scaffold, snapshot
from tasks import (
    TRAIN_TASKS, VAL_TASKS, TEST_TASKS,
    evaluate, score, Task
)

# ── configuration ─────────────────────────────────────────────────────────────

N_GENERATIONS    = 4       # total evolution generations (set low for proof-of-life speed)
EPSILON          = 0.05    # minimum val-score improvement to accept a mutation
MAX_MUTATIONS    = 2       # mutations proposed per generation
MAX_TRAIN_TASKS  = 18      # cap; use all train tasks
LOGS_DIR         = Path("logs")

# ── helpers ───────────────────────────────────────────────────────────────────

def run_suite(tasks: List[Task], scaffold: ScaffoldVersion, client: anthropic.Anthropic,
              label: str) -> List[Trajectory]:
    trajs = []
    for i, task in enumerate(tasks):
        print(f"    [{label}] {i+1}/{len(tasks)} {task.id} ...", end=" ", flush=True)
        t = run_episode(
            task_id=task.id,
            task_text=task.text,
            correct_answer=task.correct_answer,
            scaffold=scaffold,
            client=client,
            evaluator=evaluate,
        )
        mark = "✓" if t.success else "✗"
        print(f"{mark}  ({t.final_answer[:50].strip()!r})")
        trajs.append(t)
    return trajs


def save_trajectories(trajs: List[Trajectory], tag: str) -> None:
    LOGS_DIR.mkdir(exist_ok=True)
    path = LOGS_DIR / f"{tag}.json"
    data = []
    for t in trajs:
        data.append({
            "task_id": t.task_id,
            "task_text": t.task_text,
            "correct_answer": t.correct_answer,
            "final_answer": t.final_answer,
            "success": t.success,
            "tokens_used": t.tokens_used,
            "elapsed_s": t.elapsed_s,
            "error": t.error,
            "steps": [
                {
                    "step_num": s.step_num,
                    "tool_name": s.tool_name,
                    "tool_input": s.tool_input,
                    "tool_output": s.tool_output[:300] if s.tool_output else None,
                    "assistant_text": s.assistant_text[:300],
                }
                for s in t.steps
            ],
        })
    path.write_text(json.dumps(data, indent=2))


def eval_val(scaffold: ScaffoldVersion, client: anthropic.Anthropic) -> float:
    trajs = run_suite(VAL_TASKS, scaffold, client, label="val")
    return score(trajs)


def generation_report(gen: int, train_score: float, val_score: float,
                      scaffold: ScaffoldVersion, accepted: list, rejected: list) -> dict:
    return {
        "generation": gen,
        "train_score": round(train_score, 4),
        "val_score": round(val_score, 4),
        "n_tools": len(scaffold.tools),
        "tool_names": [t.name for t in scaffold.tools],
        "planning_policy_chars": len(scaffold.planning_policy.system_prompt),
        "n_routing_rules": len(scaffold.router_rules),
        "accepted_mutations": accepted,
        "rejected_mutations": rejected,
        "scaffold_id": scaffold.version_id,
    }


# ── main evolution loop ───────────────────────────────────────────────────────

def evolve(api_key: str | None = None) -> None:
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])

    scaffold = make_initial_scaffold()
    scaffold = snapshot(scaffold, generation=0, rationale="initial scaffold")

    history = []
    best_test_score = None

    # Gen-0 baseline
    print("\n========== BASELINE (generation 0) ==========")
    train_trajs = run_suite(TRAIN_TASKS[:MAX_TRAIN_TASKS], scaffold, client, label="train")
    save_trajectories(train_trajs, "gen0_train")
    train_score = score(train_trajs)

    val_trajs = run_suite(VAL_TASKS, scaffold, client, label="val")
    save_trajectories(val_trajs, "gen0_val")
    val_score = score(val_trajs)

    print(f"\n  Gen-0  train={train_score:.0%}  val={val_score:.0%}")
    history.append(generation_report(0, train_score, val_score, scaffold, [], []))

    # Evolution
    for gen in range(1, N_GENERATIONS + 1):
        print(f"\n========== GENERATION {gen} ==========")

        proposals = propose_mutations(scaffold, train_trajs, client, MAX_MUTATIONS)
        accepted_names = []
        rejected_names = []

        baseline_val = val_score

        for proposal in proposals:
            print(f"  [proposal] {proposal.mutation_type}: {proposal.rationale[:80]}")
            candidate = apply_mutation(scaffold, proposal)
            if candidate.version_id == scaffold.version_id:
                print("    → no-op (skipped)")
                rejected_names.append(f"{proposal.mutation_type}(no-op)")
                continue

            print(f"    evaluating candidate on val set ...")
            candidate_val = eval_val(candidate, client)
            delta = candidate_val - baseline_val
            print(f"    val: {baseline_val:.0%} → {candidate_val:.0%}  (Δ={delta:+.0%})")

            if delta > EPSILON:
                print(f"    ✓ ACCEPTED")
                scaffold = candidate
                scaffold = snapshot(scaffold, generation=gen, rationale=proposal.rationale)
                baseline_val = candidate_val
                accepted_names.append(f"{proposal.mutation_type}: {proposal.rationale[:60]}")
            else:
                print(f"    ✗ REJECTED (Δ={delta:+.0%} ≤ ε={EPSILON:.0%})")
                rejected_names.append(f"{proposal.mutation_type}: {proposal.rationale[:60]}")

        # Re-run train on accepted scaffold for next generation's failure analysis
        print(f"\n  Running train set on accepted scaffold ...")
        train_trajs = run_suite(TRAIN_TASKS[:MAX_TRAIN_TASKS], scaffold, client, label="train")
        save_trajectories(train_trajs, f"gen{gen}_train")
        train_score = score(train_trajs)
        val_score = baseline_val

        print(f"\n  Gen-{gen}  train={train_score:.0%}  val={val_score:.0%}  "
              f"tools={len(scaffold.tools)}  accepted={len(accepted_names)}")

        history.append(generation_report(
            gen, train_score, val_score, scaffold, accepted_names, rejected_names
        ))

    # Final test evaluation (held-out)
    print("\n========== FINAL TEST EVALUATION ==========")
    test_trajs = run_suite(TEST_TASKS, scaffold, client, label="test")
    save_trajectories(test_trajs, "final_test")
    test_score = score(test_trajs)
    print(f"\n  FINAL TEST SCORE: {test_score:.0%}  ({sum(t.success for t in test_trajs)}/{len(test_trajs)})")

    # Save history
    report = {
        "generations": history,
        "final_test_score": round(test_score, 4),
        "baseline_test_not_run": True,  # test held out until the end
        "final_scaffold_id": scaffold.version_id,
        "final_tool_names": [t.name for t in scaffold.tools],
    }
    (LOGS_DIR / "evolution_report.json").write_text(json.dumps(report, indent=2))
    print("\n  Report saved to logs/evolution_report.json")
    _print_summary(history, test_score)


def _print_summary(history: list, test_score: float) -> None:
    print("\n" + "=" * 60)
    print("EVOLUTION SUMMARY")
    print("=" * 60)
    print(f"{'Gen':>4}  {'Train':>7}  {'Val':>7}  {'Tools':>6}  Accepted mutations")
    print("-" * 60)
    for h in history:
        acc = ", ".join(h["accepted_mutations"]) or "—"
        print(f"{h['generation']:>4}  {h['train_score']:>7.0%}  {h['val_score']:>7.0%}  "
              f"{h['n_tools']:>6}  {acc[:40]}")
    print("-" * 60)
    print(f"Final held-out test score: {test_score:.0%}")
    print("=" * 60)


# ── quick baseline-only mode ──────────────────────────────────────────────────

def run_baseline(api_key: str | None = None) -> None:
    """Run only generation-0 baseline. Use this to verify the setup before full evolve()."""
    client = anthropic.Anthropic(api_key=api_key or os.environ["ANTHROPIC_API_KEY"])
    scaffold = make_initial_scaffold()
    scaffold = snapshot(scaffold, generation=0)

    print("\n========== BASELINE RUN ==========")
    print("--- TRAIN ---")
    train_trajs = run_suite(TRAIN_TASKS, scaffold, client, label="train")
    save_trajectories(train_trajs, "baseline_train")
    print(f"Train score: {score(train_trajs):.0%}")

    print("\n--- VAL ---")
    val_trajs = run_suite(VAL_TASKS, scaffold, client, label="val")
    save_trajectories(val_trajs, "baseline_val")
    print(f"Val score:   {score(val_trajs):.0%}")

    print("\n--- TEST (held out — for baseline measurement only) ---")
    test_trajs = run_suite(TEST_TASKS, scaffold, client, label="test")
    save_trajectories(test_trajs, "baseline_test")
    print(f"Test score:  {score(test_trajs):.0%}")


if __name__ == "__main__":
    import sys
    mode = sys.argv[1] if len(sys.argv) > 1 else "baseline"
    if mode == "baseline":
        run_baseline()
    elif mode == "evolve":
        evolve()
    else:
        print("Usage: python evolve.py [baseline|evolve]")
