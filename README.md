# self-improving-agent

A minimal proof-of-life recursive self-improvement (RSI) scaffold: an agent that evolves its own tools, planning policy, and routing rules across generations by reflecting on its failures. Model weights are never modified — only the scaffold mutates.

## Concept

The agent has three mutable components:

| Component | What it is | How it mutates |
|---|---|---|
| **Tool registry** | Named Python functions the agent can call | Add, modify, or remove tools |
| **Planning policy** | System prompt governing task decomposition | Full rewrite |
| **Router rules** | Conditions that bias tool selection | Append new rules |

After each generation the meta-agent reads failure trajectories, proposes structured mutations, and each candidate is A/B-tested against a validation set. A mutation is accepted only if it improves val score by `> ε`.

## Architecture

```
agent_scaffold/
  scaffold.py       # Pydantic models (ToolSpec, PlanningPolicy, RouterRule, ScaffoldVersion)
                    # SHA-12 content-addressed versioning with parent pointers
  tools.py          # Built-in tools + subprocess runner for dynamically-generated tools
  agent.py          # Anthropic tool-use loop → Trajectory (steps, answer, tokens, latency)
  tasks.py          # 30-task suite (18 train / 6 val / 6 test) + evaluator
  meta_agent.py     # Reads failures → proposes JSON-structured mutations via claude-opus-4-7
  evolve.py         # Evolution loop: run → reflect → mutate → A/B gate → snapshot
  logs/             # Per-generation trajectories, scaffold snapshots, evolution_report.json
```

## Evolution loop

```python
for generation in range(N):
    trajectories = run_agent_on_train_set(current_scaffold)
    proposals    = meta_agent.propose_mutations(current_scaffold, failures)
    for proposal in proposals:
        candidate = apply(current_scaffold, proposal)
        if eval_on_val(candidate) > eval_on_val(current_scaffold) + epsilon:
            current_scaffold = candidate          # accepted
        # else: rejected, logged
    snapshot(current_scaffold, generation)

final_score = eval_on_test(current_scaffold)
```

## Task suite

30 multi-step tasks across four categories, designed for a ~45–55% baseline so there is room to improve:

| Category | Count | Requires |
|---|---|---|
| Math | 10 | `calculator`, multi-step decomposition |
| Code | 8 | `python_exec`, iteration |
| QA | 7 | `web_search`, fact retrieval |
| Hard / multi-step | 5 | New tools or chained reasoning |

Split: 60 / 20 / 20 train / val / test. Test set is held out until the final generation.

## Quickstart

```bash
git clone https://github.com/shreydesai/self-improving-agent
cd self-improving-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

export ANTHROPIC_API_KEY=sk-...

# Verify setup (no API key needed)
python3 test_scaffold.py

# Gen-0 baseline only
python3 evolve.py baseline

# Full evolution (4 generations)
python3 evolve.py evolve
```

## Mutation types

| Type | What changes | Guard |
|---|---|---|
| `add_tool` | New Python function added to registry | Val gate + no duplicate names |
| `modify_tool` | Existing tool description or implementation | Val gate |
| `rewrite_planning_policy` | System prompt replaced entirely | Val gate + 800-char soft cap |
| `add_routing_rule` | New condition → preferred tools mapping | Val gate |
| `remove_tool` | Tool removed from registry | Val gate + core tools protected |

## Safeguards

- **Sandboxing** — all tool code (including meta-agent-generated) runs in a `subprocess` with a 10 s timeout. Never `exec()`'d in the main process.
- **A/B gate** — `ε = 0.05`. No mutation ships without a measurable validation win.
- **Protected tools** — `calculator`, `web_search`, `python_exec`, `memory_read`, `memory_write` cannot be removed.
- **Budget caps** — `max_steps = 8` per episode, `max_mutations = 2` per generation, `N_GENERATIONS = 4` (all configurable at the top of `evolve.py`).
- **Determinism** — task ordering is fixed; meta-agent runs without temperature.

## Models

| Role | Model |
|---|---|
| Agent | `claude-sonnet-4-6` |
| Meta-agent | `claude-opus-4-7` |

## Output

Every run produces `logs/evolution_report.json`:

```json
{
  "generations": [
    {
      "generation": 0,
      "train_score": 0.94,
      "val_score": 1.0,
      "n_tools": 5,
      "accepted_mutations": [],
      ...
    }
  ],
  "final_test_score": 0.83
}
```

Each scaffold version is saved to `logs/scaffolds/<sha>.json` with a `parent_id` pointer — the full mutation genealogy is replayable.

## Co-authored with

Claude Sonnet 4.6 — [Claude Code](https://claude.ai/code)
