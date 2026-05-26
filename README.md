# self-improving-agent

A self-improving data analysis agent that discovers its own tools through failure.
The agent starts with only two tools (`list_files`, `read_sample`) and evolves a
richer toolkit across generations by reflecting on what it couldn't compute.

## Concept

The agent answers analytical questions about CSV files too large to fit in context.
It can only access data through tool calls. When it lacks the right tool, it fails.
A meta-agent reads those failures, proposes new tools from a pre-built library
(or writes custom ones), and gates each mutation behind a validation-set improvement.

**The discovery story:**

```
Gen-0  tools=[list_files, read_sample]         train=0%   val=0%
       → agent samples data but can't count or sum
       meta-agent: "add filter_count, column_sum"
Gen-1  tools=[..., column_sum]                 train=33%  val=100%
       → can now answer sum questions; count questions still fail
       meta-agent: "add filter_count"
Gen-2  tools=[..., filter_count]               train=67%  val=100%
       → still can't rank by group (needs group_aggregate)
Gen-3  tools=[..., group_aggregate]            train=100% val=100%
```

## Architecture

```
scenarios/
  base.py             # Scenario + Task dataclasses, auto-discovery registry
  orders_basic.py     # first scenario (single table, 5 tasks)
scaffold.py           # versioned tool registry (builtin | library | dynamic)
tools.py              # pandas-backed implementations + TOOL_LIBRARY manifest
agent.py              # Haiku agent (512 tok/step, 5 steps max)
meta_agent.py         # reads failures → proposes typed mutations
evolve.py             # A/B-gated evolution loop, scenario-agnostic
cli.py                # scenario CRUD + run commands
```

## Quickstart

```bash
git clone https://github.com/shreydesai/self-improving-agent
cd self-improving-agent
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
export ANTHROPIC_API_KEY=sk-...

# explore scenarios
python cli.py list
python cli.py show orders_basic

# run evolution
python cli.py run orders_basic
```

## Adding a scenario

Create `scenarios/my_scenario.py`:

```python
from pathlib import Path
from scenarios.base import Scenario, Task, register

def generate_data(data_dir: Path) -> None:
    # write CSV files to data_dir
    ...

scenario = Scenario(
    id="my_scenario",
    name="My scenario",
    description="...",
    data_dir=Path(__file__).parent / "data" / "my_scenario",
    generate_data=generate_data,
    tasks=[
        Task("t1", "question?", "answer", "train"),
        Task("t2", "question?", "answer", "val"),
        Task("t3", "question?", "answer", "test"),
    ],
)
register(scenario)
```

That's it — `python cli.py list` picks it up automatically.

## Tool library

The meta-agent unlocks tools by name from a pre-built library:

| Tool | What it does |
|---|---|
| `filter_count` | Count rows matching conditions |
| `column_sum` | Sum a column with optional filters |
| `column_stats` | min / max / mean / count for a column |
| `get_distinct` | Unique values of a column |
| `group_aggregate` | Group-by + aggregate (sum/count/mean/min/max) |
| `sort_top_k` | Top-k rows sorted by a column |
| `join_tables` | Join two CSVs, optionally cache result |

Custom tools (dynamic Python) are also supported when no library tool fits.

## Design constraints

- **No `python_exec`** — forces real tool discovery instead of ad-hoc scripting
- **512 tokens/step, 5 steps max** — agent must call tools, not reason from context
- **ε = 0.15 val gate** — mutations accepted only on measurable improvement
- **Content-addressed versioning** — every scaffold saved with parent pointer

## Models

| Role | Model |
|---|---|
| Agent | `claude-haiku-4-5-20251001` |
| Meta-agent | `claude-opus-4-7` |

Co-authored with Claude Sonnet 4.6 — [Claude Code](https://claude.ai/code)
