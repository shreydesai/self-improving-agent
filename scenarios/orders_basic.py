"""
Scenario: orders_basic
Single table (orders.csv, 5 000 rows). Tasks require exact filter/aggregate
answers — impossible to estimate from a small sample, which forces the
meta-agent to discover filter_count and column_sum.

Train (2): one count task, one sum task
Val   (2): one count task + one filtered-sum task (covers both tool types
           so val ceiling is 50% until filter_count is added, then 100%)
Test  (1): group rank (needs group_aggregate)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.base import Scenario, Task, register

# ── data generator ────────────────────────────────────────────────────────────

SEED = 42
N_ROWS = 5_000
STATUSES = ["completed", "pending", "cancelled"]
REGIONS   = ["North", "South", "East", "West"]


def generate_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    order_id    = np.arange(1, N_ROWS + 1)
    customer_id = rng.integers(1, 1001, N_ROWS)
    product_id  = rng.integers(1, 201, N_ROWS)
    quantity    = rng.integers(1, 11, N_ROWS)
    unit_price  = np.round(rng.uniform(5.0, 200.0, N_ROWS), 2)
    revenue     = np.round(quantity * unit_price, 2)
    status      = rng.choice(STATUSES, N_ROWS, p=[0.55, 0.30, 0.15])
    region      = rng.choice(REGIONS,  N_ROWS)

    # deterministic dates: 2024-01-01 … 2024-12-31
    base = np.datetime64("2024-01-01")
    offsets = rng.integers(0, 366, N_ROWS)
    created_at = [str((base + np.timedelta64(int(d), "D")))[:10] for d in offsets]

    df = pd.DataFrame({
        "order_id":    order_id,
        "customer_id": customer_id,
        "product_id":  product_id,
        "quantity":    quantity,
        "unit_price":  unit_price,
        "revenue":     revenue,
        "status":      status,
        "region":      region,
        "created_at":  created_at,
    })
    df.to_csv(data_dir / "orders.csv", index=False)
    print(f"  [generate] orders.csv  {len(df):,} rows → {data_dir}")


# ── ground truth (computed offline with pandas) ───────────────────────────────

def _compute_answers(data_dir: Path) -> dict:
    df = pd.read_csv(data_dir / "orders.csv")
    return {
        "t1": int((df["status"] == "completed").sum()),
        "t2": round(float(df["revenue"].sum()), 2),
        "t3": int((df["quantity"] > 3).sum()),
        "t4": round(float(df.loc[df["region"] == "West", "revenue"].sum()), 2),
        "t5": str(df.groupby("region")["revenue"].sum().idxmax()),
    }


# ── scenario definition ───────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data" / "orders_basic"

# Generate data now if not present so answers can be computed at import time.
# Re-running generate_data is idempotent (fixed seed).
if not (_DATA_DIR / "orders.csv").exists():
    generate_data(_DATA_DIR)

_GT = _compute_answers(_DATA_DIR)

scenario = Scenario(
    id="orders_basic",
    name="Orders — basic aggregations",
    description="Single orders table. Tasks need exact filter/aggregate answers.",
    data_dir=_DATA_DIR,
    generate_data=generate_data,
    tasks=[
        # train: one count, one sum
        Task("t1", "How many orders have status 'completed'? Give the exact count.",
             str(_GT["t1"]), "train",
             "filter_count(orders.csv, status == completed)"),
        Task("t2", "What is the total revenue across ALL orders? Round to 2 decimal places.",
             str(_GT["t2"]), "train",
             "column_sum(orders.csv, revenue)"),
        # val: one count + one filtered sum — covers both tool types
        # val ceiling stays at 50% until filter_count is unlocked, giving the gate real signal
        Task("t3", "How many orders have a quantity greater than 3? Give the exact count.",
             str(_GT["t3"]), "val",
             "filter_count(orders.csv, quantity > 3)"),
        Task("t4",
             "What is the total revenue for orders placed in the 'West' region? "
             "Round to 2 decimal places.",
             str(_GT["t4"]), "val",
             "column_sum(orders.csv, revenue, region==West)"),
        # test: group + rank — needs group_aggregate, held out until final eval
        Task("t5",
             "Which region has the highest total revenue? Return only the region name.",
             _GT["t5"], "test",
             "group_aggregate + sort_top_k"),
    ],
)

register(scenario)
