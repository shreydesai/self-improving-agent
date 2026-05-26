"""
Scenario: orders_multi
Three tables (orders 50k, customers 10k, products 1k). Every task requires
join_tables before any filtering or aggregation — a sample of any single
table gives no signal on cross-table questions, forcing the meta-agent to
discover join_tables.

Discovery story (starting from orders_basic-trained scaffold):
  Gen-0  tools=[list_files, read_sample, filter_count, column_sum]
         val=0%   — can't answer anything without joining
  Gen-1  tools=[..., join_tables]
         val=100% — join + existing filter/sum tools cover both val tasks
  Gen-2  group_aggregate proposed for test task, but val already saturated

Train (2): gold-tier order count; Nike-brand revenue sum
Val   (2): US-completed order count; Electronics-category revenue sum
           val ceiling = 0% until join_tables unlocked (both tasks need it)
Test  (1): which country has the highest completed-order revenue
           (needs join + group_aggregate)
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from scenarios.base import Scenario, Task, register

# ── constants ─────────────────────────────────────────────────────────────────

SEED        = 7
N_ORDERS    = 50_000
N_CUSTOMERS = 10_000
N_PRODUCTS  =  1_000

STATUSES   = ["completed", "pending",  "cancelled"]
REGIONS    = ["North",     "South",    "East",   "West"]
COUNTRIES  = ["US",        "UK",       "CA",     "AU",  "DE"]
TIERS      = ["gold",      "silver",   "bronze"]
CATEGORIES = ["Electronics", "Clothing", "Books", "Sports", "Home"]

_BRANDS: dict = {
    "Electronics": ["Apple",   "Samsung",       "Sony",         "LG",          "Dell"],
    "Clothing":    ["Nike",    "Adidas",         "Zara",         "H&M",         "Uniqlo"],
    "Books":       ["Penguin", "HarperCollins",  "Random House", "Scholastic",  "Oxford"],
    "Sports":      ["Nike",    "Adidas",         "Under Armour", "Puma",        "Reebok"],
    "Home":        ["IKEA",    "Wayfair",        "Ashley",       "Pottery Barn","Crate&Barrel"],
}


# ── data generator ────────────────────────────────────────────────────────────

def generate_data(data_dir: Path) -> None:
    data_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)

    # ── products (1 000) ─────────────────────────────────────────────
    product_id = np.arange(1, N_PRODUCTS + 1)
    # cycle through categories so each gets exactly N_PRODUCTS/5 entries
    cat_idx  = (product_id - 1) % len(CATEGORIES)
    category = [CATEGORIES[i] for i in cat_idx]
    # brand rotates within each category group of 5
    brand    = [_BRANDS[c][(i // len(CATEGORIES)) % 5] for i, c in enumerate(category)]
    cost_price = np.round(rng.uniform(5.0, 150.0, N_PRODUCTS), 2)

    products_df = pd.DataFrame({
        "product_id": product_id,
        "category":   category,
        "brand":      brand,
        "cost_price": cost_price,
    })
    products_df.to_csv(data_dir / "products.csv", index=False)

    # ── customers (10 000) ───────────────────────────────────────────
    customer_id = np.arange(1, N_CUSTOMERS + 1)
    country = rng.choice(COUNTRIES, N_CUSTOMERS, p=[0.40, 0.20, 0.15, 0.10, 0.15])
    tier    = rng.choice(TIERS,     N_CUSTOMERS, p=[0.20, 0.50, 0.30])

    base_signup    = np.datetime64("2020-01-01")
    signup_offsets = rng.integers(0, 1461, N_CUSTOMERS)
    signup_date    = [str((base_signup + np.timedelta64(int(d), "D")))[:10]
                      for d in signup_offsets]

    customers_df = pd.DataFrame({
        "customer_id": customer_id,
        "country":     country,
        "tier":        tier,
        "signup_date": signup_date,
    })
    customers_df.to_csv(data_dir / "customers.csv", index=False)

    # ── orders (50 000) ──────────────────────────────────────────────
    order_id   = np.arange(1, N_ORDERS + 1)
    o_customer = rng.integers(1, N_CUSTOMERS + 1, N_ORDERS)
    o_product  = rng.integers(1, N_PRODUCTS + 1,  N_ORDERS)
    quantity   = rng.integers(1, 11, N_ORDERS)
    unit_price = np.round(rng.uniform(5.0, 200.0, N_ORDERS), 2)
    revenue    = np.round(quantity * unit_price, 2)
    status     = rng.choice(STATUSES, N_ORDERS, p=[0.55, 0.30, 0.15])
    region     = rng.choice(REGIONS,  N_ORDERS)

    base_order    = np.datetime64("2024-01-01")
    order_offsets = rng.integers(0, 366, N_ORDERS)
    created_at    = [str((base_order + np.timedelta64(int(d), "D")))[:10]
                     for d in order_offsets]

    orders_df = pd.DataFrame({
        "order_id":    order_id,
        "customer_id": o_customer,
        "product_id":  o_product,
        "quantity":    quantity,
        "unit_price":  unit_price,
        "revenue":     revenue,
        "status":      status,
        "region":      region,
        "created_at":  created_at,
    })
    orders_df.to_csv(data_dir / "orders.csv", index=False)

    print(f"  [generate] products.csv   {len(products_df):,} rows")
    print(f"  [generate] customers.csv  {len(customers_df):,} rows")
    print(f"  [generate] orders.csv     {len(orders_df):,} rows → {data_dir}")


# ── ground truth (computed offline with pandas) ───────────────────────────────

def _compute_answers(data_dir: Path) -> dict:
    orders    = pd.read_csv(data_dir / "orders.csv")
    customers = pd.read_csv(data_dir / "customers.csv")
    products  = pd.read_csv(data_dir / "products.csv")

    oc = orders.merge(customers, on="customer_id", how="inner")
    op = orders.merge(products,  on="product_id",  how="inner")

    return {
        # t1: orders placed by gold-tier customers
        "t1": int((oc["tier"] == "gold").sum()),
        # t2: total revenue from Nike-brand product orders
        "t2": round(float(op.loc[op["brand"] == "Nike", "revenue"].sum()), 2),
        # t3: completed orders placed by US customers
        "t3": int(((oc["country"] == "US") & (oc["status"] == "completed")).sum()),
        # t4: total revenue from Electronics-category orders
        "t4": round(float(op.loc[op["category"] == "Electronics", "revenue"].sum()), 2),
        # t5: country with the highest total revenue from completed orders
        "t5": str(
            oc[oc["status"] == "completed"]
            .groupby("country")["revenue"].sum().idxmax()
        ),
    }


# ── scenario definition ───────────────────────────────────────────────────────

_DATA_DIR = Path(__file__).parent / "data" / "orders_multi"

if not (_DATA_DIR / "orders.csv").exists():
    generate_data(_DATA_DIR)

_GT = _compute_answers(_DATA_DIR)

scenario = Scenario(
    id="orders_multi",
    name="Orders — multi-table joins",
    description=(
        "Three tables (orders 50k, customers 10k, products 1k). "
        "All tasks require join_tables before any filter or aggregate."
    ),
    data_dir=_DATA_DIR,
    generate_data=generate_data,
    tasks=[
        # train: cross-table count + cross-table sum
        Task("t1",
             "How many orders were placed by customers with a 'gold' membership tier? "
             "Give the exact count.",
             str(_GT["t1"]), "train",
             "join_tables(orders, customers, on=customer_id) + filter_count(tier==gold)"),
        Task("t2",
             "What is the total revenue from orders for products made by the brand 'Nike'? "
             "Round to 2 decimal places.",
             str(_GT["t2"]), "train",
             "join_tables(orders, products, on=product_id) + column_sum(revenue, brand==Nike)"),
        # val: cross-table count + cross-table sum (different join target each)
        # val ceiling = 0% until join_tables is in the scaffold
        Task("t3",
             "How many 'completed' orders were placed by customers whose country is 'US'? "
             "Give the exact count.",
             str(_GT["t3"]), "val",
             "join_tables(orders, customers) + filter_count(country==US, status==completed)"),
        Task("t4",
             "What is the total revenue from orders for products in the 'Electronics' category? "
             "Round to 2 decimal places.",
             str(_GT["t4"]), "val",
             "join_tables(orders, products) + column_sum(revenue, category==Electronics)"),
        # test: join + group + rank — held out until final eval
        Task("t5",
             "Which country has the highest total revenue from 'completed' orders? "
             "Return only the country code (e.g. US, UK, CA, AU, DE).",
             _GT["t5"], "test",
             "join_tables(orders, customers) + group_aggregate(country, revenue, sum, status==completed) + idxmax"),
    ],
)

register(scenario)
