"""
CLI for scenario management.

  python cli.py list
  python cli.py show orders_basic
  python cli.py generate orders_basic
  python cli.py run orders_basic [--generations N]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import scenarios as sc


def cmd_list(_args) -> None:
    sc.load_all()
    rows = sc.list_all()
    if not rows:
        print("No scenarios registered.")
        return
    print(f"{'ID':<20} {'Name':<35} Tasks")
    print("-" * 65)
    for s in rows:
        n = len(s.tasks)
        print(f"{s.id:<20} {s.name:<35} {n} ({len(s.train_tasks)}tr/{len(s.val_tasks)}v/{len(s.test_tasks)}t)")


def cmd_show(args) -> None:
    sc.load_all()
    s = sc.get(args.scenario)
    print(f"\nScenario : {s.name}")
    print(f"ID       : {s.id}")
    print(f"Desc     : {s.description}")
    print(f"Data dir : {s.data_dir}")
    csv_files = list(s.data_dir.glob("*.csv")) if s.data_dir.exists() else []
    if csv_files:
        import pandas as pd
        for f in csv_files:
            n = sum(1 for _ in open(f)) - 1
            cols = ", ".join(pd.read_csv(f, nrows=0).columns)
            print(f"  {f.name}  ({n:,} rows)  [{cols}]")
    else:
        print("  (data not yet generated — run: python cli.py generate <id>)")
    print(f"\nTasks ({len(s.tasks)}):")
    for t in s.tasks:
        print(f"  [{t.split:5}] {t.id}: {t.question[:70]}")
        print(f"           answer: {t.answer}")


def cmd_generate(args) -> None:
    sc.load_all()
    s = sc.get(args.scenario)
    print(f"Generating data for '{s.id}' …")
    s.generate_data(s.data_dir)
    print("Done.")


def cmd_run(args) -> None:
    from evolve import evolve
    evolve(args.scenario, n_generations=args.generations)


def main():
    p = argparse.ArgumentParser(prog="cli")
    sub = p.add_subparsers(dest="cmd")

    sub.add_parser("list", help="List registered scenarios")

    ps = sub.add_parser("show", help="Show scenario details")
    ps.add_argument("scenario")

    pg = sub.add_parser("generate", help="Generate data for a scenario")
    pg.add_argument("scenario")

    pr = sub.add_parser("run", help="Run evolution on a scenario")
    pr.add_argument("scenario")
    pr.add_argument("--generations", type=int, default=4)

    args = p.parse_args()
    if args.cmd == "list":       cmd_list(args)
    elif args.cmd == "show":     cmd_show(args)
    elif args.cmd == "generate": cmd_generate(args)
    elif args.cmd == "run":      cmd_run(args)
    else:
        p.print_help()


if __name__ == "__main__":
    main()
