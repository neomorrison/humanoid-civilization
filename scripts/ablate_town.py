"""Controlled comparisons: switch one ingredient of the town off and see what behaviour changes.

    python scripts/ablate_town.py --base runs/town/checkpoint.pkl --iters 120 --seeds 3

Every variant (and an unchanged baseline) continues training from the same checkpoint for the
same number of iterations, then records one 60-year replay per seed and scores it with
civ/metrics.py. Writes results/ablations.json and results/ablations.md.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.mappo import MAPPOConfig, train  # noqa: E402
from civ.metrics import replay_metrics  # noqa: E402
from civ.record_town import record  # noqa: E402
from civ.town import Town, TownConfig  # noqa: E402

VARIANTS = {
    "baseline": {},
    "no_docking": {"dock": 0.0},
    "no_lunch_break": {"lunch_break": False},
    "no_police": {"police": False},
    "no_gossip": {"gossip_rate": 0.0},
}
SHOW = ["attendance", "punctuality", "lunch_in_break", "home_at_night", "docked_share", "child_survival",
        "school_attendance", "family_meals_per_day", "money_gini", "thefts_per_100_person_years", "arrest_rate",
        "hunger", "energy", "years"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="runs/town/checkpoint.pkl")
    ap.add_argument("--iters", type=int, default=120)
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--worlds", type=int, default=64)
    ap.add_argument("--out", default="runs/ablate")
    ap.add_argument("--only", nargs="*", default=None)
    args = ap.parse_args()
    results = {}
    path = os.path.join("results", "ablations.json")
    if os.path.exists(path):
        results = json.load(open(path))
    for name, over in VARIANTS.items():
        if args.only and name not in args.only:
            continue
        cfg = TownConfig(**over)
        out = os.path.join(args.out, name)
        env = Town(args.worlds, cfg, seed=1)
        t0 = time.time()
        # continue from the same checkpoint for a fixed number of iterations (minutes is only a cap)
        train(env, out, 10_000, MAPPOConfig(hidden=(256, 256), gamma=0.995, horizon=64, entropy_coef=0.015),
              seed=1, every=10_000, resume=args.base, max_iters=args.iters)
        runs = [replay_metrics(record(os.path.join(out, "policy.npz"), cfg, seed=100 + k, years=60)) for k in range(args.seeds)]
        agg = {}
        for key in SHOW:
            vals = [r[key] for r in runs if r.get(key) is not None]
            agg[key] = [round(float(np.mean(vals)), 3), round(float(np.std(vals)), 3)] if vals else None
        results[name] = dict(overrides=over, iters=args.iters, seeds=args.seeds, metrics=agg,
                             minutes=round((time.time() - t0) / 60, 1))
        os.makedirs("results", exist_ok=True)
        json.dump(results, open(path, "w"), indent=1)
        print(name, json.dumps(agg), flush=True)
    lines = ["| metric | " + " | ".join(results) + " |", "|---|" + "---|" * len(results)]
    for key in SHOW:
        row = [f"{results[v]['metrics'][key][0]:.3g} ± {results[v]['metrics'][key][1]:.2g}" if results[v]["metrics"].get(key)
               else "–" for v in results]
        lines.append(f"| {key} | " + " | ".join(row) + " |")
    open(os.path.join("results", "ablations.md"), "w").write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
