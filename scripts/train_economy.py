"""Train the economy's agents and record a replay at every snapshot.

    python scripts/train_economy.py --minutes 60 --out runs/economy

Replays land in runs/economy/replays/replay_itNNNNN.json; build the viewer
with scripts/build_viewer.py to watch the agents learn.
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.economy import Economy, EconomyConfig  # noqa: E402
from civ.mappo import MAPPOConfig, train  # noqa: E402
from civ.record import record_episode, save  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=256)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default="runs/economy")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--no-police", action="store_true")
    ap.add_argument("--no-theft", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = EconomyConfig()
    if args.no_police:
        cfg.roles = tuple(r for r in cfg.roles if r != 2) + (0,)
    if args.no_theft:
        cfg.theft_enabled = False
    env = Economy(args.worlds, cfg, seed=args.seed)
    os.makedirs(os.path.join(args.out, "replays"), exist_ok=True)

    def on_snapshot(path, it, samples):
        rec = record_episode(path, cfg, seed=123)
        save(rec, os.path.join(args.out, "replays", f"replay_it{it:05d}.json"))
        s = rec["summary"]
        print(f"[replay it={it}] boxes {s['boxes']:.0f} food sold {s['food_sold']:.0f} thefts {s['thefts']:.0f} "
              f"arrests {s['arrests']:.0f} starving-steps {s['starving']:.0f}", flush=True)

    train(env, args.out, args.minutes, MAPPOConfig(), seed=args.seed, every=args.every,
          on_snapshot=on_snapshot, resume=args.resume)


if __name__ == "__main__":
    main()
