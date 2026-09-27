"""Train Society v2 and record a multi-year replay at every snapshot.

    python scripts/train_society.py --minutes 240 --out runs/society
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.mappo import MAPPOConfig, train  # noqa: E402
from civ.record_society import record, save  # noqa: E402
from civ.society import Society, SocietyConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=96)
    ap.add_argument("--minutes", type=float, default=240)
    ap.add_argument("--out", default="runs/society")
    ap.add_argument("--every", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    args = ap.parse_args()
    cfg = SocietyConfig()
    env = Society(args.worlds, cfg, seed=args.seed)
    os.makedirs(os.path.join(args.out, "replays"), exist_ok=True)

    def on_snapshot(path, it, samples):
        rec = record(path, cfg, seed=7, years=60, stride=3)
        save(rec, os.path.join(args.out, "replays", f"replay_it{it:05d}.json"))
        s = rec["summary"]
        print(f"[replay it={it}] years={len(rec['frames']) * rec['stride'] / 60:.1f} births={s['births']:.0f} "
              f"deaths={s['deaths']:.0f} starved={s['starved']:.0f} crates={s['crates']:.0f} "
              f"harvests={s['harvest_grain'] + s['harvest_fruit'] + s['harvest_dairy']:.0f} "
              f"trades={s['trades']:.0f} food_gifts={s['gifts_food']:.0f} thefts={s['thefts']:.0f} "
              f"partnerships={s['partnerships']:.0f}", flush=True)

    train(env, args.out, args.minutes, MAPPOConfig(hidden=(256, 256), gamma=0.997, horizon=64, entropy_coef=0.015),
          seed=args.seed, every=args.every, on_snapshot=on_snapshot, resume=args.resume)


if __name__ == "__main__":
    main()
