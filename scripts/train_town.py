"""Train Town Life (v3) and record a whole-era hourly replay at every snapshot.

    python scripts/train_town.py --minutes 600 --out runs/town
"""
from __future__ import annotations

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.mappo import MAPPOConfig, train  # noqa: E402
from civ.record_town import record, save  # noqa: E402
from civ.town import YEAR, Town, TownConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worlds", type=int, default=96)
    ap.add_argument("--minutes", type=float, default=600)
    ap.add_argument("--out", default="runs/town")
    ap.add_argument("--every", type=int, default=50)
    ap.add_argument("--years", type=float, default=60)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", default=None)
    ap.add_argument("--init-from", default=None, help="warm start from a checkpoint trained on a different layout")
    args = ap.parse_args()
    cfg = TownConfig()
    env = Town(args.worlds, cfg, seed=args.seed)
    os.makedirs(os.path.join(args.out, "replays"), exist_ok=True)

    def on_snapshot(path, it, samples):
        rec = record(path, cfg, seed=7, years=args.years)
        save(rec, os.path.join(args.out, "replays", f"replay_it{it:05d}.json"))
        s = rec["summary"]
        print(f"[replay it={it}] years={len(rec['frames']) / YEAR:.1f} births={s['births']:.0f} deaths={s['deaths']:.0f} "
              f"starved={s['starved']:.0f} work_h={s['work_hours']:.0f} docked_h={s['docked_hours']:.0f} "
              f"lunches={s['meals_canteen']:.0f} groceries={s['groceries']:.0f} family_meals={s['family_meals']:.0f} "
              f"school_h={s['school_hours']:.0f} sleep_home={s['sleep_home']:.0f} partnerships={s['partnerships']:.0f} "
              f"thefts={s['thefts']:.0f}", flush=True)

    train(env, args.out, args.minutes, MAPPOConfig(hidden=(256, 256), gamma=0.995, horizon=64, entropy_coef=0.015),
          seed=args.seed, every=args.every, on_snapshot=on_snapshot, resume=args.resume, init_from=args.init_from)


if __name__ == "__main__":
    main()
