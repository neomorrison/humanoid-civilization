"""Record full-length replays from a run's policy snapshots.

    python scripts/record_stages.py --run runs/society --years 60 --stages 8

Writes <run>/replays60/replay_itNNNNN.json for an even spread of snapshots
(always including the first and the latest), for scripts/build_site.py --replays replays60.
"""
from __future__ import annotations

import argparse
import glob
import os
import re
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.record_society import record, save  # noqa: E402
from civ.society import SocietyConfig  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/society")
    ap.add_argument("--years", type=float, default=60)
    ap.add_argument("--stride", type=int, default=3)
    ap.add_argument("--stages", type=int, default=8)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    snaps = sorted(glob.glob(os.path.join(args.run, "snapshots", "policy_it*.npz")),
                   key=lambda p: int(re.findall(r"it(\d+)", p)[0]))
    if len(snaps) > args.stages:
        snaps = [snaps[i] for i in np.unique(np.linspace(0, len(snaps) - 1, args.stages).round().astype(int))]
    out = os.path.join(args.run, "replays60")
    os.makedirs(out, exist_ok=True)
    for p in snaps:
        it = int(re.findall(r"it(\d+)", p)[0])
        dst = os.path.join(out, f"replay_it{it:05d}.json")
        if os.path.exists(dst):
            continue
        rec = record(p, SocietyConfig(), seed=args.seed, years=args.years, stride=args.stride)
        save(rec, dst)
        s = rec["summary"]
        print(f"it={it} years={len(rec['frames']) * rec['stride'] / 60:.1f} births={s['births']:.0f} deaths={s['deaths']:.0f} "
              f"alive_end={sum(rec['frames'][-1]['alive'])}", flush=True)


if __name__ == "__main__":
    main()
