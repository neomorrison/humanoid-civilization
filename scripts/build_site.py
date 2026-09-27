"""Assemble the replay viewer site from a training run.

    python scripts/build_site.py --run runs/economy --out site

Writes site/index.html (full document, for GitHub Pages), site/replays/*.json
and site/replays/index.json (stages + training curves).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import shutil

import numpy as np

HERE = os.path.dirname(__file__)
HEAD = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<meta name="description" content="Learning agents build a small economy by choice: warehouse jobs, farming, hand-to-hand trade, theft and policing.">
</head>
<body>
"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/economy")
    ap.add_argument("--out", default="site")
    ap.add_argument("--max-stages", type=int, default=10)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out, "replays"), exist_ok=True)
    files = sorted(glob.glob(os.path.join(args.run, "replays", "replay_it*.json")),
                   key=lambda p: int(re.findall(r"it(\d+)", p)[0]))
    if len(files) > args.max_stages:
        keep = np.unique(np.linspace(0, len(files) - 1, args.max_stages).round().astype(int))
        files = [files[i] for i in keep]
    stages = []
    for p in files:
        rec = json.load(open(p))
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(args.out, "replays", name))
        stages.append(dict(it=rec["meta"].get("iteration", 0), samples=rec["meta"].get("samples", 0), file=name,
                           summary=rec["summary"]))
    curve = []
    log = os.path.join(args.run, "train_log.jsonl")
    if os.path.exists(log):
        recs = [json.loads(l) for l in open(log)]
        recs = [r for r in recs if r.get("econ")]
        step = max(1, len(recs) // 120)
        for k in range(0, len(recs), step):
            win = recs[max(0, k - step):k + 1]
            e = {key: float(np.mean([r["econ"][key] for r in win])) for key in win[-1]["econ"]}
            curve.append(dict(it=recs[k]["it"], **{kk: round(v, 2) for kk, v in e.items()}))
    with open(os.path.join(args.out, "replays", "index.json"), "w") as f:
        json.dump(dict(stages=stages, curve=curve), f)
    frag = open(os.path.join(HERE, "..", "viewer", "viewer.html")).read()
    with open(os.path.join(args.out, "index.html"), "w") as f:
        f.write(HEAD + frag + "\n</body>\n</html>\n")
    shutil.copy(os.path.join(HERE, "..", "viewer", "viewer.html"), os.path.join(args.out, "viewer.fragment.html"))
    print(f"site: {len(stages)} stages, {len(curve)} curve points -> {args.out}")


if __name__ == "__main__":
    main()
