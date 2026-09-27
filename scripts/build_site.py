"""Assemble the replay viewer site from training runs.

    python scripts/build_site.py --run runs/society --v1 runs/economy --out site

Writes site/index.html (Society v2 3D viewer) with site/replays/*.json and
site/replays/index.json (stages + training curves). When --v1 is given the
first economy is kept at site/economy.html (3D) and site/v1/plan.html (2D),
reading site/v1/replays/.
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


def build_replays(run, out_dir, max_stages, sub="replays", note=""):
    """Copy a spread of replay snapshots and write index.json with the training curve."""
    os.makedirs(out_dir, exist_ok=True)
    for old in glob.glob(os.path.join(out_dir, "replay_it*.json")):
        os.remove(old)
    files = sorted(glob.glob(os.path.join(run, sub, "replay_it*.json")),
                   key=lambda p: int(re.findall(r"it(\d+)", p)[0]))
    if len(files) > max_stages:
        keep = np.unique(np.linspace(0, len(files) - 1, max_stages).round().astype(int))
        files = [files[i] for i in keep]
    stages = []
    for p in files:
        rec = json.load(open(p))
        name = os.path.basename(p)
        shutil.copy(p, os.path.join(out_dir, name))
        stages.append(dict(it=rec["meta"].get("iteration", 0), samples=rec["meta"].get("samples", 0), file=name,
                           summary=rec["summary"]))
    curve = []
    log = os.path.join(run, "train_log.jsonl")
    if os.path.exists(log):
        recs = [json.loads(l) for l in open(log)]
        recs = [r for r in recs if r.get("econ")]
        step = max(1, len(recs) // 120)
        for k in list(range(0, len(recs), step)) + ([len(recs) - 1] if recs and (len(recs) - 1) % step else []):
            win = recs[max(0, k - step):k + 1]
            e = {key: float(np.mean([r["econ"][key] for r in win if key in r["econ"]])) for key in win[-1]["econ"]}
            curve.append(dict(it=recs[k]["it"], **{kk: round(v, 2) for kk, v in e.items()}))
    with open(os.path.join(out_dir, "index.json"), "w") as f:
        json.dump(dict(stages=stages, curve=curve, note=note), f)
    return len(stages), len(curve)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/society")
    ap.add_argument("--v1", default=None, help="economy v1 run to keep at economy.html")
    ap.add_argument("--out", default="site")
    ap.add_argument("--max-stages", type=int, default=10)
    ap.add_argument("--replays", default="replays", help="replay folder inside --run (e.g. replays60)")
    ap.add_argument("--note", default="", help="shown under the training curves (e.g. when the rules changed)")
    args = ap.parse_args()
    v3 = os.path.join(HERE, "..", "viewer3d")
    ns, nc = build_replays(args.run, os.path.join(args.out, "replays"), args.max_stages, args.replays, args.note)
    shutil.copy(os.path.join(v3, "index.html"), os.path.join(args.out, "index.html"))
    shutil.copy(os.path.join(v3, "CREDITS.md"), os.path.join(args.out, "CREDITS.md"))
    for sub in ("assets", "vendor"):
        dst = os.path.join(args.out, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(v3, sub), dst)
    msg = f"site: society {ns} stages, {nc} curve points"
    if args.v1:
        n1, c1 = build_replays(args.v1, os.path.join(args.out, "v1", "replays"), args.max_stages)
        shutil.copy(os.path.join(v3, "economy.html"), os.path.join(args.out, "economy.html"))
        frag = open(os.path.join(HERE, "..", "viewer", "viewer.html")).read()
        with open(os.path.join(args.out, "v1", "plan.html"), "w") as f:
            f.write(HEAD + frag + "\n</body>\n</html>\n")
        msg += f"; economy v1 {n1} stages, {c1} curve points"
    open(os.path.join(args.out, ".nojekyll"), "w").close()
    print(msg + f" -> {args.out}")


if __name__ == "__main__":
    main()
