"""Assemble the replay viewer site from training runs.

    python scripts/build_site.py --run runs/town --v2 runs/society --v2-replays replays60 --v1 runs/economy --out site

Writes site/index.html (Town Life v3 3D viewer) with site/replays/*.json and
site/replays/index.json (stages + training curves). When --v2 is given the
Society is kept at site/society.html with its replays in site/v2/replays/. When --v1 is given the
first economy is kept at site/economy.html (3D) and site/v1/plan.html (2D),
reading site/v1/replays/.

The live sandbox (site/index.html?live=1) is built into site/live/ whenever the town policy exists:
the simulator's python files (numpy only, run in the browser by Pyodide) and the trained policies
(--live-town, and --live-evil for harm-seekers when there is one). --no-live leaves it out.
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
LIVE_CIV = ["__init__.py", "numpy_policy.py", "town.py", "record_town.py", "sandbox.py"]   # sandbox + its imports
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


def build_live(out_dir, town_policy, evil_policy):
    """site/live/: what the browser needs to run civ/sandbox.py (see viewer3d/town/live-worker.js)."""
    civ = os.path.join(HERE, "..", "civ")
    for f in LIVE_CIV:     # the browser has numpy only: every module the sandbox imports must be shipped
        for dep in re.findall(r"^\s*from \.(\w+) import", open(os.path.join(civ, f)).read(), re.M):
            assert dep + ".py" in LIVE_CIV, f"civ/{f} imports civ/{dep}.py, which the live sandbox does not ship"
    if os.path.exists(out_dir):
        shutil.rmtree(out_dir)
    os.makedirs(os.path.join(out_dir, "civ"))
    for f in LIVE_CIV:
        shutil.copy(os.path.join(civ, f), os.path.join(out_dir, "civ", f))

    def policy(src, name):
        dst = os.path.join(out_dir, name)
        shutil.copy(src, dst)
        z = np.load(dst)           # a policy still being trained may be caught mid-write: check the copy
        return json.loads(bytes(z["meta_json"]).decode())

    meta = dict(town=policy(town_policy, "town.npz"), evil=None)
    if evil_policy and os.path.exists(evil_policy):
        try:
            meta["evil"] = policy(evil_policy, "evil.npz")
        except Exception as err:    # harm-seekers then act with the town policy and evil traits
            print(f"skipping harm-seeker policy {evil_policy}: {err}")
            os.remove(os.path.join(out_dir, "evil.npz"))
    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(dict(civ=LIVE_CIV, town="town.npz", evil="evil.npz" if meta["evil"] else None, meta=meta), f)
    it = lambda m: m.get("iteration", "?") if m else None     # noqa: E731
    return f"live sandbox (town it {it(meta['town'])}, harm-seeker {'it ' + str(it(meta['evil'])) if meta['evil'] else 'none'})"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/town")
    ap.add_argument("--v2", default=None, help="society v2 run to keep at society.html")
    ap.add_argument("--v2-replays", default="replays60")
    ap.add_argument("--v1", default=None, help="economy v1 run to keep at economy.html")
    ap.add_argument("--out", default="site")
    ap.add_argument("--max-stages", type=int, default=10)
    ap.add_argument("--replays", default="replays", help="replay folder inside --run (e.g. replays60)")
    ap.add_argument("--note", default="", help="shown under the training curves (e.g. when the rules changed)")
    ap.add_argument("--live", action=argparse.BooleanOptionalAction, default=True,
                    help="build the live sandbox into <out>/live/ (default: when --live-town exists)")
    ap.add_argument("--live-town", default="runs/town/policy.npz", help="townspeople policy for the live sandbox")
    ap.add_argument("--live-evil", default="runs/evil/policy.npz", help="harm-seeker policy (optional)")
    args = ap.parse_args()
    v3 = os.path.join(HERE, "..", "viewer3d")
    ns, nc = build_replays(args.run, os.path.join(args.out, "replays"), args.max_stages, args.replays, args.note)
    shutil.copy(os.path.join(v3, "town.html"), os.path.join(args.out, "index.html"))
    shutil.copy(os.path.join(v3, "CREDITS.md"), os.path.join(args.out, "CREDITS.md"))
    for sub in ("assets", "vendor", "town"):
        dst = os.path.join(args.out, sub)
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(os.path.join(v3, sub), dst)
    msg = f"site: town {ns} stages, {nc} curve points"
    if args.live and os.path.exists(args.live_town):
        msg += "; " + build_live(os.path.join(args.out, "live"), args.live_town, args.live_evil)
    elif os.path.exists(os.path.join(args.out, "live")):
        shutil.rmtree(os.path.join(args.out, "live"))
    if args.v2:
        n2, c2 = build_replays(args.v2, os.path.join(args.out, "v2", "replays"), args.max_stages, args.v2_replays)
        shutil.copy(os.path.join(v3, "society.html"), os.path.join(args.out, "society.html"))
        msg += f"; society v2 {n2} stages, {c2} curve points"
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
