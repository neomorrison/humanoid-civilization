"""Record one Town Life world hour by hour in the format of docs/TOWN_REPLAY.md."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .mappo import NumpyPolicy
from .mappo_rnn import NumpyRNNPolicy
from .town import ACTIVITIES, DAY, DAYS_PER_YEAR, FOODS, JOBS, SEASONS, YEAR, Town, TownConfig

SLOT_COLS = ["pid", "x", "y", "act", "age10", "hunger", "energy", "health", "money", "g", "f", "d", "job", "home",
             "partner", "flags"]


def _looks(t: int, herd_staff: int) -> tuple:
    """How the fields, orchard and pasture look at absolute hour t (0-9 each)."""
    hour = t % DAY
    season = (t // DAY) % DAYS_PER_YEAR
    u = hour / DAY
    crops = [1 + 3 * u, 5 + 3 * u, 9 - 6 * u, 0][season]
    fruit = [1 + 2 * u, 3 + 4 * u, 9 - 7 * u, 0][season]
    day = 6 <= hour < 20
    herd = (7 + min(2, herd_staff)) if day and season < 3 else (3 if day else 1)
    return int(round(crops)), int(round(fruit)), int(herd)


def frame(env: Town, m: int = 0) -> dict:
    c = env.cfg
    t = int(env.t[m]) - 1                       # the hour that was just lived
    hour, season = t % DAY, (t // DAY) % DAYS_PER_YEAR
    S = env.S
    kid = env.age[m] < c.child_age
    on_shift = env.shift_on[env.job[m], hour, season] & (env.job[m] > 0)
    rows = []
    for s in range(S):
        if not env.alive[m, s]:
            rows.append(0)
            continue
        flags = (1 * (env.pregnant[m, s] > 0) | 2 * (env.detained[m, s] > 0) | 4 * env.enlisted[m, s] | 8 * env.asleep[m, s]
                 | 16 * (env.saw_theft[m, :, s].max() > 0) | 32 * on_shift[s] | 64 * env.docked_now[m, s] | 128 * kid[s])
        rows.append([int(env.pid[m, s]), int(env.pos[m, s, 0]), int(env.pos[m, s, 1]), int(env.act[m, s]),
                     int(round(env.age[m, s] * 10)), int(round(env.hunger[m, s] * 100)), int(round(env.energy[m, s] * 100)),
                     int(round(env.health[m, s] * 100)), int(round(env.money[m, s])), *[int(v) for v in env.food[m, s]],
                     int(env.job[m, s]), int(env.home[m, s]), int(env.partner[m, s]), int(flags)])
    open_, _ = env.canteen_open()
    dairy_staff = int((env.at_work()[m] & (env.job[m] == JOBS.index("dairy"))).sum())
    crops, fruit, herd = _looks(t, dairy_staff)
    shop = int(c.shop_hours[0] <= hour < c.shop_hours[1])
    w = {"stock": [int(round(v)) for v in env.stock[m]], "treasury": int(round(env.treasury[m])),
         "canteen_open": int(open_[m]), "shop_open": shop, "crops": crops, "fruit": fruit, "herd": herd,
         "price": [float(p) for p in env.canteen_price()[m]]}
    return {"t": t, "s": rows, "w": w}


def load_policy(path):
    """A feed-forward or recurrent numpy policy, whichever the file holds."""
    z = np.load(path)
    meta = json.loads(bytes(z["meta_json"]).decode())
    return NumpyRNNPolicy(path) if meta.get("recurrent") else NumpyPolicy(path)


def record(policy_path=None, cfg: TownConfig | None = None, seed: int = 7, years: float = 60.0) -> dict:
    """policy_path: a policy .npz, a callable env -> actions (e.g. a scripted routine), or None for random."""
    cfg = cfg or TownConfig()
    env = Town(1, cfg, seed=seed, record=True)
    pol = load_policy(policy_path) if isinstance(policy_path, str) else None
    fn = policy_path if callable(policy_path) else None
    rng = np.random.default_rng(seed)
    obs, _, _ = env.observe()
    people, frames, events = {}, [], []

    def note_people():
        for s in range(env.S):
            if env.alive[0, s]:
                pid = int(env.pid[0, s])
                if pid not in people:
                    people[pid] = dict(sex=int(env.sex[0, s]), mother=int(env.mother[0, s]), father=int(env.father[0, s]),
                                       traits=np.round(env.traits[0, s], 2).tolist(),
                                       born_step=int(env.t[0] - round(env.age[0, s] * YEAR)))

    start = int(env.t[0])
    ended = None
    for _ in range(int(years * YEAR)):
        note_people()
        before = env.pid[0].copy()
        if pol is not None and hasattr(pol, "reset"):
            pol.reset(env.reset_mask())
        a = pol.sample(obs, rng) if pol else fn(env) if fn else rng.integers(0, env.act_dim, env.N)
        obs, _, _, _, info = env.step(a)
        after = env.pid[0]
        if env.t[0] == cfg.start_hour:            # the era ended and the world was reset
            ended = env.stats[-1]
            break
        frames.append(frame(env))
        for (m, kind, i, j, amt) in info["events"]:
            pi = int(after[i]) if kind == "born" else int(before[i]) if i >= 0 else 0
            pj = int(before[j]) if j >= 0 else 0
            events.append([int(env.t[0]) - 1, kind, pi, pj, float(amt) if isinstance(amt, float) else int(amt)])
    note_people()
    summary = ended or {k: float(v[0]) for k, v in env.ep.items()}
    meta = pol.meta if pol else dict(iteration=0, samples=0, scripted=bool(fn))
    conf = asdict(cfg)
    conf.update(day_steps=DAY, days_per_year=DAYS_PER_YEAR, seasons=SEASONS)
    return dict(kind="town", version=3, meta=meta, config=conf, start_step=start, stride=1,
                activities=ACTIVITIES, jobs=JOBS, foods=FOODS, people={str(k): v for k, v in people.items()},
                slot_cols=SLOT_COLS, frames=frames, events=events, summary=summary)


def save(rec: dict, path: str):
    with open(path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
