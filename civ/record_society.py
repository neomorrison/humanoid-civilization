"""Record a stretch of a trained society for the 3D viewer (compact format)."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .mappo import NumpyPolicy
from .society import YEAR, Society, SocietyConfig

FRAME_KEYS = ["alive", "pos", "age", "hunger", "health", "money", "food", "nutr", "skill", "carry", "pregnant",
              "partner", "home", "enlisted", "detained", "seen_thief", "regard", "pid"]


def record(policy_path: str | None, cfg: SocietyConfig | None = None, seed: int = 7, years: float = 20.0,
           stride: int = 2) -> dict:
    cfg = cfg or SocietyConfig()
    env = Society(1, cfg, seed=seed, record=True)
    pol = NumpyPolicy(policy_path) if policy_path else None
    rng = np.random.default_rng(seed)
    obs, _ = env.observe_all()
    people, frames, events = {}, [], []

    def note_people():
        snap = env.snapshot(0)
        for s in range(cfg.slots):
            if snap["alive"][s] and snap["pid"][s] not in people:
                people[snap["pid"][s]] = dict(sex=snap["sex"][s], mother=snap["mother"][s], father=snap["father"][s],
                                             traits=snap["traits"][s], born_step=int(env.t[0]) - int(snap["age"][s] * YEAR))
        return snap

    steps = int(years * YEAR)
    ended = None
    for t in range(steps):
        if t % stride == 0:
            snap = note_people()
            frames.append({k: snap[k] for k in FRAME_KEYS} | {"t": snap["t"], "sites": snap["sites"]})
        a = pol.sample(obs, rng) if pol else rng.integers(0, env.act_dim, env.N)
        obs, _, _, done, info = env.step(a)
        for (m, kind, i, j, amt) in info["events"]:
            events.append([t + 1, kind, int(env.pid[0, i]) if i >= 0 else 0,
                           int(env.pid[0, j]) if j >= 0 else 0, amt])
        if env.t[0] == 0:            # the world ended (extinction or era end)
            ended = env.stats[-1]
            break
    summary = ended or {k: float(v[0]) for k, v in env.ep.items()}
    meta = pol.meta if pol else dict(iteration=0, samples=0)
    return dict(kind="society", meta=meta, config=asdict(cfg), stride=stride, year=YEAR, people=people,
                sites=env.sites.tolist(), site_type=env.site_type.tolist(), foods=["grain", "fruit", "dairy"],
                frames=frames, events=events, summary=summary)


def save(rec, path):
    with open(path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
