"""Record episodes of a trained (or untrained) economy for the replay viewer."""
from __future__ import annotations

import json
from dataclasses import asdict

import numpy as np

from .economy import ROLE_NAMES, Economy, EconomyConfig
from .mappo import NumpyPolicy


def record_episode(policy_path: str | None, cfg: EconomyConfig | None = None, seed: int = 123) -> dict:
    cfg = cfg or EconomyConfig()
    env = Economy(1, cfg, seed=seed, record=True)
    pol = NumpyPolicy(policy_path) if policy_path else None
    rng = np.random.default_rng(seed)
    obs, _ = env.observe_all()
    frames, events = [env.snapshot(0)], []
    for t in range(cfg.episode_steps):
        a = pol.sample(obs, rng) if pol else rng.integers(0, env.act_dim, env.N)
        stats_before = len(env.stats)
        obs, _, _, done, info = env.step(a)
        for (m, kind, i, j, amt) in info["events"]:
            events.append([t + 1, kind, int(i), int(j), amt])
        if done[0]:
            summary = env.stats[stats_before]
            break
        frames.append(env.snapshot(0))
    meta = pol.meta if pol else dict(iteration=0, samples=0)
    return dict(meta=meta, config=asdict(cfg), roles=[ROLE_NAMES[r] for r in cfg.roles],
                plots=env.plots.tolist(), frames=frames, events=events, summary=summary)


def save(rec: dict, path: str):
    with open(path, "w") as f:
        json.dump(rec, f, separators=(",", ":"))
