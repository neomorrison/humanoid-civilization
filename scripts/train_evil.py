"""Train a harm-seeker: one newcomer per town whose only reward is everyone else's unhappiness.

    python scripts/train_evil.py --town runs/town/policy.npz --minutes 60 --out runs/evil

The townspeople act with a fixed, already trained policy; only the newcomer learns. It has
no special powers, just the actions everyone has (theft, gossip, shunning, courting and
leaving, buying up food, taking jobs, ...), so whatever harm it finds is a strategy it
discovered. When it dies, another arrives a day later.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.mappo import MAPPOConfig, train  # noqa: E402
from civ.record_town import load_policy  # noqa: E402
from civ.town import DAY, Town, TownConfig  # noqa: E402


class EvilTown:
    """The learner controls one harm-seeker per world; everyone else follows a fixed policy."""

    def __init__(self, town_policy: str, n_worlds: int = 32, seed: int = 0, harm_scale: float = 0.3):
        self.town = Town(n_worlds, TownConfig(), seed=seed)
        self.pol = load_policy(town_policy)
        self.rng = np.random.default_rng(seed + 1)
        self.M, self.S = self.town.M, self.town.S
        self.N = self.M
        self.harm_scale = harm_scale
        self.slot = np.full(self.M, -1)
        self.wait = np.zeros(self.M, int)
        self.obs_dim, self.cobs_dim, self.act_dim = self.town.obs_dim, self.town.cobs_dim, self.town.act_dim
        self.obs_names, self.cobs_names, self.act_names = self.town.obs_names, self.town.cobs_names, self.town.act_names
        self.stats = []
        self.harm_events = np.zeros(self.M)
        for m in range(self.M):
            self._arrive(m)
        self._obs, self._cobs, _ = self.town.observe()

    def _arrive(self, m):
        e, c = self.town, self.town.cfg
        free = np.nonzero(~e.alive[m])[0]
        if not len(free):
            self.slot[m] = -1
            return
        s = int(free[0])
        res = e.residents()[m]
        h = next((k for k in range(e.H) if res[k] < c.house_beds), -1)
        x0, y0, x1, y1 = c.houses[h] if h >= 0 else c.places["market"]
        e._new_person(m, s, int(self.rng.integers(0, 2)), float(self.rng.uniform(20, 35)),
                      np.array([0.0, 1.0, 1.0, self.rng.random()]),
                      [int(self.rng.integers(x0, x1 + 1)), int(self.rng.integers(y0, y1 + 1))])
        e.home[m, s] = h
        self.slot[m] = s

    def _rows(self, arr):
        a = arr.reshape(self.M, self.S, -1)
        return a[np.arange(self.M), np.clip(self.slot, 0, self.S - 1)]

    def observe_all(self):
        return self._rows(self._obs), self._rows(self._cobs)

    def alive_mask(self):
        return ((self.slot >= 0) & self.town.alive[np.arange(self.M), np.clip(self.slot, 0, self.S - 1)]).astype(np.float32)

    def decision_mask(self):
        s = np.clip(self.slot, 0, self.S - 1)
        return (self.alive_mask() > 0) & (self.town.intent[np.arange(self.M), s] < 0)

    def step(self, a_evil):
        e = self.town
        a = self.pol.sample(self._obs, self.rng).reshape(self.M, self.S)
        ok = self.slot >= 0
        a[np.nonzero(ok)[0], self.slot[ok]] = np.asarray(a_evil)[ok]
        was = self.alive_mask() > 0
        n_before = e.ep["thefts"].copy()
        self._obs, self._cobs, rew, done, info = e.step(a.reshape(-1))
        rew = rew.reshape(self.M, self.S)
        others = e.alive.copy()
        others[np.nonzero(ok)[0], self.slot[ok]] = False
        harm = -(rew * others).sum(1) * self.harm_scale                  # everyone else's unhappiness
        self.harm_events += e.ep["thefts"] - n_before
        died = was & ~(self.alive_mask() > 0)
        timeout = np.asarray(info["timeout"]).reshape(self.M, self.S)[np.arange(self.M), np.clip(self.slot, 0, self.S - 1)]
        for m in np.nonzero(died | (e.t == e.cfg.start_hour))[0]:          # died, or the world was reset
            self.slot[m] = -1
            self.wait[m] = DAY
        for m in np.nonzero(self.slot < 0)[0]:
            self.wait[m] -= 1
            if self.wait[m] <= 0:
                self._arrive(m)
        if e.stats:
            self.stats = e.stats
        self._obs, self._cobs, _ = e.observe()
        ob, cob = self.observe_all()
        return ob, cob, np.where(was, harm, 0).astype(np.float32), died.astype(np.float32), \
            dict(timeout=timeout & was, events=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--town", default="runs/town/policy.npz")
    ap.add_argument("--worlds", type=int, default=48)
    ap.add_argument("--minutes", type=float, default=60)
    ap.add_argument("--out", default="runs/evil")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args()
    env = EvilTown(args.town, args.worlds)
    train(env, args.out, args.minutes, MAPPOConfig(hidden=(256, 256), gamma=0.99, horizon=128, entropy_coef=0.01),
          seed=0, every=50, max_iters=args.max_iters)


if __name__ == "__main__":
    main()
