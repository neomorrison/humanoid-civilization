"""A live town you can poke: step it under the trained policy and intervene at any moment.

Designed to run in the browser (Pyodide: numpy only) as well as locally:

    sb = Sandbox("policy.npz", evil_policy="evil.npz")
    frames = sb.run(24)                 # 24 hours -> frames in the docs/TOWN_REPLAY.md format
    sb.kill(pid)                        # someone dies (their family grieves, their job and bed free up)
    sb.spawn(sex=1, age=25)             # a newcomer arrives in town
    sb.spawn(evil=True)                 # a newcomer who only wants to cause harm
    sb.set_rule("dock", 0.0)            # change the world's rules on the fly
"""
from __future__ import annotations

import json
from dataclasses import fields

import numpy as np

from .record_town import SLOT_COLS, frame, load_policy
from .town import ACTIVITIES, DAY, DAYS_PER_YEAR, FOODS, JOBS, SEASONS, YEAR, Town, TownConfig

RULES = {  # what can be changed live, with a short explanation for the UI
    "wage": "coins per hour at work",
    "dock": "coins docked per missing hour, per unit of wage",
    "lunch_break": "shifts have a lunch break",
    "police": "people can volunteer as police",
    "gossip_rate": "how strongly gossip moves opinions",
    "price_base": "base price of food",
    "export": "coins the outside world pays per warehouse hour",
    "hunger_awake": "how fast hunger grows while awake",
    "conceive_prob": "chance per hour that a couple at home conceives",
}


class Sandbox:
    def __init__(self, policy_path: str, evil_policy: str | None = None, seed: int = 0, cfg: TownConfig | None = None):
        self.cfg = cfg or TownConfig()
        self.env = Town(1, self.cfg, seed=seed, record=True)
        self.pol = load_policy(policy_path)
        self.evil_pol = load_policy(evil_policy) if evil_policy else None
        self.evil_pids = set()                          # people who only want to cause harm
        self.rng = np.random.default_rng(seed)
        self.obs, _, _ = self.env.observe()
        self.people = {}
        self.events = []
        self._note_people()

    # --------------------------------------------------------------- running
    def _note_people(self):
        e = self.env
        for s in range(e.S):
            if e.alive[0, s]:
                pid = int(e.pid[0, s])
                if pid not in self.people:
                    self.people[pid] = dict(sex=int(e.sex[0, s]), mother=int(e.mother[0, s]), father=int(e.father[0, s]),
                                            traits=np.round(e.traits[0, s], 2).tolist(),
                                            born_step=int(e.t[0] - round(e.age[0, s] * YEAR)), evil=pid in self.evil_pids)

    def run(self, hours: int = 1) -> list:
        e = self.env
        out = []
        for _ in range(int(hours)):
            reset = e.reset_mask()
            evil = np.isin(e.pid[0], list(self.evil_pids)) & e.alive[0]
            for p in (self.pol, self.evil_pol):
                if p is not None and hasattr(p, "reset"):
                    p.reset(reset)
            a = self.pol.sample(self.obs, self.rng)
            if self.evil_pol is not None and evil.any():
                a = np.where(evil, self.evil_pol.sample(self.obs, self.rng), a)
            before = e.pid[0].copy()
            self.obs, _, _, _, info = e.step(a)
            after = e.pid[0]
            self._note_people()
            fr = frame(e)
            fr["evil"] = sorted(int(p) for p in self.evil_pids if (e.pid[0][e.alive[0]] == p).any())
            out.append(fr)
            for (m, kind, i, j, amt) in info["events"]:
                pi = int(after[i]) if kind == "born" else int(before[i]) if i >= 0 else 0
                pj = int(before[j]) if j >= 0 else 0
                self.events.append([int(e.t[0]) - 1, kind, pi, pj, float(amt) if isinstance(amt, float) else int(amt)])
        return out

    # --------------------------------------------------------- interventions
    def _slot(self, pid):
        hit = np.nonzero((self.env.pid[0] == pid) & self.env.alive[0])[0]
        return int(hit[0]) if len(hit) else None

    def kill(self, pid: int) -> bool:
        s = self._slot(pid)
        if s is None:
            return False
        rew = np.zeros((1, self.env.S))
        ev = []
        self.env._die(0, s, rew, ev, cause="old age")
        t = int(self.env.t[0])
        self.events.append([t, "died (removed)", int(pid), 0, round(float(self.env.age[0, s]), 1)])
        return True

    def spawn(self, sex: int | None = None, age: float = 25.0, evil: bool = False, traits=None) -> int | None:
        e, c = self.env, self.cfg
        free = np.nonzero(~e.alive[0])[0]
        if not len(free):
            return None
        s = int(free[0])
        sex = int(self.rng.integers(0, 2)) if sex is None else int(sex)
        tr = np.asarray(traits, float) if traits is not None else self.rng.random(4)
        if evil:
            tr = np.array([0.0, 1.0, 1.0, tr[3]])                 # no empathy, all greed and boldness
        res = e.residents()[0]
        free_house = [h for h in range(e.H) if res[h] < c.house_beds]
        h = int(free_house[0]) if free_house else -1
        x0, y0, x1, y1 = c.houses[h] if h >= 0 else c.places["market"]
        e._new_person(0, s, sex, float(age), tr, [int(self.rng.integers(x0, x1 + 1)), int(self.rng.integers(y0, y1 + 1))])
        e.home[0, s] = h
        e.energy[0, s] = 0.9
        e.nutr[0, s] = 0.6
        self.obs, _, _ = e.observe()
        pid = int(e.pid[0, s])
        if evil:
            self.evil_pids.add(pid)
        self._note_people()
        self.people[pid]["evil"] = bool(evil)
        self.events.append([int(e.t[0]), "arrived (evil)" if evil else "arrived", pid, 0, 0])
        return pid

    def set_rule(self, key: str, value) -> bool:
        if key not in RULES or key not in {f.name for f in fields(self.cfg)}:
            return False
        cur = getattr(self.cfg, key)
        setattr(self.cfg, key, type(cur)(value) if not isinstance(cur, bool) else bool(value))
        if key == "lunch_break":                                   # shifts are precomputed: rebuild them
            self.env.shift_on = _shift_table(self.cfg)
        self.events.append([int(self.env.t[0]), f"rule {key}", 0, 0, float(value) if not isinstance(value, bool) else int(value)])
        return True

    # -------------------------------------------------------------- export
    def header(self) -> dict:
        from dataclasses import asdict
        conf = asdict(self.cfg)
        conf.update(day_steps=DAY, days_per_year=DAYS_PER_YEAR, seasons=SEASONS)
        return dict(kind="town", version=3, live=True, config=conf, start_step=int(self.env.t[0]), stride=1,
                    activities=ACTIVITIES, jobs=JOBS, foods=FOODS, slot_cols=SLOT_COLS, rules=RULES)

    def people_json(self) -> str:
        return json.dumps({str(k): v for k, v in self.people.items()})


def _shift_table(c: TownConfig):
    from .town import JOBS as J
    sh = np.zeros((len(J), DAY, DAYS_PER_YEAR), bool)
    for k, j in enumerate(J[1:], 1):
        s0, s1, b0, b1 = c.shifts[j]
        if not c.lunch_break:
            b0 = b1 = -1
        for h in range(DAY):
            on = s0 <= h < s1 and not b0 <= h < b1
            for d in range(DAYS_PER_YEAR):
                sh[k, h, d] = on and not (d == 3 and j in c.winter_off)
    return sh
