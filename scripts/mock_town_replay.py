#!/usr/bin/env python3
"""Write a plausible Town Life (v3) replay for developing the 3D viewer (viewer3d/town.html).

The replay follows docs/TOWN_REPLAY.md: one frame per hour, people in slots, the town's
state and an event list. It is a scripted toy, not the learned simulation: a dozen-odd
people sleep at home, walk to work along the roads, take a lunch break at the canteen,
walk back, go home in the evening; children go to school; couples form, babies are born,
the old die; seasons drive the crops, the fruit and the herd.

    python scripts/mock_town_replay.py                       # default folder, 3 years
    python scripts/mock_town_replay.py --out site_dir --years 6 --days-per-year 4

It writes <out>/replays/index.json (two stages and a small fake training curve) and the
two replay files next to it. Standard library only.
"""
import argparse
import heapq
import json
import math
import os
import random

DEFAULT_OUT = "/tmp/claude-0/-home-user-malecns-test/22f9c804-43d4-5a75-8578-13605c94b08c/scratchpad/town_site"

ACTS = ["idle", "walking", "working", "eating", "buying", "sleeping", "school",
        "socialising", "home", "detained", "nursing"]
A = {a: k for k, a in enumerate(ACTS)}
JOBS = ["none", "farm", "orchard", "dairy", "warehouse", "canteen"]
J = {j: k for k, j in enumerate(JOBS)}
FOODS = ["grain", "fruit", "dairy"]
SLOT_COLS = ["pid", "x", "y", "act", "age10", "hunger", "energy", "health", "money",
             "g", "f", "d", "job", "home", "partner", "flags"]
F_PREG, F_DET, F_POLICE, F_ASLEEP, F_THIEF, F_SHIFT, F_DOCKED, F_CHILD = 1, 2, 4, 8, 16, 32, 64, 128


def make_config(days_per_year, slots):
    return {
        "width": 32, "height": 24,
        "day_steps": 24, "days_per_year": days_per_year,
        "seasons": ["spring", "summer", "autumn", "winter"],
        "roads": [[0, 11, 31, 11], [0, 18, 31, 18], [9, 0, 9, 23], [22, 0, 22, 23]],
        "houses": [[1, 20, 2, 21], [5, 20, 6, 21], [1, 16, 2, 17], [5, 16, 6, 17], [1, 13, 2, 14],
                   [5, 13, 6, 14], [16, 20, 17, 21], [19, 20, 20, 21], [24, 20, 25, 21],
                   [28, 20, 29, 21], [28, 14, 29, 15]],
        "house_beds": 4,
        "places": {"farm": [11, 1, 20, 9], "orchard": [24, 6, 30, 9], "dairy": [24, 0, 30, 4],
                   "warehouse": [1, 5, 7, 9], "canteen": [16, 13, 20, 16], "market": [11, 13, 14, 16],
                   "school": [10, 20, 14, 22], "station": [24, 13, 26, 16]},
        "shifts": {"farm": [8, 17, 12, 13], "orchard": [8, 17, 12, 13], "dairy": [6, 15, 11, 12],
                   "warehouse": [8, 17, 12, 13], "canteen": [10, 19, 15, 16]},
        "school_hours": [8, 15],
        "shop_hours": [7, 20],
        "start_hour": 6,
        "slots": slots,
        "wage": 2,
    }


class Grid:
    """Cheapest walking routes on the grid: roads cost 1, lots 2, grass 4."""

    def __init__(self, cfg):
        self.w, self.h = cfg["width"], cfg["height"]
        self.cost = [[4.0] * self.h for _ in range(self.w)]
        for box in list(cfg["places"].values()) + cfg["houses"]:
            for x, y in cells(box):
                self.cost[x][y] = 2.0
        for box in cfg["roads"]:
            for x, y in cells(box):
                self.cost[x][y] = 1.0
        self.cache = {}

    def path(self, a, b):
        key = (a, b)
        if key in self.cache:
            return self.cache[key]
        dist, prev, pq = {a: 0.0}, {}, [(0.0, a)]
        while pq:
            d, c = heapq.heappop(pq)
            if c == b:
                break
            if d > dist[c]:
                continue
            x, y = c
            for n in ((x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)):
                if 0 <= n[0] < self.w and 0 <= n[1] < self.h:
                    nd = d + self.cost[n[0]][n[1]]
                    if nd < dist.get(n, 1e18):
                        dist[n], prev[n] = nd, c
                        heapq.heappush(pq, (nd, n))
        out, c = [b], b
        while c != a:
            c = prev[c]
            out.append(c)
        out.reverse()
        self.cache[key] = out
        return out


def cells(box):
    x0, y0, x1, y1 = box
    return [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y1 + 1)]


class Person:
    def __init__(self, pid, sex, age, born_step, mother=0, father=0, rng=None):
        self.pid, self.sex, self.age, self.born_step = pid, sex, age, born_step
        self.mother, self.father = mother, father
        self.traits = [round(rng.random(), 2) for _ in range(4)]
        self.hunger, self.energy, self.health = rng.randint(10, 40), rng.randint(60, 95), 100
        self.money = rng.randint(4, 30)
        self.food = [0, 0, 0]
        self.job, self.home, self.partner = 0, -1, 0
        self.police = False
        self.pregnant_until = -1
        self.detained_until = -1
        self.seen_stealing = -1
        self.slot, self.pos = -1, (0, 0)
        self.workcell = None
        self.lazy = rng.random() * 0.12
        self.oversleep_until = -1
        self.worked_hours = 0
        self.act, self.shift, self.docked = "home", False, False
        self.alive = True


class Town:
    def __init__(self, cfg, seed, skill):
        self.cfg, self.rng, self.skill = cfg, random.Random(seed), skill
        self.grid = Grid(cfg)
        self.people, self.slots, self.events, self.next_pid = {}, [None] * cfg["slots"], [], 1
        self.stock, self.treasury = [30, 20, 15], 260
        self.ypd = cfg["days_per_year"]
        self.year_steps = 24 * self.ypd
        self.places = cfg["places"]
        self.houses = cfg["houses"]
        self.found()

    # ------------------------------------------------------------ population
    def add(self, sex, age, t, mother=0, father=0, home=-1):
        p = Person(self.next_pid, sex, age, t - int(age * self.year_steps), mother, father, self.rng)
        self.next_pid += 1
        free = [k for k, q in enumerate(self.slots) if q is None]
        if not free:
            return None
        p.slot = free[0]
        self.slots[p.slot] = p
        self.people[p.pid] = p
        p.home = home
        if home >= 0:
            p.pos = self.bed(p)
        return p

    def residents(self, h):
        return [q for q in self.slots if q is not None and q.home == h]

    def bed(self, p):
        box = self.houses[p.home]
        beds = cells(box)
        mates = sorted(q.pid for q in self.residents(p.home))
        k = mates.index(p.pid) if p.pid in mates else 0
        return beds[k % len(beds)]

    def found(self):
        r = self.rng
        jobs = ["farm", "farm", "orchard", "dairy", "warehouse", "warehouse", "canteen", "canteen", "orchard", "dairy"]
        # couples with children, some singles, two elders
        plan = [((0, 34), (1, 36), [8, 4]), ((0, 29), (1, 31), [2]), ((0, 41), (1, 44), [11, 9]),
                ((0, 26), None, []), ((1, 27), None, []), ((1, 23), None, []), ((0, 22), None, []),
                ((0, 71), (1, 74), [])]
        h, jk = 0, 0
        for a, b, kids in plan:
            pa = self.add(a[0], a[1] + r.random(), 0, home=h)
            pb = self.add(b[0], b[1] + r.random(), 0, home=h) if b else None
            if pb:
                pa.partner, pb.partner = pb.pid, pa.pid
            for age in kids:
                self.add(r.randint(0, 1), age + r.random(), 0, mother=pa.pid, father=pb.pid if pb else 0, home=h)
            for q in (pa, pb):
                if q and 16 <= q.age < 65:
                    self.hire(q, jobs[jk % len(jobs)], quiet=True)
                    jk += 1
            h += 1
        for q in self.slots:
            if q is not None:
                q.pos = self.bed(q)
        police = [q for q in self.slots if q is not None and 20 < q.age < 50]
        police[3].police = True
        # one founding mother is already expecting, due in the first year
        mom = next(q for q in self.slots if q is not None and q.sex == 0 and q.partner and 20 < q.age < 32)
        mom.pregnant_until = int(0.4 * self.year_steps)
        self.ev(self.cfg["start_hour"], "expecting", mom.pid, mom.partner)

    def hire(self, p, job, quiet=False, t=0):
        p.job = J[job]
        box = self.places[job]
        cs = cells(box)
        # stand in the working part of the place (not the road-side edge)
        p.workcell = cs[(p.pid * 7 + 3) % len(cs)]
        if not quiet:
            self.ev(t, "hired", p.pid, 0, p.job)

    def ev(self, t, kind, i, j=0, amt=0):
        self.events.append([t, kind, i, j, amt])

    # ------------------------------------------------------------ routine
    def target(self, p, t):
        """Where p wants to be this hour and what p does there."""
        hour, day = t % 24, t // 24
        cfg = self.cfg
        home = self.bed(p) if p.home >= 0 else self.center("market")
        if p.detained_until > t:
            return self.center("station"), "detained"
        if p.age < 2:
            m = self.people.get(p.mother)
            if m is not None and m.alive:
                return m.pos, "nursing"
            return home, "sleeping" if hour >= 20 or hour < 7 else "home"
        if p.age < 12:
            s0, s1 = cfg["school_hours"]
            if hour >= 20 or hour < 7:
                return home, "sleeping"
            if s0 <= hour < s1:
                cs = cells(self.places["school"])
                return cs[(p.pid * 5) % len(cs)], "school"
            if s1 <= hour < 18:
                return self.spot(p, day, "park"), "socialising"
            return home, "home"
        job = JOBS[p.job]
        if job != "none":
            s0, s1, b0, b1 = cfg["shifts"][job]
            wake, bed_h = s0 - 2, 22 if s0 >= 8 else 21
        else:
            s0 = s1 = b0 = b1 = -1
            wake, bed_h = 7, 22
        if p.oversleep_until > t:
            return home, "sleeping"
        if hour >= bed_h or hour < wake:
            return home, "sleeping"
        if s0 <= hour < s1:
            if b0 <= hour < b1:
                return self.canteen_seat(p), "eating"
            if job == "canteen":
                return self.counter(p), "working"
            return p.workcell, "working"
        if job == "none":
            if 10 <= hour < 12:
                return self.spot(p, day, "park"), "socialising"
            if 12 <= hour < 13:
                return self.canteen_seat(p), "eating"
            if 13 <= hour < 16 and (p.pid + day) % 2 == 0:
                return self.stall(p), "buying"
            if 16 <= hour < 18:
                return self.spot(p, day, "square"), "socialising"
            return home, "home"
        if hour >= s1 and hour < 19 and (p.pid + day) % 3 != 0:
            if hour == s1 and (p.pid + day) % 2 == 0:
                return self.stall(p), "buying"
            return self.spot(p, day, "square" if day % 2 else "park"), "socialising"
        return home, "home"

    def center(self, name):
        x0, y0, x1, y1 = self.places[name]
        return ((x0 + x1) // 2, (y0 + y1) // 2)

    def canteen_seat(self, p):
        x0, y0, x1, y1 = self.places["canteen"]
        cs = [(x, y) for x in range(x0, x1 + 1) for y in range(y0, y0 + 2)]   # the terrace, south side
        return cs[(p.pid * 3) % len(cs)]

    def counter(self, p):
        x0, y0, x1, y1 = self.places["canteen"]
        return (x0 + 1 + p.pid % max(1, x1 - x0 - 1), y1 - 1)

    def stall(self, p):
        x0, y0, x1, y1 = self.places["market"]
        return (x0 + p.pid % (x1 - x0 + 1), y1 - 1)

    def spot(self, p, day, kind):
        if kind == "square":
            x0, y0, x1, y1 = self.places["market"]
            return ((x0 + x1) // 2 + p.pid % 2, y0 + (p.pid // 2) % 2)
        # the green between the school and the canteen, or the lawn by the west houses
        return (15 + p.pid % 2, 19) if day % 2 else (7 + p.pid % 2, 16)

    # ------------------------------------------------------------ one hour
    def step(self, t):
        r, cfg = self.rng, self.cfg
        hour, day = t % 24, t // 24
        season = (day % self.ypd) * 4 // self.ypd
        docked_now = set()
        for p in list(self.slots):
            if p is None:
                continue
            p.age += 1.0 / self.year_steps
            job = JOBS[p.job]
            # oversleeping costs pay
            if job != "none" and p.age >= 16:
                s0 = cfg["shifts"][job][0]
                if hour == s0 - 2 and r.random() < p.lazy * (1.6 - self.skill):
                    p.oversleep_until = t + r.randint(3, 4)
            tgt, act = self.target(p, t)
            if job != "none":
                s0, s1, b0, b1 = cfg["shifts"][job]
                if s0 <= hour < s1 and not (b0 <= hour < b1) and p.oversleep_until > t:
                    docked_now.add(p.pid)
            # move up to 24 cells an hour along the roads
            if p.pos != tgt and act != "nursing":
                path = self.grid.path(p.pos, tgt)
                k = min(len(path) - 1, 24)
                p.pos = path[k]
                if p.pos != tgt:
                    act = "walking"
            elif act == "nursing":
                p.pos = tgt
            p.act = act
            # needs
            if act == "sleeping":
                p.energy = min(100, p.energy + 9)
                p.hunger = min(100, p.hunger + 1)
            else:
                p.energy = max(0, p.energy - (5 if act in ("working", "walking") else 3))
                p.hunger = min(100, p.hunger + 3)
            p.shift = act == "working"
            if act == "working":
                p.worked_hours += 1
        # canteen open when a canteen worker is at the counter
        canteen_open = any(p is not None and JOBS[p.job] == "canteen" and p.act == "working" for p in self.slots)
        self.canteen_open = 1 if canteen_open else 0
        price = [2 + (self.stock[0] < 10), 3 + (self.stock[1] < 8), 2 + (self.stock[2] < 8)]
        self.price = price
        for p in self.slots:
            if p is None:
                continue
            if p.act == "eating":
                staff = JOBS[p.job] == "canteen"
                if (self.canteen_open or staff) and p.hunger > 25:
                    k = min(range(3), key=lambda f: (self.stock[f] == 0, p.food[f], r.random()))
                    if self.stock[k] > 0 and p.money >= price[k]:
                        self.stock[k] -= 1
                        p.money -= price[k]
                        self.treasury += price[k]
                        self.ev(t, "bought " + FOODS[k], p.pid, 0, price[k])
                        self.ev(t, "ate " + FOODS[k], p.pid)
                        p.hunger = max(0, p.hunger - 45)
                elif not self.canteen_open and p.hunger > 60 and sum(p.food):
                    k = max(range(3), key=lambda f: p.food[f])
                    p.food[k] -= 1
                    p.hunger = max(0, p.hunger - 40)
                    self.ev(t, "ate " + FOODS[k], p.pid)
            elif p.act == "buying" and p.money > 4 and r.random() < 0.6:
                k = r.randrange(3)
                if self.stock[k] > 0:
                    self.stock[k] -= 1
                    p.money -= price[k]
                    self.treasury += price[k]
                    p.food[k] += 1
                    self.ev(t, "bought " + FOODS[k], p.pid, 0, price[k])
            elif p.act == "nursing":
                p.hunger = max(0, p.hunger - 8)
            elif p.act == "home" and (5 <= hour <= 9 or 18 <= hour <= 21) and p.hunger > 40:
                have = [q for q in self.residents(p.home) if sum(q.food)]
                if have:
                    q = have[0]
                    k = max(range(3), key=lambda f: q.food[f])
                    q.food[k] -= 1
                p.hunger = max(0, p.hunger - 35)
                self.ev(t, "family meal", p.pid)
            elif p.act == "school" and hour == 12:
                p.hunger = max(0, p.hunger - 40)
                self.ev(t, "ate " + FOODS[t % 3], p.pid)
            if p.act in ("socialising", "eating"):
                friends = [q for q in self.slots if q is not None and q is not p and q.act == p.act
                           and abs(q.pos[0] - p.pos[0]) + abs(q.pos[1] - p.pos[1]) <= 1]
                if friends and r.random() < 0.3:
                    q = r.choice(friends)
                    self.ev(t, "talked", p.pid, q.pid)
                    self.maybe_partner(p, q, t)
            p.hunger = min(100, p.hunger)
        # shift ends: pay; missed work: docked
        for p in self.slots:
            if p is None or p.job == 0:
                continue
            s0, s1, b0, b1 = cfg["shifts"][JOBS[p.job]]
            if p.pid in docked_now:
                lost = 2
                p.money = max(0, p.money - lost)
                p.docked = True
                self.ev(t, "docked", p.pid, 0, lost)
            else:
                p.docked = False
            if hour == s1 - 1 and p.worked_hours:
                pay = cfg["wage"] * p.worked_hours
                pay = min(pay, self.treasury)
                p.money += pay
                self.treasury -= pay
                self.ev(t, "earned", p.pid, 0, pay)
                p.worked_hours = 0
        # production
        workers = {j: sum(1 for p in self.slots if p is not None and p.act == "working" and JOBS[p.job] == j) for j in JOBS}
        if season == 2:
            self.stock[0] += workers["farm"] * 2
        elif season in (0, 1):
            self.stock[0] += workers["farm"] // 2
        if season in (1, 2):
            self.stock[1] += workers["orchard"] * (2 if season == 2 else 1)
        self.stock[2] += workers["dairy"] * (1 if season != 3 else 0) + (hour == 7)
        self.treasury += workers["warehouse"] * 4 + sum(workers.values()) * 2
        self.stock = [min(200, s) for s in self.stock]
        self.lifecycle(t)
        self.crime(t)

    def maybe_partner(self, p, q, t, force=False):
        r = self.rng
        if p.partner or q.partner or p.sex == q.sex or p.age < 18 or q.age < 18 or abs(p.age - q.age) > 12:
            return
        if force or r.random() < 0.15 + 0.2 * self.skill:
            p.partner, q.partner = q.pid, p.pid
            self.ev(t, "courted", p.pid, q.pid)
            self.ev(t, "partnered", p.pid, q.pid)
            # the one with the fuller house moves
            a, b = (p, q) if len(self.residents(p.home)) >= len(self.residents(q.home)) else (q, p)
            if len(self.residents(b.home)) < self.cfg["house_beds"]:
                a.home = b.home
                self.ev(t, "moved in", a.pid, 0, b.home)

    def lifecycle(self, t):
        r = self.rng
        hour = t % 24
        # a new couple forms every year or so among the singles who meet in the evening
        if hour == 18 and (t // 24) % max(1, self.ypd) == 1:
            singles = [p for p in self.slots if p is not None and not p.partner and 18 <= p.age < 50]
            for p in singles:
                q = next((q for q in singles if q.sex != p.sex and abs(q.age - p.age) <= 12), None)
                if q is not None and r.random() < 0.3 + 0.5 * self.skill:
                    self.maybe_partner(p, q, t, force=True)
                    break
        for p in list(self.slots):
            if p is None:
                continue
            # pregnancy: about three quarters of a year
            if p.sex == 0 and p.partner and p.pregnant_until < 0 and 19 < p.age < 42 and hour == 22:
                kids = sum(1 for q in self.people.values() if q.mother == p.pid and q.alive)
                if r.random() < (0.9 if kids < 2 else 0.15) / self.ypd:
                    p.pregnant_until = t + int(0.75 * self.year_steps)
                    self.ev(t, "expecting", p.pid, p.partner)
            if 0 <= p.pregnant_until <= t:
                p.pregnant_until = -1
                room = len(self.residents(p.home)) < self.cfg["house_beds"]
                if room or True:
                    c = self.add(r.randint(0, 1), 0.0, t, mother=p.pid, father=p.partner, home=p.home)
                    if c is not None:
                        c.pos = p.pos
                        self.ev(t, "born", c.pid, p.pid)
            # old age
            if p.age > 70 and hour == 3 and r.random() < 0.05 * (p.age - 69) / self.ypd:
                self.die(p, t, "died (old age)")
            elif p.hunger >= 100 and r.random() < 0.02:
                self.die(p, t, "died (starved)")
            # coming of age
            if p.job == 0 and 16 <= p.age < 60 and hour == 9:
                counts = {j: sum(1 for q in self.slots if q is not None and JOBS[q.job] == j) for j in JOBS[1:]}
                job = min(counts, key=counts.get)
                self.hire(p, job, t=t)
            if p.job and p.age >= 66 and hour == 9:
                p.job, p.workcell = 0, None
                self.ev(t, "resigned", p.pid)
            # leave home at 18 if the house is full and a house is empty
            if 18 <= p.age < 19 and p.partner == 0 and hour == 10 and len(self.residents(p.home)) >= 4:
                empty = [h for h in range(len(self.houses)) if not self.residents(h)]
                if empty:
                    p.home = empty[0]
                    self.ev(t, "moved in", p.pid, 0, p.home)

    def die(self, p, t, kind):
        self.ev(t, kind, p.pid, 0, round(p.age))
        p.alive = False
        self.slots[p.slot] = None
        if p.partner and p.partner in self.people:
            self.people[p.partner].partner = 0

    def crime(self, t):
        r = self.rng
        hour = t % 24
        if hour not in (14, 16, 17):
            return
        market = self.places["market"]
        at_market = [p for p in self.slots if p is not None and p.age >= 14 and inside(p.pos, market)]
        if len(at_market) >= 2 and r.random() < 0.25 * (1.2 - self.skill):
            thief = max(at_market, key=lambda p: p.traits[1] + r.random() * 0.3)
            victim = r.choice([p for p in at_market if p is not thief])
            amt = min(victim.money, r.randint(2, 6))
            if amt > 0:
                victim.money -= amt
                thief.money += amt
                self.ev(t, "theft", thief.pid, victim.pid, amt)
                thief.seen_stealing = t + 24
                police = [p for p in self.slots if p is not None and p.police and p.detained_until < t]
                if police and r.random() < 0.7:
                    cop = police[0]
                    back = min(thief.money, amt)
                    thief.money -= back
                    victim.money += back
                    thief.detained_until = t + 5
                    self.ev(t + 1, "detained", cop.pid, thief.pid, back)
                elif r.random() < 0.5:
                    thief.detained_until = t + 5
                    self.ev(t + 1, "citizens' arrest", victim.pid, thief.pid, 0)

    # ------------------------------------------------------------ recording
    def frame(self, t):
        s = []
        slot_of = {p.pid: k for k, p in enumerate(self.slots) if p is not None}
        for p in self.slots:
            if p is None:
                s.append(0)
                continue
            flags = 0
            if p.pregnant_until > t:
                flags |= F_PREG
            if p.detained_until > t:
                flags |= F_DET
            if p.police:
                flags |= F_POLICE
            if p.act == "sleeping":
                flags |= F_ASLEEP
            if p.seen_stealing > t:
                flags |= F_THIEF
            if getattr(p, "shift", False):
                flags |= F_SHIFT
            if getattr(p, "docked", False):
                flags |= F_DOCKED
            if p.age < 12:
                flags |= F_CHILD
            s.append([p.pid, p.pos[0], p.pos[1], A[p.act], int(p.age * 10), int(p.hunger), int(p.energy),
                      int(p.health), int(p.money), p.food[0], p.food[1], p.food[2], p.job, p.home,
                      slot_of.get(p.partner, -1), flags])
        day = t // 24
        hour = t % 24
        season = (day % self.ypd) * 4 // self.ypd
        frac = hour / 24.0
        crops = [1 + round(3 * frac), 5 + round(3 * frac), 9 if hour < 14 else 2, 0][season]
        fruit = [0, 2 + round(4 * frac), max(3, 9 - max(0, hour - 8) // 2), 0][season]
        out = hour < 7 or hour >= 20
        herd = [6, 8, 7, 2][season] - (4 if out and season != 3 else 0)
        shop = 1 if self.cfg["shop_hours"][0] <= hour < self.cfg["shop_hours"][1] else 0
        w = {"stock": list(self.stock), "treasury": int(self.treasury), "canteen_open": self.canteen_open,
             "shop_open": shop,
             "crops": int(crops), "fruit": int(fruit), "herd": int(max(0, herd)), "price": list(self.price)}
        return {"t": t, "s": s, "w": w}

    def people_json(self):
        return {str(p.pid): {"sex": p.sex, "mother": p.mother, "father": p.father, "traits": p.traits,
                             "born_step": p.born_step} for p in self.people.values()}


def inside(pos, box):
    return box[0] <= pos[0] <= box[2] and box[1] <= pos[1] <= box[3]


def record(cfg, years, seed, skill, iteration, samples):
    town = Town(cfg, seed, skill)
    start = cfg["start_hour"]
    steps = years * town.year_steps
    frames = []
    for t in range(start, start + steps):
        town.step(t)
        frames.append(town.frame(t))
    ev = sorted(town.events, key=lambda e: e[0])
    count = lambda pred: sum(1 for e in ev if pred(e[1]))
    summary = {
        "years": round(steps / town.year_steps, 2),
        "births": count(lambda k: k == "born"),
        "deaths": count(lambda k: k.startswith("died")),
        "starved": count(lambda k: k == "died (starved)"),
        "malnourished": count(lambda k: k == "died (malnutrition)"),
        "hired": count(lambda k: k == "hired"),
        "meals": count(lambda k: k.startswith("bought ")),
        "earned": sum(e[4] for e in ev if e[1] == "earned"),
        "docked": count(lambda k: k == "docked"),
        "thefts": count(lambda k: k == "theft"),
        "detentions": count(lambda k: k in ("detained", "citizens' arrest")),
        "partnerships": count(lambda k: k == "partnered"),
        "population": sum(1 for p in town.slots if p is not None),
    }
    return {
        "kind": "town", "version": 3,
        "meta": {"iteration": iteration, "samples": samples},
        "config": cfg, "start_step": start, "stride": 1,
        "activities": ACTS, "jobs": JOBS, "foods": FOODS,
        "people": town.people_json(), "slot_cols": SLOT_COLS,
        "frames": frames, "events": ev, "summary": summary,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=DEFAULT_OUT, help="site folder; replays go to <out>/replays/")
    ap.add_argument("--years", type=int, default=3)
    ap.add_argument("--days-per-year", type=int, default=4, help="4 = one day per season (TOWN_REPLAY.md)")
    ap.add_argument("--slots", type=int, default=24)
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    cfg = make_config(args.days_per_year, args.slots)
    rdir = os.path.join(args.out, "replays")
    os.makedirs(rdir, exist_ok=True)
    stages = []
    for it, skill, samples in ((0, 0.2, 0), (1000, 1.0, 98_304_000)):
        rep = record(cfg, args.years, args.seed + it, skill, it, samples)
        name = f"town_it{it:05d}.json"
        with open(os.path.join(rdir, name), "w") as f:
            json.dump(rep, f, separators=(",", ":"))
        stages.append({"it": it, "samples": samples, "file": name, "summary": rep["summary"]})
        print(f"{name}: {len(rep['frames'])} frames, {len(rep['events'])} events, {rep['summary']}")
    rng = random.Random(args.seed)
    curve = []
    for k in range(13):
        it = k * 100
        u = 1 - math.exp(-it / 350)
        curve.append({"it": it, "years": round(20 + 40 * u + rng.uniform(-3, 3), 2),
                      "population": round(6 + 10 * u + rng.uniform(-1, 1), 2),
                      "births": round(4 + 12 * u + rng.uniform(-1, 1), 2),
                      "deaths": round(14 - 4 * u + rng.uniform(-1, 1), 2),
                      "starved": round(max(0, 8 * (1 - u) + rng.uniform(-0.5, 0.5)), 2),
                      "meals": round(300 + 2400 * u + rng.uniform(-60, 60), 1),
                      "earned": round(2000 + 9000 * u + rng.uniform(-200, 200), 1),
                      "docked": round(max(0, 600 * (1 - u) + 40 + rng.uniform(-20, 20)), 1),
                      "thefts": round(max(0, 12 * (1 - u) + 1 + rng.uniform(-1, 1)), 2),
                      "detentions": round(max(0, 3 + 2 * u + rng.uniform(-1, 1)), 2),
                      "partnerships": round(2 + 9 * u + rng.uniform(-1, 1), 2),
                      "happiness": round(-200 + 2600 * u + rng.uniform(-80, 80), 1)})
    index = {"stages": stages, "curve": curve,
             "note": "Mock data from scripts/mock_town_replay.py, for developing the viewer."}
    with open(os.path.join(rdir, "index.json"), "w") as f:
        json.dump(index, f)
    print("wrote", rdir)


if __name__ == "__main__":
    main()
