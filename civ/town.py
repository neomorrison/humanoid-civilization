"""Town Life (v3): whole lives lived one hour at a time.

One step is an hour and a day has 24 of them. Each day is a season (spring,
summer, autumn, winter), so a year is four days: bodies and daily routines run
on the clock, while lives (childhood, pregnancy, old age) run on the calendar.

People sleep, work, eat and raise families:

* Energy drains while awake, faster at night (circadian sleep pressure), and
  returns while asleep, best in your own bed at night. The exhausted collapse
  where they stand.
* Anyone can take a job at the farm, orchard, dairy, warehouse or canteen. The
  contract is simple: every hour you are at work during your shift is paid; every
  hour you are missing outside the lunch break is docked one hour's pay. Mothers
  of infants and the sick are excused.
* The farm, orchard and dairy workers' produce goes into the businesses' stock,
  which the canteen sells while canteen staff are on shift; the warehouse earns
  from the outside world. Wages come out of what the businesses earn.
* Everyone has a bed in one of the houses. Couples move in together; children
  are born into their mother's house if it has a free bed. Children can go to
  school, which raises the wage they will earn as adults.
* Everything personal stays voluntary, as before: gifts, private sales at your
  own price, theft, gossip, shunning, courtship, feeding your children, police
  volunteering, detentions by police or by two witnesses.

The reward is happiness (see DESIGN.md). Vectorised over M worlds with S person
slots each.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

DAY = 24
DAYS_PER_YEAR = 4
YEAR = DAY * DAYS_PER_YEAR
SEASONS = ["spring", "summer", "autumn", "winter"]
F = 0
GRAIN, FRUIT, DAIRY = 0, 1, 2
FOODS = ["grain", "fruit", "dairy"]
JOBS = ["none", "farm", "orchard", "dairy", "warehouse", "canteen"]
ACTIVITIES = ["idle", "walking", "working", "eating", "buying", "sleeping", "school", "socialising", "home",
              "detained", "nursing"]
ACT_ID = {n: i for i, n in enumerate(ACTIVITIES)}
K = 4                               # neighbours you can deal with
# base actions; 4.. are intentions: chosen once, carried out until done (walking is the body's job)
BASE = ["idle", "eat", "price+", "price-", "work", "lunch", "groceries", "go_home", "sleep", "feed_child",
        "go_family", "school", "go_market", "enlist"] + [f"apply_{j}" for j in JOBS[1:]]
(IDLE, EAT, PRICE_UP, PRICE_DOWN, WORK, LUNCH, GROCERIES, GO_HOME, SLEEP, FEED_CHILD, GO_FAMILY, SCHOOL,
 GO_MARKET, ENLIST) = range(14)
APPLY0 = 14
N_BASE = len(BASE)
N_PER = 8
(GIVE_FOOD, GIVE_COIN, BUY, TAKE, TALK, COURT, SHUN, DETAIN) = range(N_PER)
ACT_NAMES = BASE + [f"{a}>{k}" for k in range(K) for a in
                    ["give_food", "give_coin", "buy", "take", "talk", "court", "shun", "detain"]]


@dataclass
class TownConfig:
    width: int = 32
    height: int = 24
    slots: int = 24
    founders: int = 12
    episode_years: float = 60.0
    start_hour: int = 6
    roads: tuple = ((0, 11, 31, 11), (0, 18, 31, 18), (9, 0, 9, 23), (22, 0, 22, 23))
    houses: tuple = ((1, 20, 2, 21), (5, 20, 6, 21), (1, 16, 2, 17), (5, 16, 6, 17), (1, 13, 2, 14), (5, 13, 6, 14),
                     (16, 20, 17, 21), (19, 20, 20, 21), (24, 20, 25, 21), (28, 20, 29, 21), (28, 14, 29, 15))
    house_beds: int = 4
    positions: tuple = (0, 4, 4, 4, 3, 3)   # jobs each business offers (none, farm, orchard, dairy, warehouse, canteen)
    job_tenure: int = 96               # hours before someone hired can change jobs again (a year)
    places: dict = field(default_factory=lambda: {
        "farm": (11, 1, 20, 9), "orchard": (24, 6, 30, 9), "dairy": (24, 0, 30, 4), "warehouse": (1, 5, 7, 9),
        "canteen": (16, 13, 20, 16), "market": (11, 13, 14, 16), "school": (10, 20, 14, 22), "station": (24, 13, 26, 16)})
    # (start, end, break start, break end); hours in [start, end) outside the break are paid
    shifts: dict = field(default_factory=lambda: {
        "farm": (8, 17, 12, 13), "orchard": (8, 17, 12, 13), "dairy": (6, 15, 11, 12),
        "warehouse": (8, 17, 12, 13), "canteen": (10, 19, 15, 16)})
    winter_off: tuple = ("farm", "orchard")      # nothing to sow or pick in winter
    school_hours: tuple = (8, 15)
    shop_hours: tuple = (7, 20)        # the grocery shop in the market hall sells the businesses' stock
    hot_meal: float = 1.3             # a canteen meal satisfies more than cold food
    w_hot_meal: float = 0.03          # once: enjoying a hot lunch
    speed: int = 32                    # grid cells walked per hour (any errand in town takes an hour or two)
    # economy
    wage: float = 0.7                  # coins per hour at work, before skill and schooling
    dock: float = 1.0                  # coins lost per hour missing from a shift, per unit of wage
    skill_gain: float = 0.004          # per hour worked in that job
    skill_max: float = 1.5
    edu_gain: float = 0.004            # per hour at school
    output: tuple = (1.4, 1.4, 0.9)    # grain, fruit, dairy per worker-hour before season, skill and energy
    season_yield: tuple = ((0.3, 0.8, 1.6, 0.0), (0.2, 0.8, 1.6, 0.0), (1.0, 1.1, 0.9, 0.6))
    export: float = 1.2                # coins the outside world pays per warehouse worker-hour
    stock_spoil: float = 0.003         # per hour
    price_base: float = 2.0
    price_target: float = 20.0         # the canteen charges more when stock runs below this
    canteen_serve: int = 8             # customers one canteen worker serves per hour
    treasury0: float = 200.0
    stock0: tuple = (20.0, 20.0, 20.0)
    max_food: int = 6
    spoil_hours: int = 72
    # bodies
    hunger_awake: float = 0.035
    hunger_asleep: float = 0.015
    pregnant_hunger: float = 1.3
    food_relief: float = 0.45
    satiety_floor: float = 0.1
    nutrient_decay: float = 1.0 / 72.0
    nutrient_gain: float = 0.5
    starve_damage: float = 0.01
    deficiency_damage: float = 0.004
    heal_rate: float = 0.01
    energy_awake: float = 1.0 / 20.0
    night_drain: float = 1.4           # circadian sleep pressure
    work_drain: float = 1.15
    child_drain: float = 1.2
    sleep_gain: float = 0.09           # per hour asleep in your own bed (a night restores a day)
    night_sleep: float = 1.2           # sleep restores more at night
    day_sleep: float = 0.7
    rough_sleep: float = 0.55          # sleeping away from your bed
    adult_age: float = 16.0
    child_age: float = 12.0
    school_ages: tuple = (6.0, 16.0)
    old_age: float = 60.0
    intent_patience: int = 6
    # social
    reach: int = 2                     # cells within which you can deal with someone
    witness_radius: int = 5
    memory_hours: int = 96
    detain_hours: int = 24
    gossip_rate: float = 0.3
    court_window: int = 12
    attract_threshold: float = 0.45
    conceive_prob: float = 0.01        # per hour that fed, fertile partners spend at home together
    gestation: int = 72
    nursing_age: float = 2.0
    nurse_relief: float = 0.03
    nurse_nutrient: float = 0.015
    nurse_cost: float = 0.008
    maternity_age: float = 1.0         # mothers of infants younger than this are excused from work
    fertile_ages: tuple = (18.0, 45.0)
    partner_age: float = 18.0
    mutation: float = 0.08
    # happiness (per hour unless noted)
    w_alive: float = 0.02
    w_hunger: float = 0.03
    w_starving: float = 0.06
    w_health: float = 0.03
    w_craving: float = 0.02
    crave_level: float = 0.5
    w_tired: float = 0.05
    w_robbed: float = 0.5              # once
    w_detained: float = 0.02
    w_friends: float = 0.004
    w_partner: float = 0.01
    w_together: float = 0.01
    w_family_meal: float = 0.1         # once per meal eaten at home with family there
    w_children: float = 0.02
    w_child_distress: float = 0.04
    child_grief: float = 3.0
    w_esteem: float = 0.01
    w_purpose: float = 0.01            # an hour of work or school
    w_security: float = 0.002
    w_birth: float = 3.0               # once
    w_grief: float = 4.0               # once
    w_death: float = 8.0               # once


def _in(p, box):
    x0, y0, x1, y1 = box
    return (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)


def _in_boxes(p, boxes):
    """p (..., 2), boxes (..., 4) broadcastable -> bool (...)."""
    return (p[..., 0] >= boxes[..., 0]) & (p[..., 0] <= boxes[..., 2]) & (p[..., 1] >= boxes[..., 1]) & \
        (p[..., 1] <= boxes[..., 3])


class Town:
    def __init__(self, n_worlds: int, cfg: TownConfig | None = None, seed: int = 0, record: bool = False):
        self.cfg = c = cfg or TownConfig()
        self.M, self.S = n_worlds, c.slots
        self.N = self.M * self.S
        self.rng = np.random.default_rng(seed)
        self.record = record
        self.H = len(c.houses)
        self.hbox = np.array(c.houses)
        self.job_box = np.array([[0, 0, -1, -1]] + [c.places[j] for j in JOBS[1:]])   # "none" matches nowhere
        sh = np.zeros((len(JOBS), DAY, DAYS_PER_YEAR), bool)
        for k, j in enumerate(JOBS[1:], 1):
            s0, s1, b0, b1 = c.shifts[j]
            for h in range(DAY):
                on = s0 <= h < s1 and not b0 <= h < b1
                for d in range(DAYS_PER_YEAR):
                    sh[k, h, d] = on and not (d == 3 and j in c.winter_off)
        self.shift_on = sh                             # [job, hour, season]
        self.act_dim = N_BASE + N_PER * K
        self.max_steps = int(c.episode_years * YEAR)
        self.stats = []
        self.next_pid = 1
        self._alloc()
        self._reset(np.arange(self.M))
        o, co, _ = self.observe()
        self.obs_dim, self.cobs_dim = o.shape[1], co.shape[1]

    # ------------------------------------------------------------------ state
    def _alloc(self):
        M, S = self.M, self.S
        z = lambda *s, dt=float: np.zeros((M,) + s, dt)
        self.alive = z(S, dt=bool)
        self.sex = z(S, dt=int)
        self.age = z(S)
        self.pos = z(S, 2, dt=int)
        self.hunger = z(S)
        self.energy = z(S)
        self.health = z(S)
        self.nutr = z(S, 3)
        self.money = z(S)
        self.food = z(S, 3, dt=int)
        self.price = z(S)
        self.job = z(S, dt=int)
        self.skill = z(S, len(JOBS))
        self.edu = z(S)
        self.pregnant = z(S, dt=int)
        self.partner = np.full((M, S), -1)
        self.home = np.full((M, S), -1)
        self.asleep = z(S, dt=bool)
        self.intent = np.full((M, S), -1)
        self.intent_t = z(S, dt=int)
        self.target = z(S, 2, dt=int)
        self.enlisted = z(S, dt=bool)
        self.detained = z(S, dt=int)
        self.traits = z(S, 4)                          # empathy, greed, boldness, charm
        self.pid = z(S, dt=int)
        self.mother = z(S, dt=int)
        self.father = z(S, dt=int)
        self.opinion = z(S, S)
        self.saw_theft = z(S, S, dt=int)
        self.robbed_by = z(S, S, dt=int)
        self.court = z(S, S, dt=int)
        self.shun = z(S, S, dt=bool)
        self.shift_pay = z(S)
        self.hired_t = np.full((M, S), 10_000)
        self.docked_now = z(S, dt=bool)
        self.act = z(S, dt=int)
        self.stock = z(3)
        self.treasury = z()
        self.t = z(dt=int)
        self.ep = None

    def _new_person(self, m, s, sex, age, traits, pos, mother=0, father=0):
        c = self.cfg
        adult = age >= c.adult_age
        self.alive[m, s] = True
        self.sex[m, s] = sex
        self.age[m, s] = age
        self.pos[m, s] = pos
        self.target[m, s] = pos
        self.hunger[m, s] = 0.3
        self.energy[m, s] = 0.8
        self.health[m, s] = 1.0
        self.nutr[m, s] = 0.6
        self.money[m, s] = 10.0 if adult else 0.0
        self.food[m, s] = 0
        self.price[m, s] = c.price_base
        self.job[m, s] = 0
        self.skill[m, s] = 0
        self.edu[m, s] = 0
        self.pregnant[m, s] = 0
        self.partner[m, s] = -1
        self.home[m, s] = -1
        self.asleep[m, s] = False
        self.intent[m, s] = -1
        self.intent_t[m, s] = 0
        self.enlisted[m, s] = False
        self.detained[m, s] = 0
        self.shift_pay[m, s] = 0
        self.hired_t[m, s] = 10_000
        self.traits[m, s] = np.clip(traits, 0, 1)
        self.pid[m, s] = self.next_pid
        self.next_pid += 1
        self.mother[m, s], self.father[m, s] = mother, father
        for arr in (self.opinion, self.saw_theft, self.robbed_by, self.court):
            arr[m, s, :] = 0
            arr[m, :, s] = 0
        self.shun[m, s, :] = False
        self.shun[m, :, s] = False

    def _reset(self, ws):
        c, r = self.cfg, self.rng
        for m in ws:
            self.alive[m] = False
            self.t[m] = c.start_hour
            self.stock[m] = c.stock0
            self.treasury[m] = c.treasury0
            houses = r.permutation(self.H)
            for s in range(c.founders):
                h = int(houses[s % self.H])
                x0, y0, x1, y1 = c.houses[h]
                self._new_person(m, s, s % 2, r.uniform(18, 35), r.random(4), [r.integers(x0, x1 + 1), r.integers(y0, y1 + 1)])
                self.home[m, s] = h
                self.nutr[m, s] = r.uniform(0.2, 0.9, 3)
                self.food[m, s] = r.integers(0, 3, 3)
                self.energy[m, s] = r.uniform(0.6, 1.0)
        if self.ep is None:
            self.ep = {k: np.zeros(self.M) for k in [
                "births", "deaths", "starved", "malnourished", "old_age", "work_hours", "docked_hours", "school_hours",
                "meals_canteen", "groceries", "family_meals", "hires", "wages", "exports", "produced", "trades",
                "gifts_food", "gifts_coin", "thefts", "detentions", "citizen_arrests", "partnerships", "breakups",
                "talks", "shuns", "person_steps", "happiness", "sleep_home", "sleep_rough", "night_awake"]}
        for k in self.ep:
            self.ep[k][ws] = 0

    # --------------------------------------------------------------- helpers
    def clock(self):
        hour = self.t % DAY
        season = (self.t // DAY) % DAYS_PER_YEAR
        return hour, season

    def related(self, m, i, j):
        pi, pj = self.pid[m, i], self.pid[m, j]
        mi, fi, mj, fj = self.mother[m, i], self.father[m, i], self.mother[m, j], self.father[m, j]
        return (mi == pj or fi == pj or mj == pi or fj == pi or (mi > 0 and mi == mj) or (fi > 0 and fi == fj))

    def _parent_slots(self):
        out = []
        for who in (self.mother, self.father):
            eq = (self.pid[:, None, :] == who[:, :, None]) & self.alive[:, None, :] & (who[:, :, None] > 0)
            out.append(np.where(eq.any(-1), eq.argmax(-1), -1))
        return out

    def _kids_matrix(self):
        """kid[m, i, j]: j is i's living child under adult age."""
        c = self.cfg
        return ((self.mother[:, None, :] == self.pid[:, :, None]) | (self.father[:, None, :] == self.pid[:, :, None])) \
            & self.alive[:, None, :] & (self.age[:, None, :] < c.adult_age) & (self.pid[:, :, None] > 0)

    def at_home(self):
        hb = self.hbox[np.clip(self.home, 0, self.H - 1)]
        return self.alive & (self.home >= 0) & _in_boxes(self.pos, hb)

    def at_work(self):
        return self.alive & (self.job > 0) & _in_boxes(self.pos, self.job_box[self.job])

    def residents(self):
        """Living people per house (M, H)."""
        out = np.zeros((self.M, self.H), int)
        m, s = np.nonzero(self.alive & (self.home >= 0))
        np.add.at(out, (m, self.home[m, s]), 1)
        return out

    def canteen_open(self):
        hour, season = self.clock()
        k = JOBS.index("canteen")
        on = self.shift_on[k, hour, season]
        staff = (self.at_work() & (self.job == k) & ~self.asleep).sum(1)
        return on & (staff > 0), staff

    def canteen_price(self):
        c = self.cfg
        mult = np.clip(c.price_target / (self.stock + 1.0), 0.5, 3.0)
        return np.round(c.price_base * mult * 2) / 2

    def attractiveness(self, m=None):
        sl = slice(None) if m is None else m
        health = self.health[sl][..., None, :]
        prov = np.tanh((self.money[sl] / 20.0 + (self.job[sl] > 0)) / 2.0)[..., None, :]
        op = (self.opinion[sl] + 1) / 2
        charm = self.traits[sl][..., None, :, 3]
        age = self.age[sl]
        agec = np.clip(1 - np.abs(age[..., :, None] - age[..., None, :]) / 20.0, 0, 1)
        return 0.25 * health + 0.2 * prov + 0.25 * op + 0.15 * charm + 0.15 * agec

    def _neighbours(self):
        d = np.abs(self.pos[:, :, None] - self.pos[:, None, :]).max(-1).astype(float)
        d[:, np.arange(self.S), np.arange(self.S)] = 1e9
        d = np.where(self.alive[:, None, :] & ~self.asleep[:, None, :], d, 1e9)
        idx = np.argsort(d, axis=2, kind="stable")[:, :, :K]
        dd = np.take_along_axis(d, idx, 2)
        return np.where(dd < 1e8, idx, -1), dd

    def _rel_box(self, boxes):
        """Offset to the nearest cell of a box per person; boxes (M, S, 4) or (4,)."""
        b = np.broadcast_to(np.asarray(boxes), self.pos.shape[:-1] + (4,))
        tx = np.clip(self.pos[..., 0], b[..., 0], b[..., 2])
        ty = np.clip(self.pos[..., 1], b[..., 1], b[..., 3])
        return np.stack([tx - self.pos[..., 0], ty - self.pos[..., 1]], -1)

    # ----------------------------------------------------------- observation
    def observe(self):
        c, M, S = self.cfg, self.M, self.S
        Wd, Hd = c.width, c.height
        nb, nd = self._neighbours()
        self._nb, self._nd = nb, nd
        hour, season = self.clock()
        A = self.attractiveness()
        ar = np.arange(M)[:, None]
        si = np.arange(S)[None, :]
        reg = (self.opinion * self.alive[:, :, None]).sum(1) / np.maximum(self.alive.sum(1, keepdims=True) - 1, 1)
        kidm = self._kids_matrix()
        nkids = kidm.sum(-1)
        hungriest = np.where(nkids > 0, np.where(kidm, self.hunger[:, None, :], -1).argmax(-1), -1)
        self._hungriest = hungriest
        hk = np.clip(hungriest, 0, S - 1)
        youngest_age = np.where(nkids > 0, np.where(kidm, self.age[:, None, :], 99).min(-1), 0)
        res = self.residents()
        free_beds = np.where(self.home >= 0, c.house_beds - res[ar, np.clip(self.home, 0, self.H - 1)], 0)
        on_shift = self.shift_on[self.job, hour[:, None], season[:, None]]
        s0 = np.array([0] + [c.shifts[j][0] for j in JOBS[1:]])[self.job]
        to_start = ((s0 - hour[:, None]) % DAY) * (self.job > 0) / DAY
        to_break_end = ((np.array([0] + [c.shifts[j][3] for j in JOBS[1:]])[self.job] - hour[:, None]) % DAY) * (self.job > 0) / DAY
        workday = self.shift_on.any(1)[self.job, season[:, None]]
        open_, _ = self.canteen_open()
        prices = self.canteen_price()
        staff_n = np.stack([((self.job == k) & self.alive).sum(1) for k in range(1, len(JOBS))], -1)
        vacancies = np.clip(np.array(c.positions[1:]) - staff_n, 0, None) / 4.0
        school_now = (hour >= c.school_hours[0]) & (hour < c.school_hours[1]) & (season < 3)
        at_home, at_work = self.at_home(), self.at_work()
        places = np.stack([_in(self.pos, c.places[p]) for p in ("canteen", "market", "school", "station")], -1)
        th = 2 * np.pi * hour / DAY
        timef = np.stack([np.sin(th), np.cos(th), ((hour >= 22) | (hour < 6)) * 1.0], -1)
        seasonf = np.eye(DAYS_PER_YEAR)[season]
        own = np.concatenate([
            np.stack([self.sex * 1.0, self.age / 70.0, (self.age < c.child_age) * 1.0, (self.age < c.adult_age) * 1.0,
                      self.hunger, self.energy, self.health, np.minimum(self.money, 100) / 50, self.pregnant / c.gestation,
                      (self.partner >= 0) * 1.0, (self.home >= 0) * 1.0, free_beds / c.house_beds, nkids / 4,
                      youngest_age / 16, np.where(hungriest >= 0, self.hunger[ar, hk], 0),
                      self.enlisted * 1.0, (self.detained > 0) * 1.0, self.asleep * 1.0, self.price / 10, reg,
                      self.edu, self.skill[ar, si, self.job] / c.skill_max,
                      on_shift * 1.0, to_start, to_break_end, workday * 1.0, (self.docked_now) * 1.0,
                      at_home * 1.0, at_work * 1.0], -1),
            places * 1.0, np.eye(len(JOBS))[self.job], self.food / 3.0, self.nutr, self.traits,
            np.broadcast_to(timef[:, None], (M, S, 3)), np.broadcast_to(seasonf[:, None], (M, S, 4)),
            np.broadcast_to(np.concatenate([open_[:, None] * 1.0, school_now[:, None] * 1.0, prices / 5,
                                            np.minimum(self.stock, 60) / 30, vacancies], -1)[:, None], (M, S, 13)),
            np.minimum(self.hired_t, c.job_tenure)[..., None] / c.job_tenure], -1)
        hb = self.hbox[np.clip(self.home, 0, self.H - 1)]
        rels = [np.where((self.home >= 0)[..., None], self._rel_box(hb), 0),
                np.where((self.job > 0)[..., None], self._rel_box(self.job_box[self.job]), 0),
                self._rel_box(c.places["canteen"]), self._rel_box(c.places["school"]), self._rel_box(c.places["market"])]
        pa = np.clip(self.partner, 0, S - 1)
        okp = (self.partner >= 0) & self.alive[ar, pa]
        rels += [np.where(okp[..., None], self.pos[ar, pa] - self.pos, 0), np.where((hungriest >= 0)[..., None], self.pos[ar, hk] - self.pos, 0)]
        zones = np.concatenate(rels, -1) / np.tile([Wd, Hd], len(rels))
        fam = np.stack([okp * 1.0, np.where(okp, self.hunger[ar, pa], 0), np.where(okp, self.energy[ar, pa], 0),
                        np.where(okp, self.at_home()[ar, pa], 0) * 1.0, np.where(okp, self.asleep[ar, pa], 0) * 1.0], -1)
        nbf = []
        for k in range(K):
            j = nb[:, :, k]
            ok = j >= 0
            jj = np.clip(j, 0, S - 1)
            rel = (self.pos[ar, jj] - self.pos) / [Wd, Hd]
            adj = (nd[:, :, k] <= c.reach) & ok
            partner = (self.partner == jj) & ok
            mine = ((self.mother[ar, jj] == self.pid) | (self.father[ar, jj] == self.pid)) & ok
            parent = ((self.mother == self.pid[ar, jj]) | (self.father == self.pid[ar, jj])) & ok
            sib = (((self.mother == self.mother[ar, jj]) & (self.mother > 0)) |
                   ((self.father == self.father[ar, jj]) & (self.father > 0))) & ok & ~mine & ~parent
            f = np.concatenate([np.stack([
                rel[..., 0], rel[..., 1], adj * 1.0, ok * 1.0, self.sex[ar, jj] * 1.0, self.age[ar, jj] / 70,
                (self.age[ar, jj] < c.child_age) * 1.0, partner * 1.0, mine * 1.0, parent * 1.0, sib * 1.0,
                self.opinion[ar, si, jj], A[ar, si, jj], np.minimum(self.money[ar, jj], 100) / 50,
                self.enlisted[ar, jj] * 1.0, (self.detained[ar, jj] > 0) * 1.0,
                self.saw_theft[ar, si, jj] / c.memory_hours, self.robbed_by[ar, si, jj] / c.memory_hours,
                self.shun[ar, si, jj] * 1.0, self.shun[ar, jj, si] * 1.0, (self.court[ar, jj, si] > 0) * 1.0,
                self.price[ar, jj] / 10, self.hunger[ar, jj], (self.partner[ar, jj] >= 0) * 1.0,
                (self.job[ar, jj] > 0) * 1.0], -1), (self.food[ar, jj] > 0) * 1.0], -1) * ok[..., None]
            nbf.append(f)
        obs = np.concatenate([own, zones, fam] + nbf, -1).astype(np.float32)
        alive_n = self.alive.sum(1, keepdims=True)
        employed = np.stack([((self.job == k) & self.alive).sum(1) for k in range(1, len(JOBS))], -1) / 6
        g = np.concatenate([alive_n / S, (self.hunger * self.alive).sum(1, keepdims=True) / np.maximum(alive_n, 1),
                            np.minimum(self.stock, 100) / 50, np.minimum(self.treasury, 1000)[:, None] / 500,
                            employed, (self.pregnant > 0).sum(1, keepdims=True) / 4, (self.t / self.max_steps)[:, None]], -1)
        doing = (self.intent[..., None] == np.arange(WORK, N_BASE)) * 1.0
        cobs = np.concatenate([obs, doing, np.broadcast_to(g[:, None], (M, S, g.shape[1]))], -1).astype(np.float32)
        return obs.reshape(self.N, -1), cobs.reshape(self.N, -1), self.alive.reshape(-1).copy()

    def observe_all(self):
        o, co, _ = self.observe()
        return o, co

    def alive_mask(self):
        return self.alive.reshape(-1).astype(np.float32)

    def decision_mask(self):
        return (self.alive & (self.intent < 0)).reshape(-1).astype(np.float32)

    # ------------------------------------------------------------- intentions
    def _pick_cell(self, m, s, box):
        x0, y0, x1, y1 = box
        return [self.rng.integers(x0, x1 + 1), self.rng.integers(y0, y1 + 1)]

    def _begin(self, m, s, a):
        """Choose where an intention leads; returns False if it cannot start."""
        c = self.cfg
        box = None
        if a == WORK:
            if self.job[m, s] == 0:
                return False
            box = self.job_box[self.job[m, s]]
        elif a == LUNCH:
            box = c.places["canteen"]
        elif a == GROCERIES:
            box = c.places["market"]
        elif a in (GO_HOME, SLEEP):
            if self.home[m, s] < 0:
                if a == SLEEP:              # no bed: sleep where you are
                    self.target[m, s] = self.pos[m, s]
                    return True
                return False
            box = c.houses[self.home[m, s]]
        elif a == SCHOOL:
            box = c.places["school"]
        elif a == GO_MARKET:
            box = c.places["market"]
        elif a == ENLIST:
            box = c.places["station"]
        elif a >= APPLY0:
            box = c.places[JOBS[a - APPLY0 + 1]]
        elif a in (FEED_CHILD, GO_FAMILY):
            return True                     # a moving target, set every step
        else:
            return False
        if _in(self.pos[m, s], box):
            self.target[m, s] = self.pos[m, s]
        else:
            self.target[m, s] = self._pick_cell(m, s, box)
        return True

    # ------------------------------------------------------------------- step
    def step(self, actions):
        c, M, S, r = self.cfg, self.M, self.S, self.rng
        a = np.asarray(actions).reshape(M, S).astype(int)
        was_alive = self.alive.copy()
        rew = np.zeros((M, S))
        ev = [] if self.record else None
        nb, nd = self._nb, self._nd
        hour, season = self.clock()
        ar = np.arange(M)[:, None]
        child = self.age < c.child_age
        adult = self.age >= c.adult_age
        free = self.alive & (self.detained == 0)
        a = np.where(free, a, IDLE)
        kid_ok = np.isin(a, [IDLE, EAT, GO_HOME, SLEEP, GO_FAMILY, SCHOOL, GO_MARKET, LUNCH]) | \
            ((a >= N_BASE) & (((a - N_BASE) % N_PER) == TALK))
        a = np.where(child & ~kid_ok, IDLE, a)
        a = np.where(~adult & (a >= APPLY0) & (a < N_BASE), IDLE, a)
        committed = (self.intent >= 0) & free
        a = np.where(committed, self.intent, a)
        self.docked_now[:] = False
        moved = np.zeros((M, S), bool)
        social = np.zeros((M, S), bool)
        ate_canteen = np.zeros((M, S), bool)
        bought = np.zeros((M, S), bool)
        # ---- start new intentions
        new = free & ~committed & (a >= WORK) & (a < N_BASE)
        for m, s in zip(*np.nonzero(new)):
            if self._begin(m, s, a[m, s]):
                self.intent[m, s] = a[m, s]
                self.intent_t[m, s] = 0
            else:
                a[m, s] = IDLE
        # moving targets
        pa = np.clip(self.partner, 0, S - 1)
        mom, dad = self._parent_slots()
        par = np.where(mom >= 0, mom, dad)
        fam_t = np.where((self.age < c.adult_age), np.clip(par, 0, S - 1), pa)
        fam_ok = np.where(self.age < c.adult_age, par >= 0, (self.partner >= 0) & self.alive[ar, pa])
        hk = np.clip(self._hungriest, 0, S - 1)
        kid_t_ok = self._hungriest >= 0
        doing = self.intent
        sel = (doing == GO_FAMILY) & fam_ok
        self.target = np.where(sel[..., None], self.pos[ar, fam_t], self.target)
        sel = (doing == FEED_CHILD) & kid_t_ok
        self.target = np.where(sel[..., None], self.pos[ar, hk], self.target)
        lost = ((doing == GO_FAMILY) & ~fam_ok) | ((doing == FEED_CHILD) & ~kid_t_ok)
        self.intent = np.where(lost, -1, self.intent)
        # ---- walk (asleep and detained people stay put)
        walking = (self.intent >= 0) & ~self.asleep & free
        d = self.target - self.pos
        person_goal = np.isin(self.intent, [GO_FAMILY, FEED_CHILD])
        near_goal = person_goal & (np.abs(d).max(-1) <= 1)
        steps = np.where(walking & ~near_goal, c.speed, 0)
        dx = np.clip(d[..., 0], -steps, steps)
        rest = steps - np.abs(dx)
        dy = np.clip(d[..., 1], -rest, rest)
        moved = (dx != 0) | (dy != 0)
        self.pos = self.pos + np.stack([dx, dy], -1)
        arrived = (self.intent >= 0) & ((np.abs(self.target - self.pos).sum(-1) == 0) | (person_goal & (np.abs(self.target - self.pos).max(-1) <= 1)))
        # ---- what arriving means
        open_, staff = self.canteen_open()
        shop_open = (hour >= c.shop_hours[0]) & (hour < c.shop_hours[1])
        prices = self.canteen_price()
        served = np.zeros(M, int)
        for m, s in zip(*np.nonzero(arrived & free)):
            it = self.intent[m, s]
            if it in (LUNCH, GROCERIES):
                can = (open_[m] and served[m] < c.canteen_serve * staff[m]) if it == LUNCH else shop_open[m]
                if can:
                    t = int(np.where(self.stock[m] >= 1, self.nutr[m, s] + 0.3 * (self.food[m, s] > 0), 9).argmin())
                    p = prices[m, t]
                    if self.stock[m, t] >= 1 and self.money[m, s] >= p and self.food[m, s].sum() < c.max_food:
                        self.money[m, s] -= p
                        self.treasury[m] += p
                        self.stock[m, t] -= 1
                        self.food[m, s, t] += 1
                        served[m] += 1
                        if ev is not None:
                            ev.append((m, "bought " + FOODS[t], s, -1, float(p)))
                        if it == LUNCH:
                            self._eat_one(m, s, t, rew, ev, at_home=False, boost=c.hot_meal)
                            rew[m, s] += c.w_hot_meal
                            ate_canteen[m, s] = True
                            self.ep["meals_canteen"][m] += 1
                        else:
                            bought[m, s] = True
                            self.ep["groceries"][m] += 1
                self.intent[m, s] = -1
            elif it == SLEEP:
                self.asleep[m, s] = True
            elif it == ENLIST:
                if adult[m, s]:
                    self.enlisted[m, s] = not self.enlisted[m, s]
                    if ev is not None:
                        ev.append((m, "enlisted" if self.enlisted[m, s] else "resigned", s, -1, 0))
                self.intent[m, s] = -1
            elif it >= APPLY0:
                k = it - APPLY0 + 1
                staff_k = ((self.job[m] == k) & self.alive[m]).sum()
                if adult[m, s] and self.job[m, s] != k and self.hired_t[m, s] >= c.job_tenure and staff_k < c.positions[k]:
                    self.job[m, s] = k
                    self.shift_pay[m, s] = 0
                    self.hired_t[m, s] = 0
                    self.ep["hires"][m] += 1
                    if ev is not None:
                        ev.append((m, "hired", s, -1, k))
                self.intent[m, s] = -1
            elif it == FEED_CHILD:
                j = self._hungriest[m, s]
                if j >= 0 and self.food[m, s].sum() > 0 and not self.shun[m, j, s] and self.food[m, j].sum() < c.max_food:
                    self._give_food(m, s, j, rew, ev)
                    social[m, s] = True
                self.intent[m, s] = -1
            elif it in (GO_HOME, GO_MARKET, GO_FAMILY, WORK, SCHOOL):
                self.intent[m, s] = -1          # there: from now on they choose hour by hour
        # ---- eating (anywhere; at home with family it is a family meal)
        eat = (a == EAT) & free & (self.food.sum(-1) > 0) & ~self.asleep
        home_now = self.at_home()
        for m, s in zip(*np.nonzero(eat)):
            t = int(np.where(self.food[m, s] > 0, self.nutr[m, s], 9).argmin())
            self._eat_one(m, s, t, rew, ev, at_home=home_now[m, s])
        self.price = np.clip(self.price + 0.5 * ((a == PRICE_UP) * 1 - (a == PRICE_DOWN)), 0.5, 20.0)
        # ---- dealings with neighbours
        busy = (a >= N_BASE) & free & ~self.asleep
        if busy.any():
            d_all = np.abs(self.pos[:, :, None] - self.pos[:, None, :]).max(-1)
            for m in np.nonzero(busy.any(1))[0]:
                for i in r.permutation(S):
                    if not busy[m, i] or not self.alive[m, i] or self.detained[m, i] > 0:
                        continue
                    k, act = divmod(a[m, i] - N_BASE, N_PER)
                    j = nb[m, i, k]
                    if j < 0 or not self.alive[m, j] or d_all[m, i, j] > c.reach:
                        continue
                    social[m, i] = True
                    self._social(m, i, j, act, rew, ev, d_all)
        # ---- work: pay, produce, dock
        at_work = self.at_work() & ~self.asleep
        on_shift = self.shift_on[self.job, hour[:, None], season[:, None]] & self.alive & (self.job > 0)
        working = on_shift & at_work & free
        infant = np.zeros((M, S), bool)
        kidm = self._kids_matrix()
        infant = (kidm & (self.age[:, None, :] < c.maternity_age)).any(-1) & (self.sex == F)
        excused = infant | (self.health < 0.4) | (self.detained > 0)
        missing = on_shift & ~at_work & ~excused
        k_skill = self.skill[ar, np.arange(S)[None, :], self.job]
        wage = c.wage * (1 + 0.5 * k_skill + 0.5 * self.edu)
        eff = (0.4 + 0.6 * self.energy) * (1 + k_skill)
        for m in range(M):
            idx = np.nonzero(working[m])[0]
            if len(idx):
                pay = wage[m, idx]
                total = pay.sum()
                scale = min(1.0, self.treasury[m] / total) if total > 0 else 1.0
                self.money[m, idx] += pay * scale
                self.shift_pay[m, idx] += pay * scale
                self.treasury[m] -= total * scale
                self.ep["wages"][m] += total * scale
                self.ep["work_hours"][m] += len(idx)
                for s in idx:
                    jn = JOBS[self.job[m, s]]
                    if jn in ("farm", "orchard", "dairy"):
                        f = ("farm", "orchard", "dairy").index(jn)
                        out = c.output[f] * c.season_yield[f][season[m]] * eff[m, s]
                        self.stock[m, f] += out
                        self.ep["produced"][m] += out
                    elif jn == "warehouse":
                        self.treasury[m] += c.export * eff[m, s]
                        self.ep["exports"][m] += c.export * eff[m, s]
                    self.skill[m, s, self.job[m, s]] = min(c.skill_max, self.skill[m, s, self.job[m, s]] + c.skill_gain)
        rew += c.w_purpose * working
        for m, s in zip(*np.nonzero(missing)):
            loss = min(self.money[m, s], c.dock * wage[m, s])
            self.money[m, s] -= loss
            self.treasury[m] += loss
            self.docked_now[m, s] = True
            self.ep["docked_hours"][m] += 1
            if ev is not None and loss > 0:
                ev.append((m, "docked", s, -1, round(float(loss), 1)))
        # end of a shift block: report what was earned
        nxt_on = self.shift_on[self.job, ((hour + 1) % DAY)[:, None], (((self.t + 1) // DAY) % DAYS_PER_YEAR)[:, None]]
        ending = on_shift & ~nxt_on & (self.shift_pay > 0)
        for m, s in zip(*np.nonzero(ending)):
            if ev is not None:
                ev.append((m, "earned", s, -1, round(float(self.shift_pay[m, s]), 1)))
            self.shift_pay[m, s] = 0
        # school
        school_now = (hour >= c.school_hours[0]) & (hour < c.school_hours[1]) & (season < 3)
        pupils = free & ~self.asleep & _in(self.pos, c.places["school"]) & school_now[:, None] & \
            (self.age >= c.school_ages[0]) & (self.age < c.school_ages[1])
        self.edu = np.where(pupils, np.minimum(1.0, self.edu + c.edu_gain), self.edu)
        rew += c.w_purpose * pupils
        self.ep["school_hours"] += pupils.sum(1)
        self.hired_t += 1
        # an errand that has not arrived in time is given up
        self.intent_t = np.where(self.intent >= 0, self.intent_t + 1, 0)
        stuck = (self.intent >= 0) & ~self.asleep & (self.intent_t > c.intent_patience)
        self.intent = np.where(stuck, -1, self.intent)
        # ---- bodies, families, time
        self._lifecycle(rew, ev, hour, season, working)
        # ---- happiness
        al = self.alive
        craving = np.clip((c.crave_level - self.nutr) / c.crave_level, 0, 1).sum(-1)
        rew += c.w_alive * self.health * al
        rew -= c.w_hunger * self.hunger * al
        rew -= c.w_starving * (self.hunger >= 1) * al
        rew -= c.w_health * (1 - self.health) * al
        rew -= c.w_craving * craving * al
        rew -= c.w_tired * (np.clip(0.5 - self.energy, 0, 0.5) / 0.5) ** 2 * al    # tiredness bites when energy runs low
        rew -= c.w_detained * (self.detained > 0) * (1 - 0.5 * self.traits[..., 2]) * al
        mutual = (self.opinion > 0.3) & (np.transpose(self.opinion, (0, 2, 1)) > 0.3) & al[:, None, :]
        rew += c.w_friends * np.minimum(mutual.sum(2), 5) * (0.5 + self.traits[..., 0]) * al
        pa = np.clip(self.partner, 0, S - 1)
        rew += c.w_partner * (self.partner >= 0) * al
        near = (self.partner >= 0) & self.alive[ar, pa] & (np.abs(self.pos[ar, pa] - self.pos).max(-1) <= c.reach)
        home_now = self.at_home()
        rew += c.w_together * near * (1 + (home_now & home_now[ar, pa])) * al
        reg = (self.opinion * al[:, :, None]).sum(1) / np.maximum(al.sum(1, keepdims=True) - 1, 1)
        rew += c.w_esteem * reg * al
        rew += c.w_security * np.log1p(np.maximum(self.money, 0)) * (0.5 + self.traits[..., 1]) * al
        kidm = self._kids_matrix()
        n = kidm.sum(-1)
        fed = ((1 - self.hunger)[:, None, :] * kidm).sum(-1) / np.maximum(n, 1)
        suffering = np.clip((self.hunger - 0.4) / 0.6, 0, 1) + craving / 3 + (1 - self.health)
        distress = (suffering[:, None, :] * kidm).sum(-1)
        rew += c.w_children * np.where(n > 0, fed, 0) * (0.5 + self.traits[..., 0]) * al
        rew -= c.w_child_distress * distress * (0.5 + self.traits[..., 0]) * al
        rew = np.where(was_alive, rew, 0.0)
        self.ep["person_steps"] += al.sum(1)
        self.ep["happiness"] += rew.sum(1)
        # ---- what each person is doing this hour (for the viewer)
        act = np.full((M, S), ACT_ID["idle"])
        act = np.where(home_now, ACT_ID["home"], act)
        act = np.where(social, ACT_ID["socialising"], act)
        act = np.where(moved, ACT_ID["walking"], act)
        act = np.where(bought, ACT_ID["buying"], act)
        act = np.where(ate_canteen | eat, ACT_ID["eating"], act)
        act = np.where(pupils, ACT_ID["school"], act)
        act = np.where(working, ACT_ID["working"], act)
        mom, _ = self._parent_slots()
        mm = np.clip(mom, 0, S - 1)
        nursing = al & (self.age < c.nursing_age) & (mom >= 0) & (np.abs(self.pos[ar, mm] - self.pos).max(-1) <= 1)
        act = np.where(nursing & ~self.asleep, ACT_ID["nursing"], act)
        act = np.where(self.asleep, ACT_ID["sleeping"], act)
        act = np.where(self.detained > 0, ACT_ID["detained"], act)
        self.act = act
        # ---- business stock, clocks
        self.stock *= (1 - c.stock_spoil)
        for arr in (self.saw_theft, self.robbed_by, self.court):
            np.maximum(arr - 1, 0, out=arr)
        released = self.detained == 1
        self.detained = np.maximum(self.detained - 1, 0)
        if released.any():
            self.intent = np.where(released, -1, self.intent)
        self.t += 1
        died = was_alive & ~self.alive
        done_w = (self.t >= self.max_steps + c.start_hour) | ~self.alive.any(1)
        done = (died | done_w[:, None]).reshape(-1)
        info = dict(timeout=np.repeat(self.t >= self.max_steps + c.start_hour, S) & was_alive.reshape(-1),
                    fallen=died.reshape(-1), mask=was_alive.reshape(-1), events=ev)
        if done_w.any():
            for m in np.nonzero(done_w)[0]:
                st = {k: float(v[m]) for k, v in self.ep.items()}
                st["population"] = float(self.alive[m].sum())
                st["years"] = float((self.t[m] - c.start_hour) / YEAR)
                st["employed"] = float(((self.job[m] > 0) & self.alive[m]).sum())
                st["treasury"] = float(self.treasury[m])
                self.stats.append(st)
            self._reset(np.nonzero(done_w)[0])
        obs, cobs, _ = self.observe()
        return obs, cobs, rew.reshape(-1).astype(np.float32), done, info

    # ------------------------------------------------------------ actions
    def _eat_one(self, m, s, t, rew, ev, at_home, boost=1.0):
        c = self.cfg
        self.food[m, s, t] -= 1
        relief = boost * c.food_relief * (c.satiety_floor + (1 - c.satiety_floor) * (1 - self.nutr[m, s, t]))
        self.hunger[m, s] = max(0.0, self.hunger[m, s] - relief)
        self.nutr[m, s, t] = min(1.0, self.nutr[m, s, t] + c.nutrient_gain)
        family = False
        if at_home:
            h = self.home[m, s]
            here = self.alive[m] & (self.home[m] == h) & ~self.asleep[m] & _in(self.pos[m], self.cfg.houses[h])
            here[s] = False
            family = bool(here.any())
        if family:
            rew[m, s] += c.w_family_meal
            self.ep["family_meals"][m] += 1
        if ev is not None:
            ev.append((m, "family meal" if family else "ate " + FOODS[t], s, -1, t))

    def _give_food(self, m, i, j, rew, ev):
        c, O = self.cfg, self.opinion
        t = int(np.where(self.food[m, i] > 0, self.nutr[m, j], 9).argmin())
        need = 0.5 * (1 - self.nutr[m, j, t]) + 0.5 * self.hunger[m, j]
        self.food[m, i, t] -= 1
        self.food[m, j, t] += 1
        O[m, j, i] = min(1, O[m, j, i] + 0.15 * (0.2 + 0.8 * need))
        self._witness_good(m, i, 0.03)
        self.ep["gifts_food"][m] += 1
        if ev is not None:
            ev.append((m, "gave " + FOODS[t], i, j, 1))

    def _move_in(self, m, i, j, ev):
        """Partners i and j try to share a house: hers, his, or an empty one."""
        c = self.cfg
        res = self.residents()[m]
        for h, mover in ((self.home[m, i], j), (self.home[m, j], i)):
            if h >= 0 and res[h] < c.house_beds:
                if self.home[m, mover] != h:
                    self.home[m, mover] = h
                    if ev is not None:
                        ev.append((m, "moved in", mover, -1, int(h)))
                return
        empty = np.nonzero(res == 0)[0]
        if len(empty):
            h = int(empty[0])
            for x in (i, j):
                self.home[m, x] = h
                if ev is not None:
                    ev.append((m, "moved in", x, -1, h))

    def _social(self, m, i, j, act, rew, ev, d_all):
        c = self.cfg
        refused = self.shun[m, j, i]
        O = self.opinion
        if act == GIVE_FOOD and self.food[m, i].sum() > 0 and not refused and self.food[m, j].sum() < c.max_food:
            self._give_food(m, i, j, rew, ev)
        elif act == GIVE_COIN and self.money[m, i] >= 1 and not refused:
            self.money[m, i] -= 1
            self.money[m, j] += 1
            O[m, j, i] = min(1, O[m, j, i] + 0.1)
            self._witness_good(m, i, 0.02)
            self.ep["gifts_coin"][m] += 1
            if ev is not None:
                ev.append((m, "gave coin", i, j, 1))
        elif act == BUY and not refused and not self.shun[m, i, j] and self.food[m, j].sum() > 0 \
                and self.money[m, i] >= self.price[m, j] and self.food[m, i].sum() < c.max_food:
            t = int(np.where(self.food[m, j] > 0, self.nutr[m, i] + 0.2 * (self.food[m, i] > 0), 9).argmin())
            p = self.price[m, j]
            self.money[m, i] -= p
            self.money[m, j] += p
            self.food[m, j, t] -= 1
            self.food[m, i, t] += 1
            O[m, i, j] = min(1, O[m, i, j] + 0.03)
            O[m, j, i] = min(1, O[m, j, i] + 0.03)
            self.ep["trades"][m] += 1
            if ev is not None:
                ev.append((m, "bought " + FOODS[t], i, j, round(float(p), 1)))
        elif act == TAKE and self.money[m, j] >= 1.0:
            amt = round(min(10.0, max(0.5, 0.5 * self.money[m, j])), 1)
            self.money[m, j] -= amt
            self.money[m, i] += amt
            O[m, j, i] = max(-1, O[m, j, i] - 0.6)
            self.robbed_by[m, j, i] = c.memory_hours
            rew[m, j] -= c.w_robbed
            wit = self.alive[m] & ~self.asleep[m] & (np.abs(self.pos[m] - self.pos[m, i]).max(1) <= c.witness_radius)
            wit[i] = False
            if not self.asleep[m, j]:
                wit[j] = True
            self.saw_theft[m, wit, i] = c.memory_hours
            O[m, wit, i] = np.maximum(-1, O[m, wit, i] - 0.3)
            self.ep["thefts"][m] += 1
            if ev is not None:
                ev.append((m, "theft", i, j, amt))
        elif act == TALK and not refused and not self.shun[m, i, j]:
            for (x, y) in ((i, j), (j, i)):
                trust = max(0.0, O[m, y, x])
                keep = O[m, y, x]
                O[m, y] += c.gossip_rate * trust * (O[m, x] - O[m, y])
                O[m, y, y] = 0
                O[m, y, x] = keep
            O[m, i, j] = min(1, O[m, i, j] + 0.02)
            O[m, j, i] = min(1, O[m, j, i] + 0.02)
            np.clip(O[m], -1, 1, out=O[m])
            self.ep["talks"][m] += 1
            if ev is not None and self.rng.random() < 0.2:
                ev.append((m, "talked", i, j, 0))
        elif act == COURT and not refused and self.age[m, i] >= c.partner_age and self.age[m, j] >= c.partner_age \
                and self.sex[m, i] != self.sex[m, j] and self.partner[m, i] < 0 and self.partner[m, j] < 0 \
                and not self.related(m, i, j):
            self.court[m, i, j] = c.court_window
            if ev is not None:
                ev.append((m, "courted", i, j, 0))
            if self.court[m, j, i] > 0:
                A = self.attractiveness(m)
                if A[i, j] >= c.attract_threshold and A[j, i] >= c.attract_threshold:
                    self.partner[m, i], self.partner[m, j] = j, i
                    O[m, i, j] = min(1, O[m, i, j] + 0.3)
                    O[m, j, i] = min(1, O[m, j, i] + 0.3)
                    self.ep["partnerships"][m] += 1
                    if ev is not None:
                        ev.append((m, "partnered", i, j, 0))
                    fem, mal = (i, j) if self.sex[m, i] == F else (j, i)
                    self._move_in(m, fem, mal, ev)
        elif act == SHUN:
            self.shun[m, i, j] = not self.shun[m, i, j]
            if self.shun[m, i, j]:
                self.ep["shuns"][m] += 1
                if self.partner[m, i] == j:
                    self.partner[m, i] = self.partner[m, j] = -1
                    self.ep["breakups"][m] += 1
                    if ev is not None:
                        ev.append((m, "separated", i, j, 0))
            if ev is not None:
                ev.append((m, "shunned" if self.shun[m, i, j] else "forgave", i, j, 0))
        elif act == DETAIN and self.saw_theft[m, i, j] > 0 and self.age[m, i] >= c.adult_age:
            near = self.alive[m] & (d_all[m, j] <= c.reach) & (self.saw_theft[m, :, j] > 0) & (self.age[m] >= c.adult_age)
            near[j] = False
            citizens = near.sum() >= 2
            if self.enlisted[m, i] or citizens:
                police = self.enlisted[m, i] and not citizens
                takers = [i] if police else list(np.nonzero(near)[0])
                amt = max(0.0, self.money[m, j])
                self.money[m, j] = 0
                for x in takers:
                    self.money[m, x] += amt / len(takers)
                self.detained[m, j] = c.detain_hours
                self.intent[m, j] = -1
                self.asleep[m, j] = False
                st = c.places["station"]
                self.pos[m, j] = [st[0], st[1]]
                self.saw_theft[m, :, j] = 0
                self.enlisted[m, j] = False
                seen = self.alive[m] & (np.abs(self.pos[m] - self.pos[m, i]).max(1) <= c.witness_radius)
                O[m, seen, i] = np.minimum(1, O[m, seen, i] + 0.1)
                self.ep["detentions"][m] += 1
                self.ep["citizen_arrests"][m] += (not police)
                if ev is not None:
                    ev.append((m, "detained" if police else "citizens' arrest", i, j, round(amt, 1)))

    def _witness_good(self, m, i, amount):
        c = self.cfg
        wit = self.alive[m] & ~self.asleep[m] & (np.abs(self.pos[m] - self.pos[m, i]).max(1) <= c.witness_radius)
        wit[i] = False
        self.opinion[m, wit, i] = np.minimum(1, self.opinion[m, wit, i] + amount)

    # ---------------------------------------------------------- life & death
    def _lifecycle(self, rew, ev, hour, season, working):
        c, M, S, r = self.cfg, self.M, self.S, self.rng
        al = self.alive
        ar = np.arange(M)[:, None]
        night = ((hour >= 22) | (hour < 6))[:, None]
        home_now = self.at_home()
        # energy: drains awake (more at night, at work, for children), returns asleep (best in bed at night)
        drain = c.energy_awake * np.where(night, c.night_drain, 1.0) * np.where(working, c.work_drain, 1.0) * \
            np.where(self.age < c.child_age, c.child_drain, 1.0)
        gain = c.sleep_gain * np.where(night, c.night_sleep, c.day_sleep) * np.where(home_now, 1.0, c.rough_sleep)
        self.energy = np.clip(np.where(self.asleep, self.energy + gain, self.energy - drain), 0, 1)
        self.ep["sleep_home"] += (self.asleep & home_now & al).sum(1)
        self.ep["sleep_rough"] += (self.asleep & ~home_now & al).sum(1)
        self.ep["night_awake"] += (~self.asleep & night & al & (self.detained == 0)).sum(1)
        collapse = al & ~self.asleep & (self.energy <= 0)
        for m, s in zip(*np.nonzero(collapse)):
            self.asleep[m, s] = True
            self.intent[m, s] = SLEEP
            self.target[m, s] = self.pos[m, s]
            if ev is not None:
                ev.append((m, "fell asleep outside" if not home_now[m, s] else "fell asleep", s, -1, 0))
        # the body clock keeps a night's sleep going until morning; naps end when rested; hunger wakes anyone
        wake = self.asleep & (((self.energy >= 0.97) & ~night) | (self.hunger >= 0.95))
        self.asleep = self.asleep & ~wake
        self.intent = np.where(wake, -1, self.intent)
        # hunger, nutrients, health
        rate = np.where(self.asleep, c.hunger_asleep, c.hunger_awake) * np.where(self.pregnant > 0, c.pregnant_hunger, 1.0)
        self.hunger = np.where(al, np.minimum(1.0, self.hunger + rate), 0)
        self.nutr = np.maximum(0.0, self.nutr - c.nutrient_decay)
        missing = (self.nutr <= 0).sum(-1)
        heal = c.heal_rate * (1 + 0.5 * home_now + 0.5 * self.asleep)
        self.health = np.where(self.hunger >= 1, self.health - c.starve_damage,
                               np.where((self.hunger < 0.6) & (missing == 0), np.minimum(1, self.health + heal), self.health))
        self.health = self.health - c.deficiency_damage * missing
        self.age += al / YEAR
        # infants nurse when they are with their fed mother
        mom, _ = self._parent_slots()
        mm = np.clip(mom, 0, S - 1)
        nursing = al & (self.age < c.nursing_age) & (mom >= 0) & (np.abs(self.pos[ar, mm] - self.pos).max(-1) <= 1) \
            & (self.hunger[ar, mm] < 0.8)
        if nursing.any():
            self.hunger = np.where(nursing, np.maximum(0, self.hunger - c.nurse_relief), self.hunger)
            self.nutr = np.where(nursing[..., None], np.minimum(1, self.nutr + c.nurse_nutrient), self.nutr)
            cost = np.zeros((M, S))
            np.add.at(cost, (np.nonzero(nursing)[0], mm[nursing]), c.nurse_cost)
            self.hunger = np.minimum(1, self.hunger + cost)
        # babies and toddlers stay with their mother
        tiny = al & (self.age < 3) & (mom >= 0)
        self.pos = np.where(tiny[..., None] & ~self.asleep[..., None], self.pos[ar, mm], self.pos)
        spoil = (self.food > 0) & (r.random((M, S, 3)) < self.food / c.spoil_hours)
        self.food -= spoil
        p_old = np.where(self.age > c.old_age, ((self.age - c.old_age) / 30.0) ** 2 * 0.0025, 0)
        dies = al & ((self.health <= 0) | (r.random((M, S)) < p_old))
        for m, s in zip(*np.nonzero(dies)):
            cause = "old age" if self.health[m, s] > 0 else "starved" if self.hunger[m, s] >= 1 else "malnutrition"
            self._die(m, s, rew, ev, cause)
        # pregnancy and birth
        lo, hi = c.fertile_ages
        cand = self.alive & (self.sex == F)
        for m, s in zip(*np.nonzero(cand & (self.pregnant > 0))):
            self.pregnant[m, s] -= 1
            if self.pregnant[m, s] == 0:
                self._birth(m, s, rew, ev)
        res = self.residents()
        home_now = self.at_home()
        ready = cand & (self.pregnant == 0) & (self.partner >= 0) & home_now & (self.age >= lo) & (self.age <= hi) & \
            (self.hunger < 0.6)
        for m, s in zip(*np.nonzero(ready)):
            pa = self.partner[m, s]
            h = self.home[m, s]
            if self.alive[m, pa] and home_now[m, pa] and self.home[m, pa] == h and self.hunger[m, pa] < 0.6 \
                    and res[m, h] < c.house_beds and (~self.alive[m]).any() and r.random() < c.conceive_prob:
                self.pregnant[m, s] = c.gestation
                if ev is not None:
                    ev.append((m, "expecting", s, int(pa), 0))

    def _birth(self, m, mother, rew, ev):
        c, r = self.cfg, self.rng
        freeslots = np.nonzero(~self.alive[m])[0]
        if len(freeslots) == 0:
            return
        s = int(freeslots[0])
        father = self.partner[m, mother]
        ft = self.traits[m, father] if father >= 0 else self.traits[m, mother]
        traits = (self.traits[m, mother] + ft) / 2 + r.normal(0, c.mutation, 4)
        self._new_person(m, s, int(r.integers(0, 2)), 0.0, traits, self.pos[m, mother].tolist(), mother=self.pid[m, mother],
                         father=self.pid[m, father] if father >= 0 else 0)
        self.hunger[m, s] = 0.2
        self.home[m, s] = self.home[m, mother]
        for p in (mother, father):
            if p >= 0:
                self.opinion[m, p, s] = self.opinion[m, s, p] = 0.8
                rew[m, p] += c.w_birth
        self.ep["births"][m] += 1
        if ev is not None:
            ev.append((m, "born", s, int(mother), 0))

    def _die(self, m, s, rew, ev, cause="old age"):
        c = self.cfg
        rew[m, s] -= c.w_death
        pid = self.pid[m, s]
        for j in np.nonzero(self.alive[m])[0]:
            if j == s:
                continue
            kin = (self.partner[m, j] == s or self.mother[m, j] == pid or self.father[m, j] == pid
                   or self.mother[m, s] == self.pid[m, j] or self.father[m, s] == self.pid[m, j])
            if kin:
                parent = self.mother[m, s] == self.pid[m, j] or self.father[m, s] == self.pid[m, j]
                rew[m, j] -= c.w_grief * (0.5 + self.traits[m, j, 0]) * (c.child_grief if parent else 1.0)
        p = self.partner[m, s]
        if p >= 0:
            self.partner[m, p] = -1
        self.alive[m, s] = False
        self.asleep[m, s] = False
        self.intent[m, s] = -1
        self.enlisted[m, s] = False
        self.pregnant[m, s] = 0
        self.home[m, s] = -1
        self.job[m, s] = 0
        self.ep["deaths"][m] += 1
        self.ep["starved"][m] += cause == "starved"
        self.ep["malnourished"][m] += cause == "malnutrition"
        self.ep["old_age"][m] += cause == "old age"
        if ev is not None:
            ev.append((m, f"died ({cause})", s, -1, round(float(self.age[m, s]), 1)))
