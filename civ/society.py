"""Society v2: a small human society that has to discover cooperation.

See DESIGN.md for the reasoning. People live whole lives in one continuous
world (60 steps = 1 year): they eat or starve, work, farm, trade, steal, gossip,
court, pair up, raise children and die. Their reward is *happiness*: fed and
healthy with a varied diet, safe, belonging (friends, partner, thriving
children), respected, doing useful work, and - with diminishing returns -
having savings.

Three foods come from three places: grain (fields you plant), fruit (orchard
trees that regrow) and dairy (a herd you tend). Eating the same food again gives
less and less; going without a food type harms health; experience makes you a
better producer of what you produce. Trade between specialists is therefore
positive-sum.

No transfer is automatic: every gift, sale, theft, detention, courtship and
shunning is an action someone chose, and receiving anything requires that the
receiver has not shunned the giver.

Vectorised over M independent worlds with S person slots each (dead or unborn
slots are masked out of training).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

YEAR = 60                       # steps per year
F, M_ = 0, 1                    # sexes
GRAIN, FRUIT, DAIRY = 0, 1, 2
FOODS = ["grain", "fruit", "dairy"]
K = 4                           # neighbours you can deal with / see in detail
# stay, 4 moves, interact, eat, price+, price-, enlist, then intentions: each walks one step toward its
# target per step (the body's motor skill) and, for work, does the work on arrival. Choosing is still theirs.
INTENTS = ["harvest_grain", "pick_fruit", "milk", "plant", "haul", "go_home", "go_market", "go_partner",
           "go_child", "go_station"]
N_BASE = 10 + len(INTENTS)
N_PER = 8                       # per-neighbour actions
(GIVE_FOOD, GIVE_COIN, BUY, TAKE, TALK, COURT, SHUN, DETAIN) = range(N_PER)
ACT_NAMES = ["stay", "up", "down", "left", "right", "interact", "eat", "price+", "price-", "enlist"] + INTENTS + \
    [f"{a}>{k}" for k in range(K) for a in ["give_food", "give_coin", "buy", "take", "talk", "court", "shun", "detain"]]
MOVES = np.zeros((N_BASE + N_PER * K, 2), int)
MOVES[1:5] = [[0, 1], [0, -1], [-1, 0], [1, 0]]


@dataclass
class SocietyConfig:
    width: int = 24
    height: int = 18
    slots: int = 20
    founders: int = 10
    episode_years: float = 60.0
    # places (x0, y0, x1, y1)
    shelves: tuple = (1, 13, 2, 16)
    dock: tuple = (7, 13, 8, 16)
    fields: tuple = (9, 12, 14, 14)        # grain: plant, grow, harvest
    orchard: tuple = (15, 7, 19, 8)        # fruit trees: regrow after picking
    pasture: tuple = (5, 6, 7, 9)          # herd: tend for dairy, regrows
    market: tuple = (10, 7, 12, 9)         # the three food sources surround the market
    station: tuple = (1, 1, 2, 2)
    homes: tuple = ((4, 1, 5, 2), (9, 1, 10, 2), (14, 1, 15, 2), (19, 1, 20, 2), (20, 12, 21, 13), (1, 8, 2, 9))
    # work and food
    crate_pay: float = 5.0
    team_bonus: float = 0.5
    team_window: int = 6
    seed_cost: int = 1              # grain kept back from your own stock to sow a plot
    wild_seed: float = 0.002        # an empty plot re-seeds itself with wild grain
    grow_steps: int = 60           # grain
    fruit_regrow: int = 90
    dairy_regrow: int = 70
    base_yield: tuple = (2, 1, 1)  # grain, fruit, dairy per harvest before skill
    skill_gain: float = 0.08       # per harvest of that food
    skill_max: float = 2.0
    max_food: int = 8
    spoil_steps: int = 150
    price0: float = 3.0
    founder_food: tuple = (2, 1, 1)
    ripe_at_start: float = 0.5
    # bodies and diet
    hunger_rate: float = 1.0 / 240.0
    pregnant_hunger: float = 1.6
    food_relief: float = 0.45
    nutrient_decay: float = 1.0 / 240.0   # a full store of one nutrient lasts ~4 years; one meal ~2
    nutrient_gain: float = 0.5
    satiety_floor: float = 0.1         # sensory-specific satiety: a food you are full of barely satisfies
    starve_damage: float = 0.01
    deficiency_damage: float = 0.003   # per missing nutrient; a deficient body cannot heal
    heal_rate: float = 0.004
    shelter_heal: float = 1.5          # extra healing while resting at your own home
    intent_patience: int = 30         # give up an errand that has not arrived after this many steps
    adult_age: float = 16.0
    child_age: float = 12.0
    old_age: float = 60.0
    # social
    witness_radius: int = 5
    memory_steps: int = 90
    detain_steps: int = 60
    gossip_rate: float = 0.3
    court_window: int = 12
    attract_threshold: float = 0.45
    conceive_prob: float = 0.015     # per step that fed, fertile partners spend together
    gestation: int = 45
    nursing_age: float = 2.0         # infants nurse from their mother when they are together
    nurse_relief: float = 0.02
    nurse_nutrient: float = 0.01
    nurse_cost: float = 0.006        # extra hunger for the nursing mother
    fertile_ages: tuple = (18.0, 45.0)
    mutation: float = 0.08
    # happiness weights
    w_alive: float = 0.03             # contentment: being alive and healthy is worth something
    w_hunger: float = 0.04
    w_starving: float = 0.06
    w_health: float = 0.03
    w_variety: float = 0.0
    w_craving: float = 0.04           # per nutrient: the felt craving for a food your body is running low on
    crave_level: float = 0.5           # craving starts when a nutrient store drops below this
    w_robbed: float = 0.5
    w_detained: float = 0.02
    w_friends: float = 0.006
    w_partner: float = 0.012
    w_together: float = 0.015         # companionship: time with your partner, twice as sweet at home
    w_children: float = 0.03
    w_esteem: float = 0.015
    w_purpose: float = 0.03           # small satisfaction of work itself; its real value is what it feeds
    w_security: float = 0.003
    w_birth: float = 1.5
    w_grief: float = 2.0
    w_death: float = 5.0


def _in(p, box):
    x0, y0, x1, y1 = box
    return (p[..., 0] >= x0) & (p[..., 0] <= x1) & (p[..., 1] >= y0) & (p[..., 1] <= y1)


def _cells(box):
    x0, y0, x1, y1 = box
    return [[x, y] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


class Society:
    def __init__(self, n_worlds: int, cfg: SocietyConfig | None = None, seed: int = 0, record: bool = False):
        self.cfg = c = cfg or SocietyConfig()
        self.M, self.S = n_worlds, c.slots
        self.N = self.M * self.S
        self.rng = np.random.default_rng(seed)
        self.record = record
        # food sites: (x, y, food type)
        sites = [(x, y, GRAIN) for x, y in _cells(c.fields)] + [(x, y, FRUIT) for x, y in _cells(c.orchard)] + \
                [(x, y, DAIRY) for x, y in _cells(c.pasture)]
        self.sites = np.array([[x, y] for x, y, _ in sites])
        self.site_type = np.array([t for _, _, t in sites])
        self.P = len(sites)
        self.regrow = np.array([c.grow_steps, c.fruit_regrow, c.dairy_regrow])[self.site_type]
        self.H = len(c.homes)
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
        self.health = z(S)
        self.nutr = z(S, 3)                       # nutrient levels per food type
        self.skill = z(S, 3)
        self.money = z(S)
        self.food = z(S, 3, dt=int)
        self.price = z(S)
        self.carry = z(S, dt=bool)
        self.pregnant = z(S, dt=int)
        self.partner = np.full((M, S), -1)
        self.home = np.full((M, S), -1)
        self.home_seen = np.full((M, S), 999)   # steps since last at own home
        self.intent = np.full((M, S), -1)       # the intention being carried out (action id), -1 when deciding
        self.intent_t = np.zeros((M, S), int)
        self.enlisted = z(S, dt=bool)
        self.detained = z(S, dt=int)
        self.traits = z(S, 4)                     # empathy, greed, boldness, charm
        self.pid = z(S, dt=int)
        self.mother = z(S, dt=int)
        self.father = z(S, dt=int)
        self.opinion = z(S, S)                    # opinion[i, j]: what i thinks of j, in [-1, 1]
        self.saw_theft = z(S, S, dt=int)
        self.robbed_by = z(S, S, dt=int)
        self.court = z(S, S, dt=int)
        self.shun = z(S, S, dt=bool)
        self.site_t = z(self.P, dt=int)           # grain: 0 empty, >0 growing, -1 ripe; others: >0 regrowing, -1 ready
        self.last_ship = np.full(M, 999)
        self.t = z(dt=int)
        self.ep = None

    def _new_person(self, m, s, sex, age, traits, pos, mother=0, father=0):
        c = self.cfg
        adult = age >= c.adult_age
        self.alive[m, s] = True
        self.sex[m, s] = sex
        self.age[m, s] = age
        self.pos[m, s] = pos
        self.hunger[m, s] = 0.3
        self.health[m, s] = 1.0
        self.nutr[m, s] = 0.6
        self.skill[m, s] = 0.0
        self.money[m, s] = 5.0 if adult else 0.0
        self.food[m, s] = np.array(c.founder_food) if adult else 0
        self.price[m, s] = c.price0
        self.carry[m, s] = False
        self.pregnant[m, s] = 0
        self.partner[m, s] = -1
        self.home[m, s] = -1
        self.home_seen[m, s] = 999
        self.intent[m, s] = -1
        self.enlisted[m, s] = False
        self.detained[m, s] = 0
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
            ripe = r.random(self.P) < c.ripe_at_start
            self.site_t[m] = np.where(ripe, -1, np.where(self.site_type == GRAIN, 0, r.integers(1, self.regrow + 1)))
            self.last_ship[m] = 999
            self.t[m] = 0
            for s in range(c.founders):
                pos = [r.integers(0, c.width), r.integers(0, c.height)]
                self._new_person(m, s, s % 2, r.uniform(18, 35), r.random(4), pos)
                # founders arrive with varied bodies and provisions, so every craving is met somewhere early on
                self.nutr[m, s] = r.uniform(0.1, 0.9, 3)
                self.food[m, s] = r.integers(0, 3, 3)
        if self.ep is None:
            self.ep = {k: np.zeros(self.M) for k in [
                "births", "deaths", "starved", "malnourished", "crates", "harvest_grain", "harvest_fruit", "harvest_dairy", "trades",
                "gifts_food", "gifts_coin", "thefts", "detentions", "citizen_arrests", "partnerships", "breakups",
                "talks", "shuns", "person_steps", "happiness", "police_pay", "variety"]}
        for k in self.ep:
            self.ep[k][ws] = 0

    # --------------------------------------------------------------- helpers
    def related(self, m, i, j):
        pi, pj = self.pid[m, i], self.pid[m, j]
        mi, fi, mj, fj = self.mother[m, i], self.father[m, i], self.mother[m, j], self.father[m, j]
        return (mi == pj or fi == pj or mj == pi or fj == pi or (mi > 0 and mi == mj) or (fi > 0 and fi == fj))

    def _parent_slots(self):
        """Slot of each person's living mother and father (-1 if none)."""
        out = []
        for who in (self.mother, self.father):
            eq = (self.pid[:, None, :] == who[:, :, None]) & self.alive[:, None, :] & (who[:, :, None] > 0)
            out.append(np.where(eq.any(-1), eq.argmax(-1), -1))
        return out

    def at_home(self):
        """Who is standing inside their own home (M, S)."""
        hb = np.array(self.cfg.homes)[np.clip(self.home, 0, self.H - 1)]
        x, y = self.pos[..., 0], self.pos[..., 1]
        return self.alive & (self.home >= 0) & (x >= hb[..., 0]) & (x <= hb[..., 2]) & (y >= hb[..., 1]) & (y <= hb[..., 3])

    def attractiveness(self, m=None):
        """A[(m,) i, j]: how attractive j is to i (0..1): health, provisioning, how i regards j
        (kindness shown, reputation heard), charm and closeness in age."""
        sl = slice(None) if m is None else m
        health = self.health[sl][..., None, :]
        prov = np.tanh((self.food[sl].sum(-1) + self.money[sl] / 5.0) / 5.0)[..., None, :]
        op = (self.opinion[sl] + 1) / 2
        charm = self.traits[sl][..., None, :, 3]
        age = self.age[sl]
        agec = np.clip(1 - np.abs(age[..., :, None] - age[..., None, :]) / 20.0, 0, 1)
        return 0.25 * health + 0.2 * prov + 0.25 * op + 0.15 * charm + 0.15 * agec

    def _neighbours(self):
        d = np.abs(self.pos[:, :, None] - self.pos[:, None, :]).max(-1).astype(float)
        d[:, np.arange(self.S), np.arange(self.S)] = 1e9
        d = np.where(self.alive[:, None, :], d, 1e9)
        idx = np.argsort(d, axis=2, kind="stable")[:, :, :K]
        dd = np.take_along_axis(d, idx, 2)
        return np.where(dd < 1e8, idx, -1), dd

    def _rel(self, box):
        x0, y0, x1, y1 = box
        tx = np.clip(self.pos[..., 0], x0, x1)
        ty = np.clip(self.pos[..., 1], y0, y1)
        return np.stack([tx - self.pos[..., 0], ty - self.pos[..., 1]], -1)

    def _rel_site(self, mask):
        d = self.sites[None, None] - self.pos[:, :, None]
        dist = np.abs(d).sum(-1).astype(float)
        dist = np.where(mask[:, None], dist, 1e9)
        k = dist.argmin(-1)
        out = np.take_along_axis(d, k[..., None, None], axis=2)[:, :, 0]
        return np.where((dist.min(-1) < 1e9)[..., None], out, 0)

    def _kids(self, s):
        c = self.cfg
        return self.alive & ((self.mother == self.pid[:, s:s + 1]) | (self.father == self.pid[:, s:s + 1])) \
            & (self.pid[:, s:s + 1] > 0) & (self.age < c.adult_age)

    # ----------------------------------------------------------- observation
    def observe(self):
        c, M, S = self.cfg, self.M, self.S
        Wd, Hd = c.width, c.height
        nb, nd = self._neighbours()
        self._nb = nb
        A = self.attractiveness()
        reg = (self.opinion * self.alive[:, :, None]).sum(1) / np.maximum(self.alive.sum(1, keepdims=True) - 1, 1)
        nkids = np.zeros((M, S))
        young = np.full((M, S), -1)
        for s in range(S):
            kids = self._kids(s)
            nkids[:, s] = kids.sum(1)
            ya = np.where(kids, self.age, 1e9)
            young[:, s] = np.where(kids.any(1), ya.argmin(1), -1)
        self._young = young                         # reused by the go_child intention
        own = np.concatenate([np.stack([
            self.sex.astype(float), self.age / 70.0, (self.age < c.child_age) * 1.0, self.hunger, self.health,
            np.minimum(self.money, 100) / 50, self.carry * 1.0, self.pregnant / c.gestation,
            (self.partner >= 0) * 1.0, (self.home >= 0) * 1.0, nkids / 4, self.enlisted * 1.0, (self.detained > 0) * 1.0,
            self.price / 10, reg, np.minimum(self.last_ship, 20)[:, None].repeat(S, 1) / 20], -1),
            self.food / 4.0, self.nutr, self.skill / c.skill_max, self.traits], -1)
        hbox = np.array(c.homes)
        hrel = np.zeros((M, S, 2))
        has = self.home >= 0
        if has.any():
            hb = hbox[np.clip(self.home, 0, self.H - 1)]
            tx = np.clip(self.pos[..., 0], hb[..., 0], hb[..., 2])
            ty = np.clip(self.pos[..., 1], hb[..., 1], hb[..., 3])
            hrel = np.where(has[..., None], np.stack([tx - self.pos[..., 0], ty - self.pos[..., 1]], -1), 0)
        ready = self.site_t < 0
        zones = np.concatenate([
            self._rel(c.shelves), self._rel(c.dock), self._rel(c.market), self._rel(c.station),
            self._rel_site(ready & (self.site_type == GRAIN)), self._rel_site(ready & (self.site_type == FRUIT)),
            self._rel_site(ready & (self.site_type == DAIRY)), self._rel_site((self.site_t == 0) & (self.site_type == GRAIN)),
            hrel], -1) / np.tile([Wd, Hd], 9)
        ar = np.arange(M)[:, None]
        fam = []
        for who in (np.where(self.partner >= 0, self.partner, -1), young):
            ok = who >= 0
            w = np.clip(who, 0, S - 1)
            d = (self.pos[ar, w] - self.pos) / [Wd, Hd]
            fam += [np.where(ok[..., None], d, 0), np.where(ok, self.hunger[ar, w], 0)[..., None], ok[..., None] * 1.0]
        fam = np.concatenate(fam, -1)
        nbf = []
        si = np.arange(S)[None, :]
        for k in range(K):
            j = nb[:, :, k]
            ok = j >= 0
            jj = np.clip(j, 0, S - 1)
            rel = (self.pos[ar, jj] - self.pos) / [Wd, Hd]
            adj = (nd[:, :, k] <= 1) & ok
            partner = (self.partner == jj) & ok
            mine = ((self.mother[ar, jj] == self.pid) | (self.father[ar, jj] == self.pid)) & ok
            parent = ((self.mother == self.pid[ar, jj]) | (self.father == self.pid[ar, jj])) & ok
            sib = (((self.mother == self.mother[ar, jj]) & (self.mother > 0)) |
                   ((self.father == self.father[ar, jj]) & (self.father > 0))) & ok & ~mine & ~parent
            f = np.concatenate([np.stack([
                rel[..., 0], rel[..., 1], adj * 1.0, ok * 1.0, self.sex[ar, jj] * 1.0, self.age[ar, jj] / 70,
                (self.age[ar, jj] < c.child_age) * 1.0, partner * 1.0, mine * 1.0, parent * 1.0, sib * 1.0,
                self.opinion[ar, si, jj], A[ar, si, jj], np.minimum(self.money[ar, jj], 100) / 50,
                self.carry[ar, jj] * 1.0, self.enlisted[ar, jj] * 1.0, (self.detained[ar, jj] > 0) * 1.0,
                self.saw_theft[ar, si, jj] / c.memory_steps, self.robbed_by[ar, si, jj] / c.memory_steps,
                self.shun[ar, si, jj] * 1.0, self.shun[ar, jj, si] * 1.0, (self.court[ar, jj, si] > 0) * 1.0,
                self.price[ar, jj] / 10, self.hunger[ar, jj], (self.partner[ar, jj] >= 0) * 1.0], -1),
                (self.food[ar, jj] > 0) * 1.0], -1) * ok[..., None]
            nbf.append(f)
        obs = np.concatenate([own, zones, fam] + nbf, -1).astype(np.float32)
        alive_n = self.alive.sum(1, keepdims=True)
        g = np.concatenate([alive_n / S, (self.hunger * self.alive).sum(1, keepdims=True) / np.maximum(alive_n, 1),
                            (self.food.sum(-1) * self.alive).sum(1, keepdims=True) / 40,
                            np.minimum((self.money * self.alive).sum(1, keepdims=True), 500) / 250,
                            (self.pregnant > 0).sum(1, keepdims=True) / 4, (self.t / self.max_steps)[:, None]], -1)
        doing = (self.intent[..., None] == np.arange(10, 10 + len(INTENTS))) * 1.0
        cobs = np.concatenate([obs, doing, np.broadcast_to(g[:, None], (M, S, g.shape[1]))], -1).astype(np.float32)
        return obs.reshape(self.N, -1), cobs.reshape(self.N, -1), self.alive.reshape(-1).copy()

    def observe_all(self):
        o, co, _ = self.observe()
        return o, co

    def alive_mask(self):
        return self.alive.reshape(-1).astype(np.float32)

    def decision_mask(self):
        """Who is choosing this step: alive and not in the middle of carrying out an intention."""
        return (self.alive & (self.intent < 0)).reshape(-1).astype(np.float32)

    def _intentions(self, a):
        """One step toward each intention's target; which work intentions have arrived."""
        c, M, S = self.cfg, self.M, self.S
        ready = self.site_t < 0
        rel = np.zeros((M, S, 2), int)
        has = np.zeros((M, S), bool)
        work = np.zeros((M, S), bool)
        ar = np.arange(M)[:, None]

        def put(which, r, ok, is_work=False):
            nonlocal rel, has, work
            sel = (a == which) & ok
            rel = np.where(sel[..., None], r, rel)
            has |= sel
            work |= sel & is_work

        def sites(mask):
            r = self._rel_site(mask)
            d = self.sites[None, None] - self.pos[:, :, None]
            ok = (np.where(mask[:, None], np.abs(d).sum(-1), 1e9) < 1e9).any(-1)
            return r.astype(int), ok

        for which, mask in ((10, ready & (self.site_type == GRAIN)), (11, ready & (self.site_type == FRUIT)),
                            (12, ready & (self.site_type == DAIRY)), (13, (self.site_t == 0) & (self.site_type == GRAIN))):
            r, ok = sites(mask)
            put(which, r, ok, True)
        put(14, np.where(self.carry[..., None], self._rel(c.dock), self._rel(c.shelves)).astype(int), np.ones((M, S), bool), True)
        hb = np.array(c.homes)[np.clip(self.home, 0, self.H - 1)]
        hr = np.stack([np.clip(self.pos[..., 0], hb[..., 0], hb[..., 2]) - self.pos[..., 0],
                       np.clip(self.pos[..., 1], hb[..., 1], hb[..., 3]) - self.pos[..., 1]], -1)
        put(15, hr, self.home >= 0)
        put(16, self._rel(c.market).astype(int), np.ones((M, S), bool))
        pa = np.clip(self.partner, 0, S - 1)
        # adults go to their partner; children go to their mother (or father)
        mom, dad = self._parent_slots()
        par = np.where(mom >= 0, mom, dad)
        kid = self.age < c.adult_age
        tgt = np.where(kid, np.clip(par, 0, S - 1), pa)
        ok = np.where(kid, par >= 0, (self.partner >= 0) & self.alive[ar, pa])
        put(17, self.pos[ar, tgt] - self.pos, ok)
        young = self._young
        yk = np.clip(young, 0, S - 1)
        put(18, self.pos[ar, yk] - self.pos, young >= 0)
        put(19, self._rel(c.station).astype(int), np.ones((M, S), bool))
        at = has & (np.abs(rel).sum(-1) == 0)
        dx, dy = rel[..., 0], rel[..., 1]
        horiz = np.abs(dx) >= np.abs(dy)
        step_vec = np.stack([np.where(horiz, np.sign(dx), 0), np.where(horiz, 0, np.sign(dy))], -1) * has[..., None]
        return step_vec.astype(int), work & at, has, at

    # ------------------------------------------------------------------- step
    def step(self, actions):
        c, M, S, r = self.cfg, self.M, self.S, self.rng
        a = np.asarray(actions).reshape(M, S).astype(int)
        was_alive = self.alive.copy()
        rew = np.zeros((M, S))
        ev = [] if self.record else None
        nb = self._nb
        child = self.age < c.child_age
        adult = self.age >= c.adult_age
        free = self.alive & (self.detained == 0)
        a = np.where(free, a, 0)
        base_ok = np.isin(a, [0, 1, 2, 3, 4, 6, 15, 16, 17])      # children can also walk home, to the market or to a parent
        talk_ok = (a >= N_BASE) & (((a - N_BASE) % N_PER) == TALK)
        a = np.where(child & ~(base_ok | talk_ok), 0, a)
        # someone carrying out an intention keeps at it until it is done; only then do they choose again
        committed = (self.intent >= 0) & self.alive
        a = np.where(committed, self.intent, a)
        step_vec, arrived_work, has, at = self._intentions(a)
        is_intent = (a >= 10) & (a < N_BASE)
        self.intent_t = np.where(committed, self.intent_t + 1, 0)
        keep = is_intent & has & ~at & (self.intent_t < c.intent_patience)
        self.intent = np.where(keep, a, -1)
        a = np.where(arrived_work, 5, a)                           # at the work site: do the work
        # movement (children and the starving move every other step)
        slow = (child | (self.hunger >= 1)) & ((self.t[:, None] % 2) == 1)
        mv = MOVES[np.minimum(a, len(MOVES) - 1)] + step_vec
        self.pos = np.clip(self.pos + mv * (~slow)[..., None] * self.alive[..., None], 0, [c.width - 1, c.height - 1])
        self.last_ship += 1
        # vectorised simple actions: eat, price, enlist
        self._eat(a == 6, ev)
        self.price = np.clip(self.price + 0.5 * ((a == 7) * 1 - (a == 8)), 0.5, 20.0)
        enl = (a == 9) & adult & _in(self.pos, c.station)
        if enl.any():
            self.enlisted ^= enl
            if ev is not None:
                for m, i in zip(*np.nonzero(enl)):
                    ev.append((m, "enlisted" if self.enlisted[m, i] else "resigned", i, -1, 0))
        # the rest needs per-agent resolution, in random order
        busy = (a == 5) | (a >= N_BASE)
        if busy.any():
            d_all = np.abs(self.pos[:, :, None] - self.pos[:, None, :]).max(-1)
            for m in np.nonzero(busy.any(1))[0]:
                for i in r.permutation(S):
                    if not busy[m, i] or not self.alive[m, i] or self.detained[m, i] > 0:
                        continue
                    ai = a[m, i]
                    if ai == 5:
                        self._interact(m, i, rew, ev)
                        continue
                    k, act = divmod(ai - N_BASE, N_PER)
                    j = nb[m, i, k]
                    if j < 0 or not self.alive[m, j] or d_all[m, i, j] > 1:
                        continue
                    self._social(m, i, j, act, rew, ev, d_all)
        self._lifecycle(rew, ev)
        # happiness that accrues every step
        al = self.alive
        variety = (self.nutr > 0.2).sum(-1) / 3.0
        rew += c.w_alive * self.health * al
        rew -= c.w_hunger * self.hunger * al
        rew -= c.w_starving * (self.hunger >= 1) * al
        rew -= c.w_health * (1 - self.health) * al
        rew += c.w_variety * variety * al
        craving = np.clip((c.crave_level - self.nutr) / c.crave_level, 0, 1).sum(-1)
        rew -= c.w_craving * craving * al
        rew -= c.w_detained * (self.detained > 0) * (1 - 0.5 * self.traits[..., 2]) * al
        mutual = (self.opinion > 0.3) & (np.transpose(self.opinion, (0, 2, 1)) > 0.3) & al[:, None, :]
        rew += c.w_friends * np.minimum(mutual.sum(2), 5) * (0.5 + self.traits[..., 0]) * al
        rew += c.w_partner * (self.partner >= 0) * al
        pa = np.clip(self.partner, 0, S - 1)
        ar_ = np.arange(M)[:, None]
        near = (self.partner >= 0) & self.alive[ar_, pa] & (np.abs(self.pos[ar_, pa] - self.pos).max(-1) <= 1)
        home_both = near & self.at_home() & self.at_home()[ar_, pa]
        rew += c.w_together * near * (1 + home_both) * al
        reg = (self.opinion * al[:, :, None]).sum(1) / np.maximum(al.sum(1, keepdims=True) - 1, 1)
        rew += c.w_esteem * reg * al
        rew += c.w_security * np.log1p(np.maximum(self.money, 0)) * (0.5 + self.traits[..., 1]) * al
        kid_w = np.zeros((M, S))
        for s in range(S):
            kids = self._kids(s)
            n = kids.sum(1)
            kid_w[:, s] = np.where(n > 0, ((1 - self.hunger) * kids).sum(1) / np.maximum(n, 1), 0)
        rew += c.w_children * kid_w * (0.5 + self.traits[..., 0]) * al
        rew = np.where(was_alive, rew, 0.0)
        self.ep["person_steps"] += al.sum(1)
        self.ep["happiness"] += rew.sum(1)
        self.ep["variety"] += (variety * al).sum(1)
        # clocks
        for arr in (self.saw_theft, self.robbed_by, self.court):
            np.maximum(arr - 1, 0, out=arr)
        self.detained = np.maximum(self.detained - 1, 0)
        wild = (self.site_t == 0) & (self.site_type == GRAIN) & (r.random(self.site_t.shape) < c.wild_seed)
        self.site_t[wild] = c.grow_steps
        grow = self.site_t > 0
        self.site_t[grow] -= 1
        self.site_t[grow & (self.site_t == 0)] = -1
        self.t += 1
        died = was_alive & ~self.alive
        done_w = (self.t >= self.max_steps) | ~self.alive.any(1)        # end of the era, or extinction
        done = (died | done_w[:, None]).reshape(-1)
        info = dict(timeout=np.repeat(self.t >= self.max_steps, S) & was_alive.reshape(-1),
                    fallen=died.reshape(-1), mask=was_alive.reshape(-1), events=ev,
                    terms={"happiness": rew.reshape(-1)})
        if done_w.any():
            for m in np.nonzero(done_w)[0]:
                st = {k: float(v[m]) for k, v in self.ep.items()}
                st["population"] = float(self.alive[m].sum())
                st["years"] = float(self.t[m] / YEAR)
                st["mean_traits"] = self.traits[m][self.alive[m]].mean(0).round(3).tolist() if self.alive[m].any() else [0, 0, 0, 0]
                self.stats.append(st)
            self._reset(np.nonzero(done_w)[0])
        obs, cobs, _ = self.observe()
        return obs, cobs, rew.reshape(-1).astype(np.float32), done, info

    # ------------------------------------------------------------ actions
    def _eat(self, want, ev):
        c = self.cfg
        has = self.food > 0
        eat = want & has.any(-1)
        if not eat.any():
            return
        # eat what your body lacks most among what you hold
        pick = np.where(has, self.nutr, 9.0).argmin(-1)
        m, s = np.nonzero(eat)
        t = pick[m, s]
        self.food[m, s, t] -= 1
        relief = c.food_relief * (c.satiety_floor + (1 - c.satiety_floor) * (1 - self.nutr[m, s, t]))
        self.hunger[m, s] = np.maximum(0.0, self.hunger[m, s] - relief)
        self.nutr[m, s, t] = np.minimum(1.0, self.nutr[m, s, t] + c.nutrient_gain)
        if ev is not None:
            for mm, ss, tt in zip(m, s, t):
                ev.append((mm, "ate " + FOODS[tt], ss, -1, 0))

    def _interact(self, m, i, rew, ev):
        c = self.cfg
        p = self.pos[m, i]
        if not self.carry[m, i] and _in(p, c.shelves):
            self.carry[m, i] = True
            return
        if self.carry[m, i] and _in(p, c.dock):
            self.carry[m, i] = False
            team = self.last_ship[m] <= c.team_window
            pay = c.crate_pay * (1 + c.team_bonus * team)
            self.money[m, i] += pay
            self.last_ship[m] = 0
            rew[m, i] += c.w_purpose
            self.ep["crates"][m] += 1
            if ev is not None:
                ev.append((m, "shipped", i, -1, round(pay, 1)))
            return
        on = (self.sites == p).all(1)
        if not on.any():
            return
        k = int(on.argmax())
        ft = self.site_type[k]
        st = self.site_t[m, k]
        if ft == GRAIN and st == 0 and self.food[m, i, GRAIN] >= c.seed_cost:
            self.site_t[m, k] = c.grow_steps
            self.food[m, i, GRAIN] -= c.seed_cost
            rew[m, i] += c.w_purpose * 0.3
            if ev is not None:
                ev.append((m, "planted", i, -1, c.seed_cost))
        elif st == -1 and self.food[m, i].sum() < c.max_food:
            # neighbours working the same kind of site help (and get nothing): a larger harvest
            same = self.alive[m] & (self.age[m] >= c.adult_age) & (np.abs(self.pos[m] - p).max(1) <= 1)
            same[i] = False
            on_site = np.array([((self.sites[self.site_type == ft] == self.pos[m, j]).all(1)).any() if same[j] else False
                                for j in range(self.S)])
            helpers = int(on_site.sum())
            got = int(c.base_yield[ft] + np.floor(self.skill[m, i, ft]) + helpers)
            got = min(got, c.max_food - int(self.food[m, i].sum()))
            self.food[m, i, ft] += got
            self.site_t[m, k] = 0 if ft == GRAIN else self.regrow[k]
            self.skill[m, i, ft] = min(c.skill_max, self.skill[m, i, ft] + c.skill_gain)
            rew[m, i] += c.w_purpose
            self.ep["harvest_" + FOODS[ft]][m] += 1
            if ev is not None:
                ev.append((m, "harvested " + FOODS[ft], i, -1, got))

    def _social(self, m, i, j, act, rew, ev, d_all):
        c = self.cfg
        refused = self.shun[m, j, i]
        O = self.opinion
        if act == GIVE_FOOD and self.food[m, i].sum() > 0 and not refused and self.food[m, j].sum() < c.max_food:
            # give what the receiver lacks most, of what you have; gratitude grows with their need
            t = int(np.where(self.food[m, i] > 0, self.nutr[m, j], 9).argmin())
            need = 0.5 * (1 - self.nutr[m, j, t]) + 0.5 * self.hunger[m, j]
            self.food[m, i, t] -= 1
            self.food[m, j, t] += 1
            O[m, j, i] = min(1, O[m, j, i] + 0.15 * (0.2 + 0.8 * need))
            self._witness_good(m, i, 0.03)
            kid = self.mother[m, j] == self.pid[m, i] or self.father[m, j] == self.pid[m, i]
            if kid and self.age[m, j] < c.adult_age:
                rew[m, i] += c.w_purpose * 0.5
            self.ep["gifts_food"][m] += 1
            if ev is not None:
                ev.append((m, "gave " + FOODS[t], i, j, 1))
        elif act == GIVE_COIN and self.money[m, i] >= 1 and not refused:
            self.money[m, i] -= 1
            self.money[m, j] += 1
            O[m, j, i] = min(1, O[m, j, i] + 0.1)
            self._witness_good(m, i, 0.02)
            if self.enlisted[m, j]:
                self.ep["police_pay"][m] += 1
            self.ep["gifts_coin"][m] += 1
            if ev is not None:
                ev.append((m, "gave coin", i, j, 1))
        elif act == BUY and not refused and not self.shun[m, i, j] and self.food[m, j].sum() > 0 \
                and self.money[m, i] >= self.price[m, j] and self.food[m, i].sum() < c.max_food:
            # buy the food you lack most, of those the seller has
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
            self.robbed_by[m, j, i] = c.memory_steps
            rew[m, j] -= c.w_robbed
            wit = self.alive[m] & (np.abs(self.pos[m] - self.pos[m, i]).max(1) <= c.witness_radius)
            wit[i] = False
            self.saw_theft[m, wit, i] = c.memory_steps
            O[m, wit, i] = np.maximum(-1, O[m, wit, i] - 0.3)
            self.ep["thefts"][m] += 1
            if ev is not None:
                ev.append((m, "theft", i, j, amt))
        elif act == TALK and not refused and not self.shun[m, i, j]:
            for (x, y) in ((i, j), (j, i)):          # both share what they know about third parties
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
        elif act == COURT and not refused and self.age[m, i] >= c.adult_age and self.age[m, j] >= c.adult_age \
                and self.sex[m, i] != self.sex[m, j] and self.partner[m, i] < 0 and self.partner[m, j] < 0 \
                and not self.related(m, i, j):
            self.court[m, i, j] = c.court_window
            if ev is not None:
                ev.append((m, "courted", i, j, 0))
            if self.court[m, j, i] > 0:
                A = self.attractiveness(m)
                if A[i, j] >= c.attract_threshold and A[j, i] >= c.attract_threshold:
                    self.partner[m, i], self.partner[m, j] = j, i
                    used = set(self.home[m][self.alive[m]].tolist())
                    freeh = [h for h in range(self.H) if h not in used]
                    if freeh:
                        self.home[m, i] = self.home[m, j] = freeh[0]
                    elif self.home[m, i] >= 0 or self.home[m, j] >= 0:   # no free house: move in with the other's family
                        self.home[m, i] = self.home[m, j] = self.home[m, i] if self.home[m, i] >= 0 else self.home[m, j]
                    O[m, i, j] = min(1, O[m, i, j] + 0.3)
                    O[m, j, i] = min(1, O[m, j, i] + 0.3)
                    self.ep["partnerships"][m] += 1
                    if ev is not None:
                        ev.append((m, "partnered", i, j, 0))
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
            near = self.alive[m] & (d_all[m, j] <= 1) & (self.saw_theft[m, :, j] > 0) & (self.age[m] >= c.adult_age)
            near[j] = False
            citizens = near.sum() >= 2
            if self.enlisted[m, i] or citizens:
                police = self.enlisted[m, i] and not citizens
                takers = [i] if police else list(np.nonzero(near)[0])
                amt = max(0.0, self.money[m, j])
                self.money[m, j] = 0
                for x in takers:
                    self.money[m, x] += amt / len(takers)
                self.detained[m, j] = c.detain_steps
                self.intent[m, j] = -1
                self.pos[m, j] = [c.station[0], c.station[1]]
                self.carry[m, j] = False
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
        wit = self.alive[m] & (np.abs(self.pos[m] - self.pos[m, i]).max(1) <= c.witness_radius)
        wit[i] = False
        self.opinion[m, wit, i] = np.minimum(1, self.opinion[m, wit, i] + amount)

    # ---------------------------------------------------------- life & death
    def _lifecycle(self, rew, ev):
        c, M, S, r = self.cfg, self.M, self.S, self.rng
        al = self.alive
        rate = c.hunger_rate * np.where(self.pregnant > 0, c.pregnant_hunger, 1.0)
        self.hunger = np.where(al, np.minimum(1.0, self.hunger + rate), 0)
        self.nutr = np.maximum(0.0, self.nutr - c.nutrient_decay)
        missing = (self.nutr <= 0).sum(-1)
        at_home = self.at_home()
        self.home_seen = np.where(at_home, 0, np.minimum(self.home_seen + 1, 999))
        heal = c.heal_rate * (1 + c.shelter_heal * at_home)
        self.health = np.where(self.hunger >= 1, self.health - c.starve_damage,
                               np.where((self.hunger < 0.5) & (missing == 0), np.minimum(1, self.health + heal), self.health))
        self.health = self.health - c.deficiency_damage * missing
        self.age += al / YEAR
        # infants nurse when they are with their fed mother
        mom, _ = self._parent_slots()
        ar = np.arange(M)[:, None]
        mm = np.clip(mom, 0, S - 1)
        nursing = al & (self.age < c.nursing_age) & (mom >= 0) & (np.abs(self.pos[ar, mm] - self.pos).max(-1) <= 1) \
            & (self.hunger[ar, mm] < 0.8)
        if nursing.any():
            self.hunger = np.where(nursing, np.maximum(0, self.hunger - c.nurse_relief), self.hunger)
            self.nutr = np.where(nursing[..., None], np.minimum(1, self.nutr + c.nurse_nutrient), self.nutr)
            cost = np.zeros((M, S))
            np.add.at(cost, (np.nonzero(nursing)[0], mm[nursing]), c.nurse_cost)
            self.hunger = np.minimum(1, self.hunger + cost)
        spoil = (self.food > 0) & (r.random((M, S, 3)) < self.food / c.spoil_steps)
        self.food -= spoil
        p_old = np.where(self.age > c.old_age, ((self.age - c.old_age) / 30.0) ** 2 * 0.004, 0)
        dies = al & ((self.health <= 0) | (r.random((M, S)) < p_old))
        for m, s in zip(*np.nonzero(dies)):
            cause = "old age" if self.health[m, s] > 0 else "starved" if self.hunger[m, s] >= 1 else "malnutrition"
            self._die(m, s, rew, ev, cause)
        # conception and birth
        lo, hi = c.fertile_ages
        cand = self.alive & (self.sex == F)
        for m, s in zip(*np.nonzero(cand & (self.pregnant > 0))):
            self.pregnant[m, s] -= 1
            if self.pregnant[m, s] == 0:
                self._birth(m, s, rew, ev)
        ready = cand & (self.pregnant == 0) & (self.partner >= 0) & (self.home >= 0) & (self.age >= lo) \
            & (self.age <= hi) & (self.hunger < 0.6)
        for m, s in zip(*np.nonzero(ready)):
            pa = self.partner[m, s]
            together = np.abs(self.pos[m, pa] - self.pos[m, s]).max() <= 1
            if self.alive[m, pa] and self.hunger[m, pa] < 0.6 and together \
                    and self.home[m, pa] == self.home[m, s] and (~self.alive[m]).any() and r.random() < c.conceive_prob:
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
        h = self.home[m, mother]
        pos = [c.homes[h][0], c.homes[h][1]] if h >= 0 else self.pos[m, mother].tolist()
        self._new_person(m, s, int(r.integers(0, 2)), 0.0, traits, pos, mother=self.pid[m, mother],
                         father=self.pid[m, father] if father >= 0 else 0)
        self.hunger[m, s] = 0.2
        self.home[m, s] = h
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
                rew[m, j] -= c.w_grief * (0.5 + self.traits[m, j, 0])
        p = self.partner[m, s]
        if p >= 0:
            self.partner[m, p] = -1
        self.alive[m, s] = False
        self.carry[m, s] = False
        self.enlisted[m, s] = False
        self.pregnant[m, s] = 0
        self.home[m, s] = -1
        self.ep["deaths"][m] += 1
        self.ep["starved"][m] += cause == "starved"
        self.ep["malnourished"][m] += cause == "malnutrition"
        if ev is not None:
            ev.append((m, f"died ({cause})", s, -1, round(float(self.age[m, s]), 1)))

    # ------------------------------------------------------------- replay
    def snapshot(self, m=0):
        al = self.alive[m]
        return dict(
            t=int(self.t[m]), alive=al.astype(int).tolist(), pos=self.pos[m].tolist(), sex=self.sex[m].tolist(),
            age=np.round(self.age[m], 1).tolist(), hunger=np.round(self.hunger[m], 2).tolist(),
            health=np.round(self.health[m], 2).tolist(), money=np.round(self.money[m], 1).tolist(),
            food=self.food[m].tolist(), nutr=np.round(self.nutr[m], 2).tolist(), skill=np.round(self.skill[m], 2).tolist(),
            carry=self.carry[m].astype(int).tolist(), pregnant=(self.pregnant[m] > 0).astype(int).tolist(),
            partner=self.partner[m].tolist(), home=self.home[m].tolist(), enlisted=self.enlisted[m].astype(int).tolist(),
            detained=(self.detained[m] > 0).astype(int).tolist(),
            seen_thief=(self.saw_theft[m].max(0) > 0).astype(int).tolist(),
            regard=np.round((self.opinion[m] * al[:, None]).sum(0) / max(al.sum() - 1, 1), 2).tolist(),
            pid=self.pid[m].tolist(), mother=self.mother[m].tolist(), father=self.father[m].tolist(),
            traits=np.round(self.traits[m], 2).tolist(), sites=self.site_t[m].tolist(),
        )
