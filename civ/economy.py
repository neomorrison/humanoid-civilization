"""A small economy the agents have to discover: jobs, trade, theft, policing.

Design rule: **no automated systems.** Every transfer of money or goods between
agents is an action one of them chooses:

* ``GIVE_j``   hand one coin to agent j (standing next to you). Paying the
               police, tipping, bribing - whatever it turns out to be.
* ``TAKE_j``   grab cash from agent j. For a police officer taking from someone
               they *saw* stealing, it is an arrest: the thief is detained and
               their cash confiscated - to the officer, who may or may not give
               it back to the victim.
* ``BUY``      buy one food from the cheapest adjacent seller at the price the
               seller set (a handshake: both are there, both chose to be).
* ``PRICE+/-`` change your own asking price for food.

Nothing is deducted or paid automatically. The only money entering the world is
from shipping boxes out at the loading dock (the outside world buys them); the
employer as an agent is the next step.

    shelves ──carry box──▶ loading dock        workers are paid per box shipped
    farm plots: plant ─▶ grow ─▶ harvest      the farmer grows food (seed costs money)
    anyone with food can sell it; anyone can eat; hunger never stops

Agents see what a person could know: positions and roles of the others, their
visible state, and their own memories of recent events (who they saw stealing,
who stole from them, who paid them, whom they paid).

Rewards per agent: staying fed (hunger costs; eating relieves it) plus a small
value on coins gained. Everything is a knob in ``EconomyConfig``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

WORKER, FARMER, POLICE = 0, 1, 2
ROLE_NAMES = ["worker", "farmer", "police"]


@dataclass
class EconomyConfig:
    width: int = 20
    height: int = 14
    roles: tuple = (WORKER, WORKER, WORKER, WORKER, FARMER, POLICE)
    episode_steps: int = 600
    box_value: float = 5.0        # paid by the outside world per box shipped
    seed_cost: float = 1.0
    grow_steps: int = 60
    harvest_food: int = 2
    max_food: int = 6
    price0: float = 4.0           # initial asking price
    price_step: float = 0.5
    price_min: float = 0.5
    price_max: float = 20.0
    hunger_rate: float = 1.0 / 300.0
    food_relief: float = 0.5
    theft_enabled: bool = True
    steal_frac: float = 0.5
    steal_max: float = 10.0
    witness_radius: int = 5       # Chebyshev distance at which a theft is seen
    memory_steps: int = 60        # how long events are remembered
    detain_steps: int = 60
    money_value: float = 0.05
    hunger_cost: float = 0.01
    starve_cost: float = 0.05
    eat_bonus: float = 0.5
    start_money: float = 5.0
    farmer_start_food: int = 4    # something to trade before the first harvest
    shelves: tuple = (1, 9, 2, 12)
    dock: tuple = (7, 9, 8, 12)
    farm: tuple = (13, 9, 18, 11)
    market: tuple = (9, 3, 11, 5)
    station: tuple = (1, 1, 2, 2)


def _in(pos, box):
    x0, y0, x1, y1 = box
    return (pos[..., 0] >= x0) & (pos[..., 0] <= x1) & (pos[..., 1] >= y0) & (pos[..., 1] <= y1)


class Economy:
    def __init__(self, n_worlds: int, cfg: EconomyConfig | None = None, seed: int = 0, record: bool = False):
        self.cfg = c = cfg or EconomyConfig()
        self.M, self.A = n_worlds, len(c.roles)
        A = self.A
        self.N = self.M * A
        self.rng = np.random.default_rng(seed)
        self.role = np.tile(np.array(c.roles), (self.M, 1))
        x0, y0, x1, y1 = c.farm
        self.plots = np.array([[x, y] for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)])
        self.P = len(self.plots)
        self.others = np.array([[j for j in range(A) if j != i] for i in range(A)])   # (A, A-1)
        # action layout
        self.STAY, self.UP, self.DOWN, self.LEFT, self.RIGHT, self.INTERACT, self.EAT, self.BUY, \
            self.PRICE_UP, self.PRICE_DOWN = range(10)
        self.GIVE0 = 10
        self.TAKE0 = 10 + (A - 1)
        self.act_dim = 10 + 2 * (A - 1)
        self.moves = np.zeros((self.act_dim, 2), int)
        self.moves[[1, 2, 3, 4]] = [[0, 1], [0, -1], [-1, 0], [1, 0]]
        self.record = record
        self.stats = []
        self._alloc()
        self.reset_all()
        o, co = self.observe()
        self.obs_dim, self.cobs_dim = o.shape[1], co.shape[1]

    def action_names(self):
        n = ["stay", "up", "down", "left", "right", "interact", "eat", "buy", "price+", "price-"]
        return n + [f"give>{k}" for k in range(self.A - 1)] + [f"take>{k}" for k in range(self.A - 1)]

    # ------------------------------------------------------------------ state
    def _alloc(self):
        M, A = self.M, self.A
        self.pos = np.zeros((M, A, 2), int)
        self.money = np.zeros((M, A))
        self.hunger = np.zeros((M, A))
        self.carry = np.zeros((M, A), bool)
        self.food = np.zeros((M, A), int)
        self.price = np.zeros((M, A))
        self.detained = np.zeros((M, A), int)
        # memories (observer, subject): steps left
        self.saw_steal = np.zeros((M, A, A), int)     # observer saw subject steal
        self.robbed_by = np.zeros((M, A, A), int)     # observer was robbed by subject
        self.paid_me = np.zeros((M, A, A))            # coins subject paid observer (decaying)
        self.i_paid = np.zeros((M, A, A))             # coins observer paid subject (decaying)
        self.plot_t = np.zeros((M, self.P), int)      # 0 empty, >0 growing, -1 ripe
        self.t = np.zeros(M, int)
        self.ep = {k: np.zeros(M) for k in ["boxes", "food_sold", "sales_value", "thefts", "stolen", "arrests",
                                            "returned", "gifts", "gifts_to_police", "eaten", "starving",
                                            "harvests"]}

    def reset_all(self):
        self._reset(np.arange(self.M))

    def _reset(self, w):
        c, r, n, A = self.cfg, self.rng, len(w), self.A
        self.pos[w, :, 0] = r.integers(0, c.width, (n, A))
        self.pos[w, :, 1] = r.integers(0, c.height, (n, A))
        self.money[w] = c.start_money
        self.hunger[w] = r.uniform(0.2, 0.5, (n, A))
        self.carry[w] = False
        self.food[w] = np.where(self.role[w] == FARMER, c.farmer_start_food, 0)
        self.price[w] = c.price0
        self.detained[w] = 0
        for arr in (self.saw_steal, self.robbed_by, self.paid_me, self.i_paid):
            arr[w] = 0
        self.plot_t[w] = 0
        self.t[w] = 0
        for k in self.ep:
            self.ep[k][w] = 0

    # ----------------------------------------------------------- observation
    def _nearest(self, box):
        x0, y0, x1, y1 = box
        tx = np.clip(self.pos[..., 0], x0, x1)
        ty = np.clip(self.pos[..., 1], y0, y1)
        return np.stack([tx - self.pos[..., 0], ty - self.pos[..., 1]], -1)

    def _nearest_plot(self, mask):
        d = self.plots[None, None] - self.pos[:, :, None]
        dist = np.abs(d).sum(-1).astype(float)
        dist = np.where(mask[:, None], dist, 1e9)
        k = dist.argmin(-1)
        out = np.take_along_axis(d, k[..., None, None], axis=2)[:, :, 0]
        return np.where((dist.min(-1) < 1e9)[..., None], out, 0)

    def observe(self):
        c, M, A = self.cfg, self.M, self.A
        W, H, mem = c.width, c.height, c.memory_steps
        own = np.concatenate([
            np.eye(3)[self.role], self.pos / [W, H], np.minimum(self.money, 100)[..., None] / 50,
            self.hunger[..., None], self.carry[..., None], self.food[..., None] / 3, self.price[..., None] / 10,
            (self.detained > 0)[..., None], np.broadcast_to((self.t / c.episode_steps)[:, None, None], (M, A, 1))], -1)
        rel = np.concatenate([self._nearest(c.shelves), self._nearest(c.dock), self._nearest(c.market),
                              self._nearest(c.station), self._nearest_plot(self.plot_t < 0),
                              self._nearest_plot(self.plot_t == 0)], -1) / np.tile([W, H], 6)
        oth = []
        for i in range(A):
            idx = self.others[i]
            d = self.pos[:, idx] - self.pos[:, i:i + 1]
            adj = np.abs(d).max(-1) <= 1
            oth.append(np.concatenate([
                d / [W, H], adj[..., None], np.eye(3)[self.role[:, idx]],
                np.minimum(self.money[:, idx], 100)[..., None] / 50, self.carry[:, idx][..., None],
                (self.food[:, idx] > 0)[..., None], self.price[:, idx][..., None] / 10,
                (self.detained[:, idx] > 0)[..., None],
                self.saw_steal[:, i, idx][..., None] / mem, self.robbed_by[:, i, idx][..., None] / mem,
                np.minimum(self.paid_me[:, i, idx], 10)[..., None] / 5,
                np.minimum(self.i_paid[:, i, idx], 10)[..., None] / 5], -1).reshape(M, -1))
        oth = np.stack(oth, 1)
        obs = np.concatenate([own, rel, oth], -1).astype(np.float32)
        g = np.concatenate([np.minimum(self.money, 100) / 50, self.hunger, self.food / 3,
                            (self.plot_t < 0).mean(1, keepdims=True), (self.plot_t > 0).mean(1, keepdims=True)], -1)
        cobs = np.concatenate([obs, np.broadcast_to(g[:, None], (M, A, g.shape[-1]))], -1).astype(np.float32)
        return obs.reshape(self.N, -1), cobs.reshape(self.N, -1)

    def observe_all(self):
        return self.observe()

    # ------------------------------------------------------------------- step
    def step(self, actions):
        c, M, A, r = self.cfg, self.M, self.A, self.rng
        a = np.asarray(actions).reshape(M, A).astype(int)
        money0 = self.money.copy()
        rew = np.zeros((M, A))
        ev = [] if self.record else None
        a = np.where(self.detained == 0, a, self.STAY)
        slow = (self.hunger >= 1.0) & ((self.t[:, None] % 2) == 1)
        self.pos = np.clip(self.pos + self.moves[a] * (~slow)[..., None], 0, [c.width - 1, c.height - 1])
        ar = np.arange(M)
        for i in r.permutation(A):
            ai, pi, ri = a[:, i], self.pos[:, i], self.role[:, i]
            idx = self.others[i]
            dist = np.abs(self.pos - pi[:, None]).max(-1)                       # Chebyshev, (M, A)
            adj = (dist <= 1) & (np.arange(A) != i)[None] & (self.detained == 0)
            inter = ai == self.INTERACT
            # ---- workers: carry boxes from the shelves to the dock (shipped = paid)
            pick = inter & (ri == WORKER) & ~self.carry[:, i] & _in(pi, c.shelves)
            self.carry[pick, i] = True
            ship = inter & (ri == WORKER) & self.carry[:, i] & _in(pi, c.dock)
            self.carry[ship, i] = False
            self.money[ship, i] += c.box_value
            self.ep["boxes"][ship] += 1
            # ---- farmer: plant / harvest the plot under them
            on_plot = (self.plots[None] == pi[:, None]).all(-1)
            k = on_plot.argmax(1)
            here = on_plot.any(1) & inter & (ri == FARMER)
            pt = self.plot_t[ar, k]
            plant = here & (pt == 0) & (self.money[:, i] >= c.seed_cost)
            self.plot_t[ar[plant], k[plant]] = c.grow_steps
            self.money[plant, i] -= c.seed_cost
            harvest = here & (pt == -1) & (self.food[:, i] < c.max_food)
            self.plot_t[ar[harvest], k[harvest]] = 0
            self.food[harvest, i] = np.minimum(c.max_food, self.food[harvest, i] + c.harvest_food)
            self.ep["harvests"][harvest] += 1
            # ---- asking price
            self.price[:, i] = np.clip(self.price[:, i] + c.price_step * ((ai == self.PRICE_UP) * 1 - (ai == self.PRICE_DOWN)),
                                       c.price_min, c.price_max)
            # ---- buy from the cheapest adjacent seller (handshake at the seller's price)
            sellers = adj & (self.food > 0)
            buy = (ai == self.BUY) & sellers.any(1) & (self.food[:, i] < 3)
            if buy.any():
                pr = np.where(sellers, self.price, np.inf)
                s = pr.argmin(1)
                p = pr[ar, s]
                ok = buy & (self.money[:, i] >= p)
                for m in np.nonzero(ok)[0]:
                    sm = s[m]
                    self.money[m, i] -= p[m]
                    self.money[m, sm] += p[m]
                    self.food[m, sm] -= 1
                    self.food[m, i] += 1
                    self.ep["food_sold"][m] += 1
                    self.ep["sales_value"][m] += p[m]
                    if ev is not None:
                        ev.append((m, "bought food", i, int(sm), round(float(p[m]), 1)))
            # ---- eat
            eat = (ai == self.EAT) & (self.food[:, i] > 0)
            relief = np.minimum(self.hunger[:, i], c.food_relief) * eat
            self.food[eat, i] -= 1
            self.hunger[:, i] -= relief
            rew[:, i] += c.eat_bonus * relief
            self.ep["eaten"][eat] += 1
            # ---- give one coin to a chosen neighbour
            g = ai - self.GIVE0
            give = (g >= 0) & (g < A - 1)
            if give.any():
                tgt = idx[np.clip(g, 0, A - 2)]
                ok = give & adj[ar, tgt] & (self.money[:, i] >= 1)
                for m in np.nonzero(ok)[0]:
                    j = tgt[m]
                    self.money[m, i] -= 1
                    self.money[m, j] += 1
                    self.paid_me[m, j, i] += 1
                    self.i_paid[m, i, j] += 1
                    self.ep["gifts"][m] += 1
                    if self.role[m, j] == POLICE:
                        self.ep["gifts_to_police"][m] += 1
                    # handing back a victim's money counts as a return
                    if self.role[m, i] == POLICE and self.robbed_by[m, j].any():
                        self.ep["returned"][m] += 1
                    if ev is not None:
                        ev.append((m, "gave", i, int(j), 1))
            # ---- take from a chosen neighbour: arrest (police, witnessed thief) or theft
            tk = ai - self.TAKE0
            take = (tk >= 0) & (tk < A - 1)
            if take.any():
                tgt = idx[np.clip(tk, 0, A - 2)]
                ok = take & adj[ar, tgt]
                for m in np.nonzero(ok)[0]:
                    j = tgt[m]
                    if self.role[m, i] == POLICE and self.saw_steal[m, i, j] > 0:
                        amt = max(self.money[m, j], 0.0)
                        self.money[m, j] -= amt
                        self.money[m, i] += amt
                        self.detained[m, j] = c.detain_steps
                        self.carry[m, j] = False
                        self.pos[m, j] = [c.station[0], c.station[1]]
                        self.saw_steal[m, :, j] = 0
                        self.ep["arrests"][m] += 1
                        if ev is not None:
                            ev.append((m, "arrest", i, int(j), round(float(amt), 1)))
                    elif c.theft_enabled:
                        amt = min(c.steal_max, c.steal_frac * max(self.money[m, j], 0.0))
                        if amt <= 0:
                            continue
                        self.money[m, j] -= amt
                        self.money[m, i] += amt
                        self.robbed_by[m, j, i] = c.memory_steps
                        seen = np.abs(self.pos[m] - self.pos[m, i]).max(-1) <= c.witness_radius
                        self.saw_steal[m, seen, i] = c.memory_steps
                        self.saw_steal[m, i, i] = 0
                        self.ep["thefts"][m] += 1
                        self.ep["stolen"][m] += amt
                        if ev is not None:
                            ev.append((m, "theft", i, int(j), round(float(amt), 1)))
            if ev is not None:
                for m in np.nonzero(ship)[0]:
                    ev.append((m, "shipped box", i, -1, c.box_value))
                for m in np.nonzero(eat)[0]:
                    ev.append((m, "ate", i, -1, 0))
                for m in np.nonzero(harvest)[0]:
                    ev.append((m, "harvested", i, -1, c.harvest_food))
                for m in np.nonzero(plant)[0]:
                    ev.append((m, "planted", i, -1, c.seed_cost))
        # clocks and bodies
        grow = self.plot_t > 0
        self.plot_t[grow] -= 1
        self.plot_t[grow & (self.plot_t == 0)] = -1
        self.detained = np.maximum(0, self.detained - 1)
        self.saw_steal = np.maximum(0, self.saw_steal - 1)
        self.robbed_by = np.maximum(0, self.robbed_by - 1)
        decay = 1.0 - 1.0 / c.memory_steps
        self.paid_me *= decay
        self.i_paid *= decay
        self.hunger = np.minimum(1.0, self.hunger + c.hunger_rate)
        starving = self.hunger >= 1.0
        self.ep["starving"] += starving.sum(1)
        rew += c.money_value * (self.money - money0)
        rew -= c.hunger_cost * self.hunger + c.starve_cost * starving
        self.t += 1
        done_w = self.t >= c.episode_steps
        done = np.repeat(done_w, A)
        info = dict(timeout=done.copy(), fallen=np.zeros(self.N, bool), events=ev)
        if done_w.any():
            for m in np.nonzero(done_w)[0]:
                s = {k: float(v[m]) for k, v in self.ep.items()}
                for q in range(3):
                    sel = self.role[m] == q
                    if sel.any():
                        s["money_" + ROLE_NAMES[q]] = float(self.money[m][sel].mean())
                self.stats.append(s)
            self._reset(np.nonzero(done_w)[0])
        obs, cobs = self.observe()
        return obs, cobs, rew.reshape(-1).astype(np.float32), done, info

    def snapshot(self, m=0):
        return dict(t=int(self.t[m]), pos=self.pos[m].tolist(), money=np.round(self.money[m], 1).tolist(),
                    hunger=np.round(self.hunger[m], 2).tolist(), carry=self.carry[m].astype(int).tolist(),
                    food=self.food[m].tolist(), price=np.round(self.price[m], 1).tolist(),
                    detained=(self.detained[m] > 0).astype(int).tolist(),
                    seen_thief=(self.saw_steal[m].max(0) > 0).astype(int).tolist(),
                    plots=self.plot_t[m].tolist())
