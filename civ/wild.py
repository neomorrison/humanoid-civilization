"""Wild: a world where the solutions must be invented (see docs/WILD.md).

Pure JAX: `reset(key, cfg)` and `step(state, actions, key, cfg)` are jittable and are
vmapped over many worlds. Nothing is named: people have bodies to keep in balance,
primitive actions, materials, recipes nobody is told about, and each other.

Items: 0 empty, 1 berries, 2 branch, 3 stone, 4 fibre, 5 sharp stone, 6 rope, 7 cooked food.
Terrain: 0 grass, 1 water, 2 woods, 3 stony ground, 4 berry scrub, 5 tall grass.
Actions: 0 stay, 1-4 move N/S/E/W, 5 grab, 6 drop, 7 eat, 8 drink, 9 combine, 10 give,
         11 sleep, 12 mate, 13.. signal k (k < n_signals).
"""
from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

N_ITEMS = 8
BERRIES, BRANCH, STONE, FIBRE, SHARP, ROPE, COOKED = 1, 2, 3, 4, 5, 6, 7
ITEM_NAMES = ["", "berries", "branch", "stone", "fibre", "sharp stone", "rope", "cooked food"]
GRASS, WATER, WOODS, STONY, SCRUB, TALL = 0, 1, 2, 3, 4, 5
N_TERRAIN = 6
RESOURCE_OF = jnp.array([0, 0, BRANCH, STONE, BERRIES, FIBRE])           # what grabbing on each terrain yields
STAY, GRAB, DROP, EAT, DRINK, COMBINE, GIVE, SLEEP, MATE, SIG0 = 0, 5, 6, 7, 8, 9, 10, 11, 12, 13
MOVES = jnp.array([[0, 0], [0, 1], [0, -1], [1, 0], [-1, 0]])
BODY = ["satiety", "hydration", "warmth", "energy", "health"]
GENES = BODY + ["birth", "kin", "company"]          # inner values: weights on each part of wellbeing
DAY, YEAR = 24, 96


class Config(NamedTuple):
    W: int = 32
    H: int = 32
    S: int = 32                     # person slots
    founders: int = 16
    n_signals: int = 8
    view: int = 3                   # sees (2*view+1)^2 cells
    # body, per hour
    satiety_loss: float = 0.02
    hydration_loss: float = 0.03
    energy_loss: float = 0.045
    sleep_gain: float = 0.11
    starve_damage: float = 0.02     # health lost per hour per critical variable
    heal: float = 0.004
    food_value: tuple = (0.0, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0, 0.45)   # satiety per item eaten
    drink_value: float = 0.5
    fire_warmth: float = 0.25       # per hour next to a fire
    shelter_warmth: float = 0.12
    huddle_warmth: float = 0.05     # per other person on your cell
    fire_hours: int = 12
    # grabbing: chance of success per try, and with a sharp stone in the other hand
    grab_p: tuple = (0.0, 0.8, 0.3, 0.5, 0.35, 0.0, 0.0, 0.0)
    grab_p_tool: tuple = (0.0, 0.8, 0.8, 0.5, 0.9, 0.0, 0.0, 0.0)
    res_max: int = 3
    regrow: tuple = (0.0, 0.0, 0.01, 0.002, 0.03, 0.02)             # per cell per hour, by terrain (summer)
    # life
    child_age: float = 4.0          # younger children cannot gather or drink on their own
    adult_age: float = 14.0
    fertile: tuple = (16.0, 42.0)
    gestation: int = 72
    conceive_p: float = 0.3
    old_age: float = 45.0
    nurse: float = 0.06             # satiety and hydration an infant gets per hour with its mother
    nurse_cost: float = 0.02
    mutation: float = 0.1
    signal_radius: int = 4
    # rewards
    death_penalty: float = 5.0
    birth_bonus: float = 3.0


class State(NamedTuple):
    t: jnp.ndarray            # hour
    terrain: jnp.ndarray      # (W, H) int
    res: jnp.ndarray          # (W, H) resource units left on the cell
    ground: jnp.ndarray       # (W, H, N_ITEMS) items lying on the cell
    fire: jnp.ndarray         # (W, H) hours of fire left
    shelter: jnp.ndarray      # (W, H) shelter pieces
    alive: jnp.ndarray        # (S,) bool
    pos: jnp.ndarray          # (S, 2)
    sex: jnp.ndarray          # (S,)
    age: jnp.ndarray          # (S,) years
    hands: jnp.ndarray        # (S, 2) item ids
    body: jnp.ndarray         # (S, 5) satiety, hydration, warmth, energy, health in [0, 1]
    asleep: jnp.ndarray       # (S,) bool
    preg: jnp.ndarray         # (S,) hours of pregnancy left
    father_of: jnp.ndarray    # (S,) slot of the father of the child being carried
    genes: jnp.ndarray        # (S, G)
    pid: jnp.ndarray          # (S,)
    mother: jnp.ndarray       # (S,) pid
    father: jnp.ndarray       # (S,) pid
    signal: jnp.ndarray       # (S,) token emitted last hour, -1 none
    next_pid: jnp.ndarray
    made: jnp.ndarray         # (N_ITEMS,) how many of each item have ever been made (inventions)
    births: jnp.ndarray
    deaths: jnp.ndarray


def _blobs(key, W, H, n, r):
    """A (W, H) mask of n random discs of radius about r."""
    kc, kr = jax.random.split(key)
    centres = jax.random.uniform(kc, (n, 2)) * jnp.array([W, H])
    radii = r * (0.6 + 0.8 * jax.random.uniform(kr, (n,)))
    xs, ys = jnp.meshgrid(jnp.arange(W), jnp.arange(H), indexing="ij")
    d = jnp.sqrt((xs[None] - centres[:, 0, None, None]) ** 2 + (ys[None] - centres[:, 1, None, None]) ** 2)
    return (d < radii[:, None, None]).any(0)


def make_terrain(key, cfg: Config):
    W, H = cfg.W, cfg.H
    k = jax.random.split(key, 5)
    t = jnp.full((W, H), GRASS)
    t = jnp.where(_blobs(k[0], W, H, 4, 4.0), TALL, t)
    t = jnp.where(_blobs(k[1], W, H, 4, 3.0), SCRUB, t)
    t = jnp.where(_blobs(k[2], W, H, 4, 4.0), WOODS, t)
    t = jnp.where(_blobs(k[3], W, H, 3, 2.5), STONY, t)
    t = jnp.where(_blobs(k[4], W, H, 2, 3.5), WATER, t)
    return t


def reset(key, cfg: Config) -> State:
    W, H, S = cfg.W, cfg.H, cfg.S
    kt, kp, ks, ka, kg = jax.random.split(key, 5)
    terrain = make_terrain(kt, cfg)
    res = jnp.where(terrain >= WOODS, cfg.res_max, 0)
    # founders start on dry land near the middle
    land = (terrain != WATER).reshape(-1)
    idx = jax.random.choice(kp, W * H, (S,), replace=False, p=land / land.sum())
    pos = jnp.stack([idx // H, idx % H], -1)
    alive = jnp.arange(S) < cfg.founders
    genes = jnp.concatenate([jnp.ones((S, 5)), jnp.ones((S, 1)), jnp.zeros((S, 2))], -1)
    genes = genes + cfg.mutation * jax.random.normal(kg, genes.shape)
    body = jnp.tile(jnp.array([0.8, 0.8, 0.8, 0.9, 1.0]), (S, 1))
    return State(t=jnp.int32(6), terrain=terrain, res=res, ground=jnp.zeros((W, H, N_ITEMS), jnp.int32),
                 fire=jnp.zeros((W, H), jnp.int32), shelter=jnp.zeros((W, H), jnp.int32), alive=alive, pos=pos,
                 sex=jnp.arange(S) % 2, age=jax.random.uniform(ka, (S,), minval=16.0, maxval=28.0),
                 hands=jnp.zeros((S, 2), jnp.int32), body=body, asleep=jnp.zeros(S, bool),
                 preg=jnp.zeros(S, jnp.int32), father_of=jnp.full(S, -1), genes=genes,
                 pid=jnp.where(alive, jnp.arange(S) + 1, 0), mother=jnp.zeros(S, jnp.int32),
                 father=jnp.zeros(S, jnp.int32), signal=jnp.full(S, -1), next_pid=jnp.int32(cfg.founders + 1),
                 made=jnp.zeros(N_ITEMS, jnp.int32), births=jnp.int32(0), deaths=jnp.int32(0))


def season(t):
    return (t // DAY) % 4


def ambient(t):
    """Air temperature pull on warmth: cold nights and winters."""
    hour = t % DAY
    night = (hour >= 21) | (hour < 6)
    s = season(t)
    base = jnp.array([0.65, 0.85, 0.6, 0.25])[s]
    return base - 0.25 * night


def drive(body, genes):
    """How far the body is from where it wants to be, weighted by what this person cares about."""
    w = jnp.maximum(genes[..., :5], 0.0)
    return (w * (1.0 - body) ** 2).sum(-1)


def _free_hand(hands):
    """Index of an empty hand (0 or 1), or -1."""
    return jnp.where(hands[0] == 0, 0, jnp.where(hands[1] == 0, 1, -1))


def _held(hands):
    """Index of a non-empty hand (0 preferred), or -1."""
    return jnp.where(hands[0] != 0, 0, jnp.where(hands[1] != 0, 1, -1))


def _combine(a, b, fire_here):
    """What two items in hand become. Returns (new item for hand 0, new item for hand 1, placed: 0 none 1 fire 2 shelter)."""
    lo, hi = jnp.minimum(a, b), jnp.maximum(a, b)
    out = (a, b, 0)
    food = (a == BERRIES) | (b == BERRIES)
    rules = [
        ((lo == STONE) & (hi == STONE), (SHARP, 0, 0)),
        ((lo == BRANCH) & (hi == FIBRE), (ROPE, 0, 0)),
        ((lo == BRANCH) & (hi == BRANCH), (0, 0, 1)),
        ((lo == BRANCH) & (hi == ROPE), (0, 0, 2)),
        (food & fire_here, (jnp.where(a == BERRIES, COOKED, a), jnp.where(b == BERRIES, COOKED, b), 0)),
    ]
    r0, r1, placed = out
    for cond, (x, y, p) in reversed(rules):
        r0 = jnp.where(cond, x, r0)
        r1 = jnp.where(cond, y, r1)
        placed = jnp.where(cond, p, placed)
    return r0, r1, placed


def _act_one(carry, inp, cfg: Config):
    """Apply one person's hand action (grab, drop, eat, drink, combine, give) to the shared world."""
    st, key = carry
    i, a = inp
    key, k1 = jax.random.split(key)
    alive = st.alive[i] & ~st.asleep[i]
    x, y = st.pos[i, 0], st.pos[i, 1]
    hands = st.hands[i]
    child = st.age[i] < cfg.child_age
    # grab: items lying on the cell first, then the terrain's resource
    free = _free_hand(hands)
    ground_here = st.ground[x, y]
    has_item = ground_here[1:].sum() > 0
    item_on_ground = jnp.argmax(ground_here[1:] > 0) + 1
    terr = st.terrain[x, y]
    rsrc = RESOURCE_OF[terr]
    tool = (hands == SHARP).any()
    p = jnp.where(tool, jnp.array(cfg.grab_p_tool)[rsrc], jnp.array(cfg.grab_p)[rsrc])
    got_res = (st.res[x, y] > 0) & (rsrc > 0) & (jax.random.uniform(k1) < p)
    do_grab = alive & (a == GRAB) & (free >= 0) & ~child
    take_item = do_grab & has_item
    take_res = do_grab & ~has_item & got_res
    new_item = jnp.where(take_item, item_on_ground, jnp.where(take_res, rsrc, 0))
    hands = jnp.where((take_item | take_res) & (jnp.arange(2) == free), new_item, hands)
    ground = st.ground.at[x, y, item_on_ground].add(-take_item.astype(jnp.int32))
    res = st.res.at[x, y].add(-take_res.astype(jnp.int32))
    # drop
    held = _held(hands)
    do_drop = alive & (a == DROP) & (held >= 0)
    dropped = jnp.where(held >= 0, hands[jnp.maximum(held, 0)], 0)
    ground = ground.at[x, y, dropped].add(do_drop.astype(jnp.int32) * (dropped > 0))
    hands = jnp.where(do_drop & (jnp.arange(2) == held), 0, hands)
    # eat: the most filling food in hand
    fv = jnp.array(cfg.food_value)
    vals = fv[hands]
    k_eat = jnp.argmax(vals)
    do_eat = alive & (a == EAT) & (vals.max() > 0)
    body = st.body[i]
    body = body.at[0].set(jnp.where(do_eat, jnp.minimum(1.0, body[0] + vals.max()), body[0]))
    hands = jnp.where(do_eat & (jnp.arange(2) == k_eat), 0, hands)
    # drink: on or next to water
    xs = jnp.clip(jnp.array([x, x + 1, x - 1, x, x]), 0, cfg.W - 1)
    ys = jnp.clip(jnp.array([y, y, y, y + 1, y - 1]), 0, cfg.H - 1)
    near_water = (st.terrain[xs, ys] == WATER).any()
    do_drink = alive & (a == DRINK) & near_water & ~child
    body = body.at[1].set(jnp.where(do_drink, jnp.minimum(1.0, body[1] + cfg.drink_value), body[1]))
    # combine the two items in hand
    fire_here = st.fire[x, y] > 0
    h0, h1, placed = _combine(hands[0], hands[1], fire_here)
    do_comb = alive & (a == COMBINE) & ~child & ((h0 != hands[0]) | (h1 != hands[1]) | (placed > 0))
    made_item = jnp.where(do_comb & (placed == 0), jnp.maximum(h0, h1), 0)
    newly = jnp.where(do_comb & (placed == 0) & ((h0 != hands[0]) | (h1 != hands[1])),
                      jnp.where(h0 != hands[0], h0, h1), 0)
    hands = jnp.where(do_comb, jnp.array([h0, h1]), hands)
    fire = st.fire.at[x, y].set(jnp.where(do_comb & (placed == 1), cfg.fire_hours, st.fire[x, y]))
    shelter = st.shelter.at[x, y].add((do_comb & (placed == 2)).astype(jnp.int32))
    made = st.made.at[newly].add((newly > 0).astype(jnp.int32))
    made = made.at[0].add(0)
    # give: to someone on your cell or next to you who has a free hand (nearest, lowest slot first)
    held = _held(hands)
    d = jnp.abs(st.pos - st.pos[i]).max(-1)
    other_free = (st.hands == 0).any(-1)
    ok = st.alive & (d <= 1) & other_free & (jnp.arange(cfg.S) != i)
    j = jnp.argmin(jnp.where(ok, d * 100 + jnp.arange(cfg.S), 10_000))
    do_give = alive & (a == GIVE) & (held >= 0) & ok.any()
    item = jnp.where(held >= 0, hands[jnp.maximum(held, 0)], 0)
    hands = jnp.where(do_give & (jnp.arange(2) == held), 0, hands)
    jfree = _free_hand(st.hands[j])
    all_hands = st.hands.at[i].set(hands)
    all_hands = all_hands.at[j].set(jnp.where(do_give & (jnp.arange(2) == jfree), item, all_hands[j]))
    st = st._replace(hands=all_hands, ground=ground, res=res, body=st.body.at[i].set(body), fire=fire,
                     shelter=shelter, made=made)
    events = jnp.array([do_give, do_comb & (placed == 1), do_comb & (placed == 2), do_eat, do_drink,
                        take_res | take_item], jnp.int32)
    return (st, key), events


def step(st: State, actions, key, cfg: Config):
    """One hour. Returns (state, reward (S,), info)."""
    S = cfg.S
    k_perm, k_act, k_mate, k_birth, k_old, k_regrow, k_mut = jax.random.split(key, 7)
    a = jnp.where(st.alive, actions, STAY)
    child = st.age < cfg.child_age
    d_before = drive(st.body, st.genes)
    # sleep and signals
    asleep = st.alive & (a == SLEEP)
    signal = jnp.where(st.alive & (a >= SIG0), a - SIG0, -1)
    # moves (not into water); small children stay with their mother
    mv = MOVES[jnp.clip(a, 0, 4)] * ((a >= 1) & (a <= 4) & st.alive & ~asleep)[:, None]
    new = jnp.clip(st.pos + mv, 0, jnp.array([cfg.W - 1, cfg.H - 1]))
    wet = st.terrain[new[:, 0], new[:, 1]] == WATER
    pos = jnp.where(wet[:, None], st.pos, new)
    mom = jnp.argmax((st.pid[None, :] == st.mother[:, None]) & st.alive[None, :] & (st.mother[:, None] > 0), -1)
    has_mom = ((st.pid[None, :] == st.mother[:, None]) & st.alive[None, :] & (st.mother[:, None] > 0)).any(-1)
    pos = jnp.where((child & has_mom)[:, None], pos[mom], pos)
    st = st._replace(pos=pos, asleep=asleep, signal=signal)
    # hand actions, one person at a time in random order
    order = jax.random.permutation(k_perm, S)
    (st, _), ev = jax.lax.scan(lambda c, inp: _act_one(c, inp, cfg), (st, k_act), (order, a[order]))
    ev_total = ev.sum(0)
    # mating: two adjacent adults of opposite sex who both choose it, fed and fertile, not related
    lo, hi = cfg.fertile
    fit = st.alive & (a == MATE) & (st.age >= lo) & (st.age <= hi) & (st.body[:, 0] > 0.4) & (st.body[:, 1] > 0.4) \
        & (st.body[:, 4] > 0.5)
    adj = (jnp.abs(st.pos[:, None] - st.pos[None]).max(-1) <= 1)
    kin = (st.mother[:, None] == st.pid[None]) | (st.mother[None] == st.pid[:, None]) | \
        (st.father[:, None] == st.pid[None]) | (st.father[None] == st.pid[:, None]) | \
        ((st.mother[:, None] == st.mother[None]) & (st.mother[:, None] > 0))
    pair = fit[:, None] & fit[None] & adj & (st.sex[:, None] == 0) & (st.sex[None] == 1) & ~kin & (st.preg[:, None] == 0)
    has_pair = pair.any(-1)
    mate_of = jnp.argmax(pair, -1)
    conceive = has_pair & (jax.random.uniform(k_mate, (S,)) < cfg.conceive_p)
    preg = jnp.where(conceive, cfg.gestation, st.preg)
    father_of = jnp.where(conceive, mate_of, st.father_of)
    # body: hunger, thirst, warmth, energy; infants nurse with their mother
    b = st.body
    sat = b[:, 0] - cfg.satiety_loss * jnp.where(asleep, 0.5, 1.0) * jnp.where(preg > 0, 1.3, 1.0)
    hyd = b[:, 1] - cfg.hydration_loss * jnp.where(asleep, 0.5, 1.0) * jnp.where(season(st.t) == 1, 1.3, 1.0)
    fire_near = jnp.stack([st.fire[jnp.clip(st.pos[:, 0] + dx, 0, cfg.W - 1), jnp.clip(st.pos[:, 1] + dy, 0, cfg.H - 1)]
                           for dx in (-1, 0, 1) for dy in (-1, 0, 1)], -1).max(-1) > 0
    in_shelter = st.shelter[st.pos[:, 0], st.pos[:, 1]] > 0
    same_cell = (st.pos[:, None] == st.pos[None]).all(-1) & st.alive[None] & (jnp.arange(S)[:, None] != jnp.arange(S)[None])
    huddle = same_cell.sum(-1)
    target = ambient(st.t) + cfg.fire_warmth * fire_near * 2 + cfg.shelter_warmth * in_shelter * 2 + cfg.huddle_warmth * huddle
    warm = b[:, 2] + 0.15 * (jnp.clip(target, 0, 1) - b[:, 2])
    sleep_gain = cfg.sleep_gain * jnp.where(in_shelter, 1.4, 1.0) * jnp.where(((st.t % DAY) >= 21) | ((st.t % DAY) < 6), 1.2, 0.7)
    en = jnp.where(asleep, b[:, 3] + sleep_gain, b[:, 3] - cfg.energy_loss * jnp.where(child, 1.2, 1.0))
    nursing = child & has_mom & (st.age < 2.5)
    sat = jnp.where(nursing, sat + cfg.nurse, sat)
    hyd = jnp.where(nursing, hyd + cfg.nurse, hyd)
    cost = jnp.zeros(S).at[mom].add(nursing * cfg.nurse_cost)
    sat, hyd = sat - cost, hyd - cost
    crit = (jnp.stack([sat, hyd, warm, en], -1) < 0.1).sum(-1)
    health = b[:, 4] - cfg.starve_damage * crit + cfg.heal * (crit == 0)
    body = jnp.clip(jnp.stack([sat, hyd, warm, en, health], -1), 0, 1)
    body = jnp.where(st.alive[:, None], body, 0.0)
    age = st.age + st.alive / YEAR
    # deaths
    p_old = jnp.where(age > cfg.old_age, ((age - cfg.old_age) / 20.0) ** 2 * 0.003, 0.0)
    dies = st.alive & ((body[:, 4] <= 0) | (jax.random.uniform(k_old, (S,)) < p_old))
    alive = st.alive & ~dies
    hands = jnp.where(dies[:, None], 0, st.hands)
    # a dead person's things fall where they lay
    ground = st.ground
    for h in range(2):
        ground = ground.at[st.pos[:, 0], st.pos[:, 1], hands[:, h] * 0 + st.hands[:, h]].add(dies * (st.hands[:, h] > 0))
    # births
    preg = jnp.where(alive & (preg > 0), preg - 1, jnp.where(alive, preg, 0))
    due = alive & (st.preg == 1)
    free_slots = ~alive
    n_free = free_slots.sum()
    rank_due = jnp.cumsum(due) - 1
    free_idx = jnp.argsort(~free_slots)                       # free slots first
    slot = jnp.where(due & (rank_due < n_free), free_idx[jnp.clip(rank_due, 0, S - 1)], -1)
    kb = jax.random.split(k_birth, 3)
    child_sex = jax.random.bernoulli(kb[0], 0.5, (S,)).astype(jnp.int32)
    dad = jnp.clip(father_of, 0, S - 1)
    child_genes = (st.genes + st.genes[dad]) / 2 + cfg.mutation * jax.random.normal(kb[1], st.genes.shape)
    born = jnp.zeros(S, bool).at[jnp.where(slot >= 0, slot, S)].set(True, mode="drop")
    src = jnp.zeros(S, jnp.int32).at[jnp.where(slot >= 0, slot, S)].set(jnp.arange(S), mode="drop")   # mother slot
    new_pid = st.next_pid + jnp.cumsum(born) - 1
    alive = alive | born
    pos = jnp.where(born[:, None], st.pos[src], st.pos)
    sex = jnp.where(born, child_sex[src], st.sex)
    age = jnp.where(born, 0.0, age)
    hands = jnp.where(born[:, None], 0, hands)
    body = jnp.where(born[:, None], jnp.array([0.8, 0.8, 0.8, 0.9, 1.0]), body)
    genes = jnp.where(born[:, None], child_genes[src], st.genes)
    pid = jnp.where(born, new_pid, jnp.where(dies & ~born, 0, st.pid))
    mother = jnp.where(born, st.pid[src], st.mother)
    father = jnp.where(born, st.pid[dad[src]], st.father)
    preg = jnp.where(born, 0, preg)
    father_of = jnp.where(born, -1, father_of)
    asleep = jnp.where(born, False, st.asleep)
    # the world: regrowth (seasonal), fires burn down
    s = season(st.t)
    grow_p = jnp.array(cfg.regrow)[st.terrain] * jnp.array([1.0, 1.0, 0.6, 0.0])[s]
    grow = (jax.random.uniform(k_regrow, st.res.shape) < grow_p) & (st.res < cfg.res_max)
    res = st.res + grow
    fire = jnp.maximum(st.fire - 1, 0)
    # rewards: the body coming back into balance, weighted by each person's own inner values
    d_after = drive(body, genes)
    r = jnp.where(st.alive & ~dies, d_before - d_after, 0.0)
    r = r - cfg.death_penalty * dies
    # parents: a birth, and their young children's wellbeing (weighted by the 'kin' gene)
    is_parent_of = (mother[None, :] == st.pid[:, None]) | (father[None, :] == st.pid[:, None])
    births_to = (is_parent_of & born[None, :]).sum(-1)
    r = r + cfg.birth_bonus * jnp.maximum(st.genes[:, 5], 0) * births_to * st.alive
    young = alive & (age < cfg.adult_age)
    kid_d = (is_parent_of & young[None, :]) * (1.0 - body[None, :, :]).mean(-1)[None, :]
    r = r - st.genes[:, 6] * kid_d.sum(-1) * 0.05 * st.alive
    r = r + st.genes[:, 7] * (huddle > 0) * 0.01 * st.alive           # company
    st = st._replace(t=st.t + 1, res=res, ground=ground, fire=fire, alive=alive, pos=pos, sex=sex, age=age, hands=hands,
                     body=body, asleep=asleep, preg=preg, father_of=father_of, genes=genes, pid=pid, mother=mother,
                     father=father, next_pid=st.next_pid + born.sum(), births=st.births + born.sum(),
                     deaths=st.deaths + dies.sum())
    info = dict(died=dies, born=born, gives=ev_total[0], fires=ev_total[1], shelters=ev_total[2], meals=ev_total[3],
                drinks=ev_total[4], grabs=ev_total[5], mates=conceive.sum())
    return st, r, info


def observe(st: State, cfg: Config):
    """Egocentric view (terrain, resources, items, fires, shelters, people, signals) plus own body and hands."""
    W, H, S, v = cfg.W, cfg.H, cfg.S, cfg.view
    terr = jax.nn.one_hot(st.terrain, N_TERRAIN)
    people = jnp.zeros((W, H)).at[st.pos[:, 0], st.pos[:, 1]].add(st.alive * 1.0)
    sig = jnp.zeros((W, H, cfg.n_signals)).at[st.pos[:, 0], st.pos[:, 1], jnp.maximum(st.signal, 0)].add(
        (st.alive & (st.signal >= 0)) * 1.0)
    grid = jnp.concatenate([terr, (st.res > 0)[..., None] * 1.0, (st.ground[..., 1:] > 0) * 1.0,
                            (st.fire > 0)[..., None] * 1.0, (st.shelter > 0)[..., None] * 1.0,
                            jnp.minimum(people, 3)[..., None] / 3.0, jnp.minimum(sig, 1.0)], -1)
    C = grid.shape[-1]
    pad = jnp.pad(grid, ((v, v), (v, v), (0, 0)))
    # outside the map reads as water, so the edge is visible
    pad = pad.at[:v, :, WATER].set(1.0).at[-v:, :, WATER].set(1.0).at[:, :v, WATER].set(1.0).at[:, -v:, WATER].set(1.0)

    def window(p):
        return jax.lax.dynamic_slice(pad, (p[0], p[1], 0), (2 * v + 1, 2 * v + 1, C))
    views = jax.vmap(window)(st.pos).reshape(S, -1)
    hour = st.t % DAY
    th = 2 * jnp.pi * hour / DAY
    timef = jnp.concatenate([jnp.array([jnp.sin(th), jnp.cos(th)]), jax.nn.one_hot(season(st.t), 4)])
    # kin: where my mother and my youngest child are, relative to me
    is_mom = (st.pid[None, :] == st.mother[:, None]) & st.alive[None, :] & (st.mother[:, None] > 0)
    mom = jnp.argmax(is_mom, -1)
    kid = (st.mother[None, :] == st.pid[:, None]) | (st.father[None, :] == st.pid[:, None])
    kid = kid & st.alive[None, :] & (st.age[None, :] < cfg.adult_age)
    youngest = jnp.argmin(jnp.where(kid, st.age[None, :], 1e9), -1)
    rel = lambda j, ok: jnp.where(ok[:, None], jnp.clip((st.pos[j] - st.pos) / 8.0, -1, 1), 0.0)
    own = jnp.concatenate([
        st.body, jax.nn.one_hot(st.hands[:, 0], N_ITEMS), jax.nn.one_hot(st.hands[:, 1], N_ITEMS),
        jnp.stack([st.age / 50.0, st.sex * 1.0, (st.preg > 0) * 1.0, (st.age < cfg.child_age) * 1.0,
                   (st.age < cfg.adult_age) * 1.0, st.asleep * 1.0], -1),
        jnp.broadcast_to(timef, (S, 6)), st.genes / 2.0,
        rel(mom, is_mom.any(-1)), is_mom.any(-1)[:, None] * 1.0, rel(youngest, kid.any(-1)), kid.any(-1)[:, None] * 1.0], -1)
    return jnp.concatenate([views, own], -1).astype(jnp.float32)


def n_actions(cfg: Config):
    return SIG0 + cfg.n_signals
