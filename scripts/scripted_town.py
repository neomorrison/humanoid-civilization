"""A hand-written daily routine for Town Life, used to check the world is livable and balanced.

    python scripts/scripted_town.py --years 20 --worlds 8

Adults with a job sleep at night, eat breakfast, work their shift, have lunch at the canteen,
buy groceries after work and eat dinner at home; the jobless apply where staff is short;
children go to school; parents feed hungry children; singles court. Not used in training.
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from civ.town import (APPLY0, COURT, DAY, EAT, FEED_CHILD, GO_FAMILY, GO_HOME, GROCERIES, IDLE, JOBS, LUNCH,  # noqa: E402
                      N_BASE, N_PER, SCHOOL, SLEEP, WORK, YEAR, Town, TownConfig)


def routine(env: Town) -> np.ndarray:
    c, M, S = env.cfg, env.M, env.S
    hour, season = env.clock()
    a = np.full((M, S), IDLE)
    staff = np.stack([((env.job == k) & env.alive).sum(1) for k in range(len(JOBS))], -1)
    want = np.array([0, 3, 3, 3, 2, 2])
    for m in range(M):
        h = hour[m]
        for s in np.nonzero(env.alive[m])[0]:
            if env.intent[m, s] >= 0:
                continue
            age = env.age[m, s]
            night = h >= 21 or h < 6
            if night:
                a[m, s] = SLEEP
                continue
            if env.hunger[m, s] > 0.35 and env.food[m, s].sum() > 0:
                a[m, s] = EAT
                continue
            open_now = env.canteen_open()[0][m]
            if env.hunger[m, s] > 0.45 and env.food[m, s].sum() == 0 and open_now and age >= c.adult_age:
                a[m, s] = LUNCH                  # hungry with nothing to eat: a meal at the canteen
                continue
            if age < c.adult_age:
                school = c.school_hours[0] - 1 <= h < c.school_hours[1] and season[m] < 3 and c.school_ages[0] <= age
                a[m, s] = SCHOOL if school else GO_FAMILY if age < 12 else GO_HOME
                continue
            kidm = env._kids_matrix()[m, s]
            if kidm.any() and env.food[m, s].sum() > 0 and env.hunger[m][kidm].max() > 0.4:
                a[m, s] = FEED_CHILD
                continue
            j = env.job[m, s]
            if j == 0:
                k = int(np.argmax(want - staff[m]))
                if want[k] - staff[m, k] > 0 and k > 0:
                    a[m, s] = APPLY0 + k - 1
                    staff[m, k] += 1
                    continue
            else:
                s0, s1, b0, b1 = c.shifts[JOBS[j]]
                if env.shift_on[j, (h + 1) % DAY, season[m]] or env.shift_on[j, h, season[m]]:
                    at_canteen = JOBS[j] == "canteen" and env.at_work()[m, s]
                    a[m, s] = LUNCH if at_canteen and env.hunger[m, s] > 0.45 and env.food[m, s].sum() == 0 else WORK
                    continue
                if b0 <= h < b1:
                    a[m, s] = LUNCH if env.hunger[m, s] > 0.15 else IDLE
                    continue
            if env.food[m, s].sum() < 3 and c.shop_hours[0] <= h < c.shop_hours[1] - 1 and env.money[m, s] >= 3:
                a[m, s] = GROCERIES
                continue
            if env.partner[m, s] < 0 and age >= c.partner_age:
                for k in range(4):
                    jn = env._nb[m, s, k]
                    if jn >= 0 and env._nd[m, s, k] <= c.reach and env.sex[m, jn] != env.sex[m, s] and env.partner[m, jn] < 0:
                        a[m, s] = N_BASE + k * N_PER + COURT
                        break
                if a[m, s] != IDLE:
                    continue
            a[m, s] = GO_HOME
    return a.reshape(-1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--years", type=float, default=20)
    ap.add_argument("--worlds", type=int, default=8)
    args = ap.parse_args()
    env = Town(args.worlds, TownConfig(), seed=1)
    R = 0.0
    n = 0
    for t in range(int(args.years * YEAR)):
        al = env.alive.reshape(-1).copy()
        _, _, rew, _, _ = env.step(routine(env))
        R += float((rew * al).sum())
        n += int(al.sum())
        if (t + 1) % (5 * YEAR) == 0:
            print(f"year {(t + 1) / YEAR:4.0f}: alive {env.alive.sum(1).tolist()} treasury {env.treasury.round(0).tolist()} "
                  f"stock {env.stock.mean(0).round(1).tolist()} mean money {env.money[env.alive].mean():.1f}", flush=True)
    ep = {k: round(float(v.mean()), 1) for k, v in env.ep.items()}
    print("per world:", ep)
    print(f"happiness per person-hour {R / max(n, 1):+.4f}")


if __name__ == "__main__":
    main()
