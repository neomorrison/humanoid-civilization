"""Behaviour metrics for a Town Life replay (docs/TOWN_REPLAY.md format).

    python -m civ.metrics runs/town/replays/replay_it00250.json

Everything is computed from what the viewer sees, so any replay can be scored:

routine      attendance (share of shift hours an employed adult is at work), punctuality
             (share of shift starts they are already at work), lunch_in_break (canteen and
             shop meals eaten by workers during their break rather than their shift),
             home_at_night (share of night hours spent asleep at home)
work         employment (share of adults with a job), job_spell_days (how long people keep
             a job), docked_share (share of shift hours docked)
family       child_survival (share of children born who reach 16, among those who could),
             school_attendance (share of school hours children aged 6-15 spend at school),
             family_meals_per_day (per household)
equality     money_gini (Gini of adults' money, averaged over the replay)
justice      thefts_per_100_person_years, arrest_rate (arrests per theft)
wellbeing    hunger, energy, health (means over living adults)
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict

import numpy as np


def gini(x):
    x = np.sort(np.asarray(x, float))
    if len(x) == 0 or x.sum() <= 0:
        return 0.0
    n = len(x)
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def replay_metrics(r: dict) -> dict:
    cfg = r["config"]
    C = {c: i for i, c in enumerate(r["slot_cols"])}
    A = {a: i for i, a in enumerate(r["activities"])}
    jobs = r["jobs"]
    day = cfg.get("day_steps", 24)
    shifts = cfg["shifts"]
    winter_off = set(cfg.get("winter_off", []))
    s_lo, s_hi = cfg["school_hours"]
    age_lo, age_hi = cfg.get("school_ages", [6, 16])

    def on_shift(job, hour, season):
        if job == 0:
            return False
        s0, s1, b0, b1 = shifts[jobs[job]]
        return s0 <= hour < s1 and not b0 <= hour < b1 and not (season == 3 and jobs[job] in winter_off)

    def in_break(job, hour):
        if job == 0:
            return False
        _, _, b0, b1 = shifts[jobs[job]]
        return b0 <= hour < b1

    shift_h = present_h = docked_h = 0
    starts = on_time = 0
    night_h = night_home = 0
    school_h = school_present = 0
    adult_h = employed_h = 0
    hunger, energy, health = [], [], []
    ginis = []
    spells = defaultdict(list)            # pid -> list of (job, first_t, last_t)
    prev_shift = {}
    for fr in r["frames"]:
        t = fr["t"]
        hour, season = t % day, (t // day) % 4
        adults_money = []
        for row in fr["s"]:
            if not row:
                continue
            pid, job, act, flags = row[C["pid"]], row[C["job"]], row[C["act"]], row[C["flags"]]
            age = row[C["age10"]] / 10
            if age >= 16:
                adult_h += 1
                employed_h += job > 0
                adults_money.append(max(row[C["money"]], 0))
                hunger.append(row[C["hunger"]])
                energy.append(row[C["energy"]])
                health.append(row[C["health"]])
                sp = spells[pid]
                if job > 0:
                    if sp and sp[-1][0] == job and sp[-1][2] >= t - 1:
                        sp[-1][2] = t
                    else:
                        sp.append([job, t, t])
                excused = flags & 2          # detained
                if on_shift(job, hour, season) and not excused:
                    shift_h += 1
                    at = act == A["working"]
                    present_h += at
                    docked_h += bool(flags & 64)
                    if not prev_shift.get(pid, False):
                        starts += 1
                        on_time += at
                prev_shift[pid] = on_shift(job, hour, season)
                if hour >= 22 or hour < 5:
                    night_h += 1
                    night_home += act == A["sleeping"] and (flags & 8) > 0 and row[C["home"]] >= 0
            elif age_lo <= age < age_hi and s_lo <= hour < s_hi and season < 3:
                school_h += 1
                school_present += act == A["school"]
        if len(adults_money) > 1:
            ginis.append(gini(adults_money))
    # meals during shift vs break, from events (canteen/shop purchases by workers)
    first_frame = {fr["t"]: fr for fr in r["frames"]}
    meals_break = meals_shift = 0
    thefts = arrests = family_meals = 0
    for t, kind, i, j, amt in r["events"]:
        if kind == "theft":
            thefts += 1
        elif kind in ("detained", "citizens' arrest"):
            arrests += 1
        elif kind == "family meal":
            family_meals += 1
        elif kind.startswith("bought ") and j == 0:
            fr = first_frame.get(t)
            if not fr:
                continue
            for row in fr["s"]:
                if row and row[C["pid"]] == i and row[C["job"]] > 0:
                    hour, season = t % day, (t // day) % 4
                    if in_break(row[C["job"]], hour):
                        meals_break += 1
                    elif on_shift(row[C["job"]], hour, season):
                        meals_shift += 1
    # child survival: children born in the replay who reached 16, among those who either died or had time to
    last_t = r["frames"][-1]["t"] if r["frames"] else 0
    died = {i: t for t, kind, i, j, amt in r["events"] if kind.startswith("died")}
    year = day * 4
    born = [(int(pid), p["born_step"]) for pid, p in r["people"].items() if p["mother"]]
    grown = eligible = 0
    for pid, b in born:
        if b + 16 * year <= last_t or (pid in died and died[pid] - b < 16 * year):
            eligible += 1
            grown += not (pid in died and died[pid] - b < 16 * year)
    person_years = adult_h / year
    households = max(1, len(cfg["houses"]))
    days = max(1, len(r["frames"]) / day)
    job_days = [(b - a + 1) / day for sp in spells.values() for (_, a, b) in sp]
    f = lambda a, b: round(a / b, 3) if b else None
    return dict(
        attendance=f(present_h, shift_h), punctuality=f(on_time, starts),
        lunch_in_break=f(meals_break, meals_break + meals_shift), home_at_night=f(night_home, night_h),
        employment=f(employed_h, adult_h), job_spell_days=round(float(np.mean(job_days)), 1) if job_days else None,
        docked_share=f(docked_h, shift_h), child_survival=f(grown, eligible), children_born=len(born),
        school_attendance=f(school_present, school_h), family_meals_per_day=round(family_meals / days / households, 3),
        money_gini=round(float(np.mean(ginis)), 3) if ginis else None,
        thefts_per_100_person_years=round(100 * thefts / max(person_years, 1e-9), 1), arrest_rate=f(arrests, thefts),
        hunger=round(float(np.mean(hunger)) / 100, 3) if hunger else None,
        energy=round(float(np.mean(energy)) / 100, 3) if energy else None,
        health=round(float(np.mean(health)) / 100, 3) if health else None,
        years=round(len(r["frames"]) / year, 1))


if __name__ == "__main__":
    for path in sys.argv[1:]:
        print(path, json.dumps(replay_metrics(json.load(open(path)))))
