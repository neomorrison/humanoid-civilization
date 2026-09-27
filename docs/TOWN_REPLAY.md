# Town Life (v3) replay format

A replay is one world recorded **every hour** (`stride` = 1 step = 1 hour) for up to a
60-year era. The simulator writes it (`civ/record_town.py`); the viewer
(`viewer3d/town.html`) reads it from `replays/<file>.json`, listed in `replays/index.json`.

## Time

* 1 step = 1 hour; `day_steps` = 24; `days_per_year` = 4, one day per season
  (`seasons` = spring, summer, autumn, winter). A year is 96 steps.
* For absolute step `t`: `hour = t % 24`, `day = floor(t / 24)`,
  `season = day % 4`, `year = floor(day / 4)`.
* Night is 21:00–06:00 (dark); dawn 06:00–07:00, dusk 20:00–21:00.

## Top level

```jsonc
{
  "kind": "town", "version": 3,
  "meta": {"iteration": 1000, "samples": 12345678},   // training stage of the policy
  "config": { ... },                                    // see below
  "start_step": 0, "stride": 1,
  "activities": ["idle", "walking", "working", "eating", "buying", "sleeping", "school",
                 "socialising", "home", "detained", "nursing"],
  "jobs": ["none", "farm", "orchard", "dairy", "warehouse", "canteen"],
  "foods": ["grain", "fruit", "dairy"],
  "people": {"<pid>": {"sex": 0, "mother": 0, "father": 0, "traits": [0.4, 0.2, 0.7, 0.5],
                       "born_step": -1728}},            // sex 0 = female, 1 = male; 0 = no parent
  "slot_cols": ["pid", "x", "y", "act", "age10", "hunger", "energy", "health", "money",
                "g", "f", "d", "job", "home", "partner", "flags"],
  "frames": [{"t": 0, "s": [[...], 0, [...]], "w": {...}}],
  "events": [[t, "kind", pid_i, pid_j, amount]],
  "summary": {"births": 12, "deaths": 9, ...}
}
```

### `config` (what the viewer needs)

```jsonc
{
  "width": 32, "height": 24,               // grid cells; x grows east, y grows north
  "day_steps": 24, "days_per_year": 4, "seasons": ["spring", "summer", "autumn", "winter"],
  "roads":  [[0, 11, 31, 11], [0, 18, 31, 18], [9, 0, 9, 23], [22, 0, 22, 23]],  // boxes of road tiles
  "houses": [[1, 20, 2, 21], [5, 20, 6, 21], ...],                              // 10 lots, 4 beds each
  "house_beds": 4,
  "places": {"farm": [11, 1, 20, 9], "orchard": [24, 6, 30, 9], "dairy": [24, 0, 30, 4],
             "warehouse": [1, 5, 7, 9], "canteen": [16, 13, 20, 16], "market": [11, 13, 14, 16],
             "school": [10, 20, 14, 22], "station": [24, 13, 26, 16]},
  "shifts": {"farm": [8, 17, 12, 13], "orchard": [8, 17, 12, 13], "dairy": [6, 15, 11, 12],
             "warehouse": [8, 17, 12, 13], "canteen": [10, 19, 15, 16]},  // start, end, break start, break end
  "school_hours": [8, 15],
  "slots": 24
}
```

Boxes are inclusive `[x0, y0, x1, y1]` in grid cells. Anything else in `config` is a
simulation parameter the viewer may ignore.

### `frames`

One frame per hour. `s` has one entry per person slot (`config.slots`): `0` for an empty
slot (unborn or dead), otherwise an array of integers in `slot_cols` order:

| column | meaning |
|---|---|
| `pid` | person id (key into `people`); a slot is reused by a new person after a death |
| `x`, `y` | grid cell |
| `act` | index into `activities` (what they are doing this hour) |
| `age10` | age in years × 10 |
| `hunger`, `energy`, `health` | 0–100 |
| `money` | coins, rounded |
| `g`, `f`, `d` | grain, fruit, dairy items carried |
| `job` | index into `jobs` |
| `home` | house index, or -1 |
| `partner` | slot index of partner, or -1 |
| `flags` | bitmask: 1 pregnant, 2 detained, 4 police volunteer, 8 asleep, 16 seen stealing, 32 on shift now, 64 docked this hour, 128 child (under 12) |

`w` is the state of the town that hour:

```jsonc
{"stock": [g, f, d],     // food held by the town's businesses (sold at the canteen)
 "treasury": 240,         // coins held by the businesses (pays wages)
 "canteen_open": 1,       // canteen staff on shift, so hot lunches can be bought at the canteen
 "shop_open": 1,          // the grocery shop in the market hall (07:00-20:00) sells the same stock to take home
 "crops": 6,              // 0-9 how grown the grain fields look (season-driven)
 "fruit": 3,              // 0-9 fruit on the orchard trees
 "herd": 7,               // 0-9 how many cows are out in the pasture
 "price": [2, 3, 2]}      // canteen price of each food, coins
```

### `events`

`[t, kind, pid_i, pid_j, amount]`; `pid_j` is 0 when there is no second person.

| kind | meaning |
|---|---|
| `ate grain` / `ate fruit` / `ate dairy` | i ate (quiet) |
| `family meal` | i ate at home with family (quiet) |
| `bought grain` ... | i bought food: from the canteen (hot lunch, eaten there) or the market shop (groceries) when j = 0 (amount = price), else from person j |
| `gave grain` ... / `gave coin` | i gave j food / a coin |
| `hired` | i took a job; amount = job index |
| `earned` | i finished a shift; amount = coins earned that shift |
| `docked` | i lost coins for missing work; amount = coins lost |
| `theft` / `detained` / `citizens' arrest` | crime and justice (i acts on j) |
| `enlisted` / `resigned` | police volunteering |
| `talked` / `courted` | social (quiet) |
| `partnered` / `separated` / `shunned` / `forgave` | relationships |
| `moved in` | i moved into house `amount` |
| `expecting` / `born` | pregnancy (i mother, j father) / birth (i child, j mother) |
| `fell asleep outside` | i collapsed from exhaustion away from home |
| `died (starved)` / `died (malnutrition)` / `died (old age)` | amount = age |

## Live sandbox

`index.html?live=1` (the "Live sandbox" chip next to the training stages) runs the real
simulator in the browser instead of playing a recording. `civ/sandbox.py` steps one town under
the trained policy inside a Web Worker (`viewer3d/town/live-worker.js`) with
[Pyodide](https://pyodide.org) 0.29.5 (CPython 3.13 + numpy, no JAX), and the page appends each
simulated hour to a growing replay in exactly the format above, playing it a few hours behind
the simulation. Pausing pauses the town; the speeds are the replay's.

Interventions land between two simulated hours and the view jumps to the hour they take effect:

* select someone, **Remove (dies now)**: `Sandbox.kill(pid)`; their partner, parents and
  children grieve (by the rule `Town._die` rewards grief with), their job and bed free up;
* **Add newcomer** (sex, age) and **Add harm-seeker**: `Sandbox.spawn(sex, age, evil)`. A
  harm-seeker acts with the harm-seeker policy (`runs/evil/policy.npz`, trained by
  `scripts/train_evil.py`); without one, with the townspeople's policy but with evil traits (no
  empathy, all greed and boldness), and the page says so. Harm-seekers get a red ring, a red
  marker and a red label;
* **Rules of the town**: every key of `sandbox.RULES` (wage, docking, lunch break, police,
  gossip, food price, export pay, hunger rate, conception) with its current value.

What the worker exchanges with the sandbox, as JSON: `Sandbox.header()` is the top level above
without `frames`/`events`/`people`, plus `live: true`, `rules` (key -> explanation) and
`policies` (`town` / `evil` meta, `evil` null when there is none). `Sandbox.advance(hours)` returns
only what is new: `{"frames": [...], "events": [...], "people": {...}, "ended": false}`. Each
live frame also has `"evil": [pid, ...]` (harm-seekers alive) and people have `"evil": true|false`.
`ended` turns true when everybody has died (the era length is set to 10,000 years, so time never
runs out while you watch; the page stops by itself after 100 years to keep memory in check).

Extra event kinds in live mode:

| kind | meaning |
|---|---|
| `died (removed)` | i was removed by the viewer; amount = age |
| `grieves` | i grieves for j, who died; amount = the grief (happiness lost) |
| `arrived` / `arrived (evil)` | newcomer i moved into town / a harm-seeker did |
| `rule <key>` | a rule was changed; amount = the new value (booleans 0/1) |

Files: `scripts/build_site.py` writes `site/live/` whenever `runs/town/policy.npz` exists (and
`--no-live` leaves it out): `manifest.json`, `civ/{__init__,numpy_policy,town,record_town,sandbox}.py`
(checked to be everything the sandbox imports) and `town.npz` / `evil.npz` (`--live-town`,
`--live-evil`; a harm-seeker policy that fails to load is skipped). Pyodide itself is vendored in
`viewer3d/vendor/pyodide/` (about 15 MB with the numpy wheel, fetched once and cached by the
browser); without it the worker loads the same version from `cdn.jsdelivr.net/pyodide`. URL
options: `seed=N` (default random), `speed=h/s` (default 3), `paused=1`, `livebase=<folder>`.
