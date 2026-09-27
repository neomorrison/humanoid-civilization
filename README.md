# humanoid-civilization

Creating a humanoid civilization where behavior is simulated and learned, not taught.

**Watch it:** https://neomorrison.github.io/humanoid-civilization/ (3D replays at different
stages of training). Earlier versions stay online: the Society (v2) and the first economy (v1).

## Where this is going

From hand-written behaviour to agents that invent their own: learned actions, then skills,
then composing skills, then social strategies, communication and culture, then physical
bodies and real robots. The world can be changed at any moment to see how they react.
See [ROADMAP.md](ROADMAP.md).

| version | world | what is learned |
|---|---|---|
| v1 Economy | warehouse, farmer, police | a policy choosing among a few jobs (everyone learned to steal) |
| v2 Society | whole lives, families, three foods, reputation | a policy choosing high-level intentions, happiness as reward |
| **v3 Town Life** | hours and seasons, sleep, jobs with shifts, canteen and shop, houses, school | daily routines, families, crime, from happiness |
| Wild (in development) | bodies to keep in balance, materials, hidden recipes | primitive actions only; inner values evolve ([design](docs/WILD.md)) |

## The rule: no automated systems

Nothing happens to anyone's money, goods or relationships unless someone chose it: gifts,
sales at your own price, theft, gossip, shunning, courtship, police volunteering and arrests
are all choices. Where the world still has fixed rules (in Town Life: the businesses' wage,
shift and docking terms, the shop's prices) they are listed in the config and are the next
things to hand over to agents.

## Town Life (v3)

One step is an hour and each day is a season. People need food of three kinds, sleep (best
in their own bed at night), and money to buy food. They choose jobs (paid per hour present
on shift, docked per hour missing outside the lunch break), lunch at the canteen, groceries
at the shop, family dinners, courtship, children, school, and everything social. Their
reward is happiness: fed, rested, healthy, with company, a partner, thriving children, the
regard of others and a little savings, minus grief. See [DESIGN.md](DESIGN.md).

```bash
pip install -r requirements.txt
python scripts/train_town.py --minutes 600 --out runs/town              # train + record replays
python -m civ.metrics runs/town/replays/replay_it00350.json             # score a replay
python scripts/ablate_town.py --base runs/town/checkpoint.pkl          # switch ingredients off, compare
python scripts/train_evil.py --town runs/town/policy.npz              # a newcomer who only wants harm
python scripts/record_stages.py --run runs/society                     # (v2) full-length stage replays
python scripts/build_site.py --run runs/town --v1 runs/economy --out site
```

Tools that keep experiments cheap and honest:

* **Warm start** (`civ/warmstart.py`): checkpoints store the names of what agents sense and
  do, so adding an input or an action keeps everything already learned.
* **Metrics** (`civ/metrics.py`): routine, work, family, inequality, justice and wellbeing
  scores for any replay.
* **Ablations** (`scripts/ablate_town.py`): the same checkpoint continued with one ingredient
  switched off (docking, lunch break, police, gossip), measured over several seeds.
* **Sandbox** (`civ/sandbox.py`): a live town you can intervene in: remove someone, add a
  newcomer or a harm-seeker, change a rule mid-life.
* **Memory** (`civ/mappo_rnn.py`): recurrent policies whose memory lasts a lifetime.

## Layout

```
civ/town.py, record_town.py     Town Life (v3) and its hourly replays (docs/TOWN_REPLAY.md)
civ/sandbox.py                  live interventions on a running town
civ/metrics.py                  behaviour metrics
civ/mappo.py, mappo_rnn.py      multi-agent PPO (feed-forward / recurrent), warm start
civ/wild.py, ppo_wild.py        Wild: JAX world and all-JAX PPO (in development)
civ/society.py, economy.py      v2 Society, v1 Economy
scripts/                        training, ablations, site building, scripted baselines
viewer3d/                       3D viewers (GitHub Pages)
```
