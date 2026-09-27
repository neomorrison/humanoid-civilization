# humanoid-civilization

Creating a humanoid civilization where behavior is simulated and learned, not taught.

**Watch it:** https://neomorrison.github.io/humanoid-civilization/ (replays of the agents at
different stages of training, with the town's ledger, an event feed and training curves).

## The rule: no automated systems

Nothing happens to an agent's money or goods unless an agent chose it:

* **Work.** Warehouse workers carry boxes from the shelves to the loading dock. Each shipped box
  earns 5¢ from the outside world, the only money that enters the town.
* **Farming.** The farmer pays 1¢ to plant a plot, waits for it to grow and harvests 2 food.
* **Trade.** There is no shop and no automatic sale. A buyer stands next to someone holding food and
  pays that person's asking price; sellers raise or lower their own price.
* **Theft.** Anyone can grab cash from a neighbour. Nearby agents see it and remember who did it.
* **Policing.** A police officer who saw a theft can detain the thief and confiscate their cash, and
  then decides whether to hand any of it back. Police are not paid a salary or a premium: anyone can
  hand an officer a coin, so they are paid only if people decide protection is worth it.
* **Bodies.** Hunger rises every step; eating food relieves it; starving agents move at half speed.

Each agent is rewarded for staying fed, plus a small value on coins gained. Money is worth having
because it buys food, which makes stealing tempting and makes paying for protection a real choice.

## How they learn

Multi-agent PPO (JAX). One network is shared by every agent with its role as an input; the critic
sees the whole economy during training, while each agent acts only on what it can perceive: the
map, the others' positions and visible state, and its own memories (who it saw stealing, who
robbed it, who paid it, whom it paid).

```bash
pip install -r requirements.txt
python scripts/train_economy.py --minutes 60 --out runs/economy      # train + record replays
python scripts/build_site.py --run runs/economy --out site            # build the viewer
python -m http.server -d site 8000                                    # watch at localhost:8000
```

Knobs (wages, prices, hunger, theft, detention, rewards, roles) are in `civ/economy.py`
(`EconomyConfig`). `--no-police` and `--no-theft` run the obvious control experiments.

## Layout

```
civ/economy.py        the town: vectorised multi-agent world, voluntary transfers only
civ/mappo.py          multi-agent PPO, numpy policy runtime
civ/record.py         records episodes for the viewer
scripts/              training, site building
viewer/viewer.html    the replay viewer (published to GitHub Pages)
```

## Roadmap

1. **Employer as an agent**, so wages are paid by someone who chooses to hire.
2. **Choosing a job**: any agent may work, farm or police; see whether specialisation emerges.
3. **More goods** (tools, seeds, housing) and trade between them; prices set by supply and demand.
4. **Embodiment**: drive physical humanoid bodies (learned walking, running, getting up, carrying)
   from [malecns-test](https://github.com/neomorrison/malecns-test), where a learned controller
   already runs a 23-joint humanoid in MuJoCo with sim-to-real training. The maleCNS fly-brain
   controller from that project can optionally act as an instinct layer (escape, pursuit).
