# humanoid-civilization

Creating a humanoid civilization where behavior is simulated and learned, not taught.

**Watch it:** https://neomorrison.github.io/humanoid-civilization/ (3D replays of the town at
different stages of training, with each person's ledger, their families, an event feed and
training curves). The first economy (v1) is kept at
[economy.html](https://neomorrison.github.io/humanoid-civilization/economy.html).

## The rule: no automated systems

Nothing happens to anyone's money, goods or relationships unless someone chose it. There are no
salaries, taxes, insurance premiums, shops or courts that act on their own:

* **Work.** Crates carried from the warehouse to the dock earn coins from the outside world, the only
  money that enters the town; shipping while others ship pays a team bonus.
* **Food.** Grain must be planted and harvested, fruit picked in the orchard, the herd milked in the
  pasture. People want all three; a one-food diet makes them sick.
* **Trade.** A buyer next to someone holding food pays that person's own asking price. Gifts are
  gifts. Sellers set their prices.
* **Theft and justice.** Anyone can grab cash from a neighbour. Bystanders see it, remember it and
  gossip. Enlisted police who saw a theft can detain the thief; two witnesses together can make a
  citizens' arrest, of police too. Nobody pays police unless they choose to.
* **Families.** Two adults who court each other and find each other attractive enough become
  partners and move into a house; couples who live together have children, who inherit their
  parents' temperament with mutation and depend on them for food.

The only reward is each person's happiness: fed, healthy and varied diet, safety, friends, a
partner, thriving children, the regard of others, useful work, a little for savings, and grief when
loved ones die. [DESIGN.md](DESIGN.md) explains which facts of human life make cooperation pay and
how each one is built into the world.

## How they learn

Multi-agent PPO (JAX). One network is shared by everyone, conditioned on their own sex, age and
traits (nature) and trained on their own experience (nurture). The critic sees the whole town
during training; each person acts only on what they can perceive: the map, nearby people and what
they visibly carry, and their own memories and opinions. Unborn and dead slots are masked out.

```bash
pip install -r requirements.txt
python scripts/train_society.py --minutes 240 --out runs/society       # train + record replays
python scripts/build_site.py --run runs/society --v1 runs/economy --out site
python -m http.server -d site 8000                                     # watch at localhost:8000
```

World knobs (map, diet, lifecycle, social rules, happiness weights) are in `civ/society.py`
(`SocietyConfig`). The v1 economy is still in `civ/economy.py` / `scripts/train_economy.py`.

## Layout

```
civ/society.py          Society v2: vectorised town of whole lives, voluntary dealings only
civ/record_society.py   records 20-year stretches of a world for the viewer
civ/economy.py          v1: warehouse, farmer, police (everyone learned to steal)
civ/mappo.py            multi-agent PPO with alive masking, numpy policy runtime
scripts/                training, site building
viewer3d/index.html     the 3D Society viewer (GitHub Pages home)
viewer3d/economy.html   the 3D v1 viewer; viewer/viewer.html is the flat v1 map
```

## Roadmap

1. **Employers and firms as agents**, so wages are paid by someone who chooses to hire.
2. **More goods** (tools, seeds, housing) and trade between them; prices from supply and demand.
3. **Collective choices**: people who pool money into shared projects (a granary, a wall) only when
   each decides it is worth it.
4. **Embodiment**: drive physical humanoid bodies (learned walking, running, getting up, carrying)
   from [malecns-test](https://github.com/neomorrison/malecns-test), where a learned controller
   already runs a 23-joint humanoid in MuJoCo with sim-to-real training. The maleCNS fly-brain
   controller from that project can optionally act as an instinct layer (escape, pursuit).
