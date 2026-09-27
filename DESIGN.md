# Society v2 — design

v1 taught us something: agents in a world where money is the only lasting value, victims have no
recourse and nobody's opinion matters learn to steal - farmer and police included. That is not a
training failure; it is what that world rewards. Human cooperation rests on a handful of structural
facts about our lives. v2 builds those facts into the world and then lets behaviour be learned.

## What makes humans cooperate, and how each enters the world

| Mechanism | Why it sustains cooperation | In the simulation |
|---|---|---|
| **Kinship** (Hamilton 1964) | Helping relatives helps copies of your genes | Two sexes, partnerships, children who depend on care. Happiness includes the wellbeing of partner and children; a relative's death is grief. |
| **Direct reciprocity** (Trivers 1971; Axelrod 1984) | Cheating someone you will meet again costs you later | Long lives in one continuous world; every agent keeps a private opinion of every other, updated by how they were treated. |
| **Indirect reciprocity and gossip** (Nowak & Sigmund 1998; Dunbar 1996) | Strangers cooperate when bad deeds become known | Deeds are seen by bystanders. A *talk* action moves the listener's opinions toward the speaker's, so reputations spread beyond witnesses. |
| **Partner choice and ostracism** (Noë & Hammerstein 1994) | Cheaters are excluded from trade and mating | Every trade, gift and partnership needs the other side to accept; anyone can *shun* someone, refusing all dealings with them. |
| **Social selection through mate choice** (Miller 2000; Nesse 2007) | Kindness, reliability and provisioning are attractive | Attractiveness combines health, provisioning, reputation, kindness shown to the chooser, shared history, age and a heritable charm trait. Prosocial agents find partners and raise more children. |
| **Costly punishment and institutions** (Fehr & Gächter 2002; Ostrom 1990) | Norm violators are sanctioned | Enlisted police can detain a thief they saw; two or more witnesses together can make a citizen's arrest, including of a corrupt officer. Nobody pays police unless they choose to. |
| **Positive-sum cooperation** (Smith 1776; Tomasello 2009) | Working together produces more than working alone | Crates shipped while others are shipping earn a team bonus; harvests yield more with helpers on neighbouring plots; food spoils, so sharing beats hoarding. |
| **Needs beyond money** (Maslow 1943; Diener; Easterlin) | Happiness saturates with wealth and depends on belonging | Reward is happiness: fed and healthy, safe, belonging (friends, partner, thriving children), respected, and doing useful work. Money counts only through what it secures, with diminishing value. |
| **Evolution of temperament** (Boyd & Richerson 1985) | Dispositions spread when they leave more descendants | Heritable traits (empathy, greed, boldness, charm) pass to children with mutation. One shared policy learns within lifetimes; traits evolve across generations. |

## World

A 24 × 18 grid: warehouse (shelves, loading dock), grain fields (18 plots), an orchard (10 fruit
trees), a pasture with a herd (12 milking spots), market square, police station and six family homes. Time: 60 steps = one year. Lives run from birth to about 70; a world run
lasts several generations.

**Bodies.** Hunger rises each step; eating lowers it. Sustained starvation erodes health and kills.

**Diet.** There are three foods (grain, fruit, dairy), each feeding its own nutrient store. A meal
satisfies hunger in proportion to how much the body lacks that nutrient (sensory-specific
satiety), so a third bowl of the same food barely helps. A body missing any nutrient cannot heal
and slowly sickens; a one-food diet kills in a few years (recorded as malnutrition). A dairy
farmer therefore needs grain and fruit from someone, which is what makes trade, sharing and
specialisation worth it. Each person gets better at producing whatever they produce often.

**Homes.** Resting at your own home heals faster (shelter). Partners who have both slept at home
recently live together, and only couples who live together conceive. Children live in their
mother's home; grown children who pair up take a free house or move in with family.
Old age raises the chance of death each year. Pregnancy lasts 45 steps and raises the mother's
hunger rate. Children under 12 cannot work or trade and depend on food handed to them.

**Work.** Adults ship crates (5¢ each from the outside world, +50% when someone else shipped in the
last few steps), farm (plant grain for 1¢, pick fruit, milk the herd; more with skill and with helpers on
neighbouring plots), or enlist as police at the station. Food spoils.

**Dealings with a neighbour** (any of the four nearest agents within reach): give food, give a
coin, buy food at their asking price, take cash (theft), talk (gossip, friendship), court, shun or
unshun, and detain (police, or two witnesses together). Receiving requires not being shunned by the
receiver.

**Partnership and children.** Two adults of opposite sex who court each other within a few steps,
are not close relatives and each find the other attractive enough become partners (a partner can
leave by shunning). Fed, fertile partners who live together conceive with some probability; the
child is born at home into a free slot, inheriting the average of its parents' traits plus
mutation.

**Opinion and reputation.** Each agent holds an opinion of every other in [-1, 1]. Being robbed:
large drop. Seeing a theft: drop. Receiving a gift or fair trade: rise. Talking: the listener's
opinions move toward the speaker's (weighted by how much the listener trusts the speaker).

## Happiness (the reward)

```
happiness = sustenance (fed, healthy)
          + safety     (not robbed, not detained)
          + belonging  (friends, partner, children's wellbeing weighted by empathy)
          + esteem     (how others regard you)
          + purpose    (productive work)
          + security   (log of savings, weighted by greed)
          - grief      (a partner or child dies)
          - death
```

## Learning

Multi-agent PPO with one shared network for everyone, conditioned on the agent's own traits, sex
and age: nature (traits) plus nurture (learning). Dead or unborn slots are masked out of training.
Traits are inherited and mutate, so the temperament mix of the population evolves over generations.
