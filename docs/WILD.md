# Wild: a world where the solutions must be invented (design)

Phases 3–5 of the roadmap need a different world from Town Life: nothing named, nothing
scripted, only bodies, materials and each other. It is written in JAX from the start so
thousands of worlds can run in parallel (on a CPU now, a GPU later).

## Bodies and drives (Phase 3)
Each person keeps a body in balance. Internal variables, each with a set point:

| variable | falls | restored by |
|---|---|---|
| satiety | every hour | eating |
| hydration | every hour, faster in heat | drinking at water |
| warmth | at night, in winter, faster in rain | fire, shelter, huddling with others |
| energy | while awake | sleep (better in shelter) |
| health | when any of the above is critical, injury | slowly, when the rest are fine |

Reward is homeostatic: the reduction in the weighted distance of the body from its set
points, plus reproduction (a birth) and minus death. The weights are the person's inner
values; they are **genes**: children inherit the average of their parents' with mutation,
so what people care about evolves by who survives and has children. Social rewards
(company, children's wellbeing) are also genes that start near zero and can grow if they
pay off across generations.

Infants depend on others: they cannot move far, gather or drink alone for their first
years. Reproduction needs two adults who both choose it, fed enough, not too old.

## Primitive actions (Phase 4)
`stay, move N/S/E/W, grab, drop, eat, drink, combine, give, sleep, mate, signal k`

* `grab` takes one unit of whatever is on your cell (berries from a bush, a branch from a
  tree, a stone, fibre from tall grass, an item someone dropped, a carcass).
* `combine` merges the two items in your hands. Recipes are fixed by the world but never
  shown: stone + stone → sharp stone (cuts: more from each grab), branch + fibre → rope,
  branch + branch on your cell → fire (warmth, cooks), berries/meat + fire → cooked food
  (more satiety per bite), branches + rope → shelter piece (dropped pieces next to each
  other form a shelter: warmth, better sleep), container from fibre (carry more, store).
* `give` passes what is in your hand to the person in front of you.
* `signal k` emits one of V arbitrary tokens heard by everyone within a few cells (Phase 5).

## World
A grid with water, woods, stony ground, berry scrub and grassland; seasons (berries in
summer and autumn, cold winters), day and night, weather. Small animals wander and can
be caught (more than one person catching together succeeds more often).

## Observation
An egocentric window (e.g. 7×7) of terrain, resources, items, fires, shelters and people
(with kin, sex, age and the signal they emitted), plus the body's own state, hands and
the time of day / season. No map, no names: memory (recurrent policies) and
communication are useful because each person sees little.

## Measuring invention
Everything is logged per era: which recipes were ever made (a tech tree timeline), how
often, by whom and whether the knowledge persisted after its inventor died (cultural
transmission), which signals were used where (for translation), who gave what to whom
(sharing networks), and the evolved inner values.
