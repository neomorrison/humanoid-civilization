# Wild: a world where people perceive, invent, build and talk on their own (design v2)

Town Life gives people named places, jobs and actions. Wild gives them only a body, senses,
materials that obey simple physics, and each other. Nothing in it is named for them: no
recipe list, no "tool", no "house", no words. Whatever they make, build or say is theirs.
It is written in JAX so thousands of worlds run in parallel (CPU now, GPU unchanged).

## Bodies and feelings
Each person keeps a body in balance. Internal variables, each with a set point:

| variable | falls | restored by |
|---|---|---|
| satiety | every hour | eating (more from food that was heated) |
| hydration | every hour, faster when hot | drinking at water, or from a container |
| warmth | at night, in winter, in wind and rain | fire, enclosure, huddling |
| energy | while awake | sleep (better when enclosed and warm) |
| health | injury, or any variable critical | slowly, when the rest are fine |

These are felt directly (interoception). Reward is homeostatic: the reduction in the
weighted distance of the body from its set points, plus a birth, minus death. The weights
are **genes** (inner values): children inherit their parents' average with mutation, so
what people care about evolves by who survives and has children. Social values (company,
children's wellbeing) and **curiosity** (a bonus for surprising the person's own world
model, i.e. for finding out something new) are genes too, starting small.

## Senses (no labels)
* **Sight**: an egocentric window of cells. Each cell is seen as physical qualities, never
  as a name: ground (height, wetness, fertility), light and heat, and for whatever lies there
  its material properties (below), plus people (size, facing, injured, kin smell, what they
  hold). A berry is "small, soft, nutritious, red"; the agent learns what that means.
* **Hearing**: signals from people nearby, with direction and loudness.
* **Touch**: the properties of what is in each hand.
* **Interoception**: the body variables above; time of day felt as light and cold.

Policies see through a small convolutional encoder and a recurrent memory that lasts a
lifetime, because each person sees little and must remember where things are.

## Materials instead of recipes
Every object is a vector of physical properties:
`mass, length, hardness, sharpness, flexibility, flammability, nutrition, moisture,
capacity, heat`. Nature supplies a few kinds (stones, branches, fibres, berries, clay,
water, animals), each with noisy properties. Everything else comes from a handful of
physical operations, each a fixed, simple rule on properties:

| primitive | rule (sketch) |
|---|---|
| `strike` held thing against thing on the ground | the softer one breaks: pieces get lower mass and a sharp edge proportional to the harder one's hardness |
| `bind` the two held things | one object: masses and lengths add, strength is the weaker part's, sharpness is the sharper end's, capacity if something flexible surrounds something hollow |
| `heat` a held thing at a fire | moisture falls, nutrition becomes digestible, clay hardens, flammable things burn (a fire) |
| `place` / `take` | put an object into the world or pick one up |

What objects *do* also follows from properties: cutting and digging speed from sharpness
and hardness, reach from length, carrying from capacity, fire from flammability, food from
nutrition. So a hand axe, a spear, a basket or a pot is not in the code. It is a
combination of properties that happens to help, and anything else that helps counts too.
Inventions are found afterwards by clustering the property vectors people made and
measuring what each cluster was used for.

## Building
Placed objects stay where they are. Cells filled with long, hard things block wind and
movement; a cell mostly enclosed by them is sheltered (warmer, drier, better sleep).
Objects left in a place are a store others can find. Structures outlive their builders,
so later generations inherit a changed world: the first form of culture.

## Speaking
Each hour a person may emit one of V arbitrary tokens, heard within a radius with its
direction. Nothing gives tokens meaning. Everything is logged for translation: the
situation around each token (what the speaker sees, holds, feels, is about to do) and
what listeners do next. A token's English gloss is the situation that best predicts it
and the behaviour it causes; sequences are glossed as phrases. Cultural transmission is
measured by whether dialects and techniques persist after the people who started them die.

## Actions
`stay, move ×4, turn, grab, place, eat, drink, strike, bind, heat, give, sleep, mate,
signal k`. That is all: everything else is a sequence of these the person learned.

## Measures
Per era: which property combinations were made and used (the invention timeline), who
made them first and whether the practice outlived them, structures standing, sharing
networks, token vocabulary and glosses, evolved inner values, population and lifespan.

## What this does not settle
Whether anything like this could be sentient is an open question that no measurement
here answers. The world is built so that the abilities associated with minds (perception,
memory, feelings that matter to the body, curiosity, invention, language, culture) have
to be learned by the agents themselves, and so that we can see when they are.
