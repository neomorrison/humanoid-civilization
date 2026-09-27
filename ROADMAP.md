# Roadmap: from hand-made routines to agents that invent their own

The goal is a society whose behaviour is *discovered*, not written: agents that learn to act,
build skills, compose them, find social strategies, invent communication, pass behaviour on
to their children, live in physical bodies, and eventually drive real robots. The world
should stay open to change at any moment, so we can watch how they cope.

Where we are: hand-coded behaviour (v1 scripts) → hand-coded high-level actions with a
learned policy choosing among them (v2 Society, v3 Town Life).

## Phase 1 — Town Life, measured (in progress)
- Finish training v3 (work, lunch, home, sleep, families, school) and publish it with the
  new 3D viewer.
- Behaviour metrics for every replay (`civ/metrics.py`): routine, work, family, inequality,
  justice, wellbeing.
- Controlled comparisons (`scripts/ablate_town.py`): switch one ingredient off (docking,
  lunch break, police, gossip) and measure what changes.
- Warm start (`civ/warmstart.py`): changing what agents sense or do keeps what they learned.

## Phase 2 — Fast, remembering, and live
- **Speed**: remove the per-person Python loops; move the simulator to JAX so thousands of
  towns step in parallel.
- **Memory**: recurrent policies, so a person can remember, adapt within a lifetime and
  react to things they were never trained on.
- **Live sandbox**: run the simulator and the trained policy in the browser. Intervene at
  any moment: remove a person and watch the grief ripple through a family, drop in an agent
  whose only goal is harm, change the rules (wages, docking, prices, seasons) and watch
  them adapt. Training can also run continuously and re-read the rules, so changes made
  mid-run are learned from.

## Phase 3 — Fewer assumptions about a good life
- Replace the hand-written happiness equation with what evolution works with: stay alive,
  keep the body in balance (hunger, thirst, temperature, energy, pain) and reproduce.
- Make each person's inner rewards heritable: the weights they put on company, children,
  status and so on are genes that mutate and are selected by survival and reproduction.
  Learning shapes behaviour within a life; selection shapes what people want across lives.
- The world must then carry more of the shaping: scarcity, seasons, danger and disease,
  helpless infants, and tasks only groups can do, so that cooperation, care and trust pay
  for themselves.

## Phase 4 — Invention
- Primitive actions only: move, grab, drop, combine, use. Materials and recipes instead of
  named jobs: nothing says "harvest" or "farm"; tools, food preparation, shelter and storage
  have to be discovered.
- Learned skills from primitives (goal-conditioned policies that reach self-chosen goals),
  and a learned layer that composes them.
- Division of labour, trade and employment should emerge as strategies (an owner is simply
  someone who pays others in goods to work their land) rather than being written in.

## Phase 5 — Communication and culture
- A channel of discrete signals between people nearby, with pressures that make sharing
  information pay (hidden food, danger, coordination).
- Reading it: every signal is logged with its context; a translator maps signals to English
  from how and when they are used (grounded probes, then a language model writes the gloss),
  shown in the viewer as "probably means: 'food here'". Early vocabularies will be small;
  grammar is a research frontier.
- Cultural transmission: children learn by watching adults (imitation within a lifetime),
  so skills and signals can outlive the people who invented them.

## Phase 6 — Bodies and the real world
- A library of physical skills in MuJoCo (walk, get up, reach, grasp, carry) learned from
  joint or muscle control, reusing the sim-to-real humanoid from
  [malecns-test](https://github.com/neomorrison/malecns-test).
- The social layer sends goals, the bodies carry them out, and the viewer shows real motion.
- Sim-to-real for the skill layer: the same controllers on a physical robot.

## Compute
This work currently runs on 4 CPU cores. Phases 1–3 and small versions of 4–5 fit that.
Physical skills and open-ended training at scale want a GPU: MuJoCo's accelerated version
(MJX) and JAX simulators run roughly 100-1000x faster there.
