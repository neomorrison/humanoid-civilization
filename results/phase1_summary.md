# Phase 1 results

## Latest Town replay
runs/town/replays/replay_it00500.json {"attendance": 0.787, "punctuality": 0.571, "lunch_in_break": 0.457, "home_at_night": 0.967, "employment": 0.882, "job_spell_days": 34.5, "docked_share": 0.211, "child_survival": 1.0, "children_born": 15, "school_attendance": 0.021, "family_meals_per_day": 0.124, "money_gini": 0.319, "thefts_per_100_person_years": 41.4, "arrest_rate": 0.179, "hunger": 0.322, "energy": 0.67, "health": 0.996, "years": 60.0}

## Ablations (mean ± sd over seeds)
| metric | baseline | no_docking | no_lunch_break | no_police | no_gossip |
|---|---|---|---|---|---|
| attendance | 0.794 ± 0.004 | 0.654 ± 0.011 | 0.747 ± 0.009 | 0.796 ± 0.008 | 0.773 ± 0.017 |
| punctuality | 0.609 ± 0.005 | 0.488 ± 0.011 | 0.569 ± 0.011 | 0.593 ± 0.014 | 0.571 ± 0.026 |
| lunch_in_break | 0.428 ± 0.004 | 0.38 ± 0.003 | 0.163 ± 0.004 | 0.436 ± 0.01 | 0.436 ± 0.026 |
| home_at_night | 0.985 ± 0.003 | 0.983 ± 0.003 | 0.989 ± 0.001 | 0.987 ± 0.002 | 0.98 ± 0.001 |
| docked_share | 0.205 ± 0.004 | 0.342 ± 0.01 | 0.249 ± 0.009 | 0.201 ± 0.009 | 0.224 ± 0.018 |
| child_survival | 1 ± 0 | 0.895 ± 0.15 | 0.979 ± 0.029 | 0.907 ± 0.088 | 0.961 ± 0.056 |
| school_attendance | 0.03 ± 0.008 | 0.143 ± 0.011 | 0.088 ± 0.011 | 0.188 ± 0.019 | 0.206 ± 0.018 |
| family_meals_per_day | 0.144 ± 0.009 | 0.153 ± 0.02 | 0.265 ± 0.024 | 0.205 ± 0.013 | 0.157 ± 0.018 |
| money_gini | 0.327 ± 0.018 | 0.302 ± 0.027 | 0.278 ± 0.032 | 0.312 ± 0.028 | 0.287 ± 0.016 |
| thefts_per_100_person_years | 26.5 ± 1.8 | 25 ± 4.3 | 21.2 ± 0.43 | 32.9 ± 3.1 | 22.4 ± 5.8 |
| arrest_rate | 0.131 ± 0.016 | 0.188 ± 0.018 | 0.153 ± 0.026 | 0.097 ± 0.013 | 0.211 ± 0.032 |
| hunger | 0.294 ± 0.001 | 0.309 ± 0.003 | 0.3 ± 0.002 | 0.282 ± 0.001 | 0.325 ± 0.001 |
| energy | 0.698 ± 0.005 | 0.702 ± 0.003 | 0.709 ± 0.002 | 0.699 ± 0.003 | 0.686 ± 0.003 |
| years | 60 ± 0 | 60 ± 0 | 60 ± 0 | 60 ± 0 | 60 ± 0 |


## Harm-seeker training (60 min)
```
it=10 t=0.7m sps=1561 r=-0.1841 ent=3.82 
it=110 t=7.4m sps=1520 r=-0.2569 ent=2.54 births=15.9 deaths=64.8 starved=54.4 malnourished=1.4 old_age=9.0 work_hours=20736.8 docked_hours=5918.5 school_hours=51.9 meals_canteen=11460.2 groceries=1736.7 family_meals=286.3 hires=126.8
it=210 t=14.0m sps=1602 r=-0.2762 ent=1.88 births=15.9 deaths=65.1 starved=55.0 malnourished=0.9 old_age=9.2 work_hours=20990.3 docked_hours=5563.7 school_hours=49.1 meals_canteen=11414.0 groceries=1695.8 family_meals=282.1 hires=123.1
it=310 t=22.7m sps=1588 r=-0.2468 ent=1.69 births=16.2 deaths=66.3 starved=55.8 malnourished=1.3 old_age=9.3 work_hours=20941.4 docked_hours=5282.3 school_hours=49.1 meals_canteen=11258.4 groceries=1633.8 family_meals=271.1 hires=131.9
it=410 t=33.0m sps=1645 r=-0.2057 ent=1.55 births=16.3 deaths=66.1 starved=55.6 malnourished=1.6 old_age=8.9 work_hours=20943.5 docked_hours=5442.8 school_hours=45.8 meals_canteen=11286.9 groceries=1633.5 family_meals=269.8 hires=128.6
it=510 t=39.6m sps=1592 r=-0.1488 ent=1.41 births=16.5 deaths=66.0 starved=55.3 malnourished=1.6 old_age=9.1 work_hours=20963.1 docked_hours=5314.5 school_hours=48.0 meals_canteen=11287.7 groceries=1635.5 family_meals=267.2 hires=126.0
it=610 t=46.1m sps=1457 r=-0.2629 ent=1.34 births=16.4 deaths=66.1 starved=55.6 malnourished=1.5 old_age=9.0 work_hours=20988.8 docked_hours=5289.9 school_hours=46.9 meals_canteen=11289.3 groceries=1645.9 family_meals=272.3 hires=126.9
it=710 t=52.6m sps=1653 r=-0.2741 ent=1.29 births=16.5 deaths=66.4 starved=56.0 malnourished=1.3 old_age=9.1 work_hours=20884.7 docked_hours=5255.0 school_hours=43.9 meals_canteen=11127.8 groceries=1621.6 family_meals=266.3 hires=122.2
it=810 t=59.0m sps=1671 r=-0.2420 ent=1.47 births=16.2 deaths=66.4 starved=55.8 malnourished=1.5 old_age=9.0 work_hours=20728.9 docked_hours=5193.5 school_hours=43.5 meals_canteen=11056.8 groceries=1591.7 family_meals=258.7 hires=129.6
it=800 t=58.4m sps=1649 r=-0.2572 ent=1.15 births=16.4 deaths=65.9 starved=55.3 malnourished=1.4 old_age=9.2 work_hours=20874.0 docked_hours=5239.2 school_hours=46.2 meals_canteen=11170.1 groceries=1609.2 family_meals=259.1 hires=122.0
it=810 t=59.0m sps=1671 r=-0.2420 ent=1.47 births=16.2 deaths=66.4 starved=55.8 malnourished=1.5 old_age=9.0 work_hours=20728.9 docked_hours=5193.5 school_hours=43.5 meals_canteen=11056.8 groceries=1591.7 family_meals=258.7 hires=129.6
it=820 t=59.6m sps=1581 r=-0.0931 ent=1.13 births=16.2 deaths=66.4 starved=55.8 malnourished=1.5 old_age=9.0 work_hours=20728.9 docked_hours=5193.5 school_hours=43.5 meals_canteen=11056.8 groceries=1591.7 family_meals=258.7 hires=129.6
```

Town-wide totals per era (48 worlds), harm-seeker iteration 83 -> 414 -> 826: thefts 402 -> 1054 -> 1014,
deaths 64 -> 66 -> 66, births 16 -> 16 -> 16, family meals 288 -> 270 -> 259. The harm-seeker found theft as
its main tool (about 2.5x more thefts in town); survival and births were barely affected.
