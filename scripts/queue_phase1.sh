#!/usr/bin/env bash
# Unattended Phase 1 queue: finish Town training at it 500, run ablations, train the harm-seeker,
# then write results/phase1_summary.md. Progress goes to runs/queue.log.
set -u
cd "$(dirname "$0")/.."
log=runs/queue.log
echo "$(date -u +%H:%M) waiting for town it 500" >> $log
until grep -q "replay it=500\]" runs/town_train.log; do sleep 60; done
ps -eo pid,args | awk '/train_town.py/ && !/awk/ {print $1}' | xargs -r kill
sleep 5
echo "$(date -u +%H:%M) ablations" >> $log
python3 scripts/ablate_town.py --base runs/town/checkpoint.pkl --iters 120 --seeds 3 >> runs/ablate.log 2>&1
echo "$(date -u +%H:%M) harm-seeker" >> $log
python3 scripts/train_evil.py --town runs/town/policy.npz --minutes 60 --out runs/evil >> runs/evil.log 2>&1
{
  echo "# Phase 1 results"; echo
  echo "## Latest Town replay"; python3 -m civ.metrics runs/town/replays/replay_it00500.json | cut -c1-2000; echo
  echo "## Ablations (mean ± sd over seeds)"; cat results/ablations.md; echo
  echo "## Harm-seeker training (last log line)"; tail -1 runs/evil/train_log.jsonl | cut -c1-600
} > results/phase1_summary.md 2>&1
echo "$(date -u +%H:%M) done" >> $log
