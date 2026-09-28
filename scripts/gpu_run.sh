#!/usr/bin/env bash
# Train on a GPU machine and push the logs and checkpoints to a results branch, so the run
# survives the machine (Nautilus removes storage) and can be read from anywhere.
#   bash scripts/gpu_run.sh wild 600          # what to train, minutes
set -euo pipefail
cd "$(dirname "$0")/.."
what=${1:-wild}; minutes=${2:-600}
tag="$what-$(hostname)-$(date -u +%Y%m%d-%H%M)"; out="runs/$tag"
case $what in
  wild) cmd="python3 -m civ.ppo_wild --minutes $minutes --out $out" ;;
  town) cmd="python3 scripts/train_town.py --minutes $minutes --out $out" ;;
  *) echo "unknown: $what"; exit 1 ;;
esac
mkdir -p "$out"; $cmd > "$out/stdout.log" 2>&1 &
pid=$!
sync_results() {
  git worktree add -q -B "results/$tag" "/tmp/$tag" HEAD 2>/dev/null || true
  mkdir -p "/tmp/$tag/$out"
  rsync -a --exclude 'replays*' --exclude 'snapshots' "$out/" "/tmp/$tag/$out/"
  (cd "/tmp/$tag" && git add -f "$out" && git commit -qm "results: $tag" && git push -qf origin "results/$tag") || true
}
while kill -0 $pid 2>/dev/null; do sleep 900; sync_results; done
sync_results; echo "done: branch results/$tag"
