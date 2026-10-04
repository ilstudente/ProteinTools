#!/usr/bin/env bash
# Serial one-time test evaluations of the three checkpoints, run only after the
# GRPO checkpoint has been selected on validation. Each dispatch waits for its
# result before the next one starts, so no two GPU jobs overlap.
set -euo pipefail
cd "$(dirname "$0")/.."
MODAL=${MODAL:-.venv/bin/modal}
for target in "" "/experiment/runs/protein-detective-v1/sft/final" "/experiment/runs/protein-detective-v1/grpo/best"; do
  label=$( [ -z "$target" ] && echo original || basename "$(dirname "$target")" )
  if [ -f "runs/protein-detective-v1/evaluation-${label}-test/summary.json" ]; then
    echo "SKIP ${label}: test summary already present"; continue
  fi
  echo "START ${label} $(date -u +%H:%M:%SZ)"
  if [ -z "$target" ]; then
    "$MODAL" run --detach modal_app.py --stage evaluation --split test > "runs/evaluation-${label}-test-dispatch.log" 2>&1
  else
    "$MODAL" run --detach modal_app.py --stage evaluation --split test --checkpoint "$target" > "runs/evaluation-${label}-test-dispatch.log" 2>&1
  fi
  grep -h '"success"\|"error"' "runs/evaluation-${label}-test-dispatch.log" | head -3 | sed "s/^/RESULT ${label}: /"
done
echo "DONE test evaluations"
