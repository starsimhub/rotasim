#!/usr/bin/env bash
# Unattended multi-wave history matching on capybara, resilient to spot reclamation.
# Runs up to 8 waves in one engine.run(); if the process dies (reclaim/crash), the
# loop resumes from the last wave checkpoint. Clear outputs/hm ONCE before launching
# for a fresh run (resume relies on the checkpoint persisting across attempts).
set -u
cd ~/GIT/rotasim
export PATH="$HOME/.local/bin:$PATH"
export HM_WORKERS="${HM_WORKERS:-118}"
for a in 1 2 3 4 5 6; do
  if [ "$a" -eq 1 ]; then FLAG=""; else FLAG="--resume"; fi
  echo "=== HM attempt $a (flag='$FLAG') $(date) ==="
  uv run --python 3.13 python experiments/07_history_matching/run_wave.py \
      --max-iter 8 --n-samples 2000 $FLAG && { echo "HM COMPLETED OK"; break; }
  echo "attempt $a interrupted (exit $?); resuming from checkpoint in 30s..."
  sleep 30
done
