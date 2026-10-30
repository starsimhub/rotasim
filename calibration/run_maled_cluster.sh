#!/bin/bash
#SBATCH --job-name=rota_maled
#SBATCH --output=maled_%x_%j.out
#SBATCH --error=maled_%x_%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

# SLURM launcher for MAL-ED birth-cohort calibration.
#
# Per-site setup:
#   - Bangladesh: birth 19, death 6 (per 1000/y), unvaccinated
#   - Pakistan:   birth 27, death 7 (per 1000/y), unvaccinated
#
# First-run recommendation: 10 trials, single site, single worker — ~3-4 hours
# wall time. Once the GOF space looks reasonable, scale to 50 trials or launch
# a SLURM array for multi-worker parallelism.
#
# Usage:
#   # Quick first read (Bangladesh, 10 trials):
#   sbatch --export=SITE=bangladesh,N_TRIALS=10 run_maled_cluster.sh
#
#   # Full production run (Bangladesh, 50 trials, sequential within one job):
#   sbatch --export=SITE=bangladesh,N_TRIALS=50 run_maled_cluster.sh
#
#   # 4-way SLURM array for faster wall time (each worker runs 13 trials,
#   # workers coordinate via the shared SQLite DB):
#   sbatch --array=1-4 --export=SITE=bangladesh,N_TRIALS=13,TOTAL_TRIALS=50 run_maled_cluster.sh

set -euo pipefail

# Defaults — overridden by `sbatch --export=...`.
SITE="${SITE:-bangladesh}"
N_TRIALS="${N_TRIALS:-10}"
TOTAL_TRIALS="${TOTAL_TRIALS:-$N_TRIALS}"
N_REPS="${N_REPS:-20}"
N_JOBS="${N_JOBS:-1}"   # parallel trials per worker (1 = sequential within worker)

# Activate your conda environment if needed
# source activate rotasim

# Change to calibration directory (adjust path for your cluster checkout)
cd /path/to/rotasim/calibration

DB_PATH="rota_maled_${SITE}.db"

echo "================================================================"
echo "MAL-ED CALIBRATION"
echo "  Site:               ${SITE}"
echo "  Trials this worker: ${N_TRIALS}"
echo "  Total trials:       ${TOTAL_TRIALS}"
echo "  Replicates / trial: ${N_REPS}"
echo "  Parallel trials:    ${N_JOBS}"
echo "  Worker ID:          ${SLURM_ARRAY_TASK_ID:-${SLURM_JOB_ID:-local}}"
echo "  Database:           ${DB_PATH}"
echo "  Started:            $(date)"
echo "================================================================"

python calibrate_maled.py \
    --site "${SITE}" \
    --n-trials "${N_TRIALS}" \
    --total-trials "${TOTAL_TRIALS}" \
    --n-reps "${N_REPS}" \
    --n-jobs "${N_JOBS}" \
    --db-path "${DB_PATH}"

echo "================================================================"
echo "Done: $(date)"
echo "================================================================"
