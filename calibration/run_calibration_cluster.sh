#!/bin/bash
#SBATCH --job-name=rota_calib
#SBATCH --output=calibration_%j.out
#SBATCH --error=calibration_%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=64G

# SLURM script for running calibration with parallel trials
# This uses Optuna's n_jobs to run multiple trials in parallel on a single node

# Activate your conda environment
# source activate rotasim  # Uncomment and modify as needed

# Change to calibration directory
cd /path/to/rotasim/calibration

# Run calibration with parallel trials
# --n-trials: Total number of calibration trials to run
# --n-jobs: Number of trials to run in parallel (use -1 for all CPUs, or specify a number)
# --n-reps: Number of simulation replicates per trial
# --n-cpus-per-trial: CPUs for MultiSim within each trial (leave empty to use all available)

python calibrate_hybrid_multisim.py \
    --n-trials 50 \
    --n-jobs 5 \
    --n-reps 20 \
    --db-path rota_hybrid_multisim.db

echo "Calibration complete!"
