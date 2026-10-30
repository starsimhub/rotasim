# VM Run Instructions

This document provides instructions for running the calibration scripts on a virtual machine.

## Files Ready to Run

### 1. calibrate_hybrid_50trials.py
**Purpose:** Run 50-trial calibration with new GOF metric

**What it does:**
- Runs 50 calibration trials using Optuna optimization
- Uses the new GOF metric: `GOF = 10 * GOF_age + GOF_incidence`
- Calibrates 8 parameters: reporting_rate, base_beta, beta0, beta1, beta2, sus_after_1, sus_after_2, sus_after_3plus
- Each trial takes ~3-5 minutes, total runtime ~2.5-4 hours

**Outputs:**
- `rota_hybrid_50trials.db` - Optuna database with all trial results
- `calibration_50trials_results.json` - Summary of best parameters and top 5 trials
- Console output showing progress and final results

**How to run:**
```bash
cd /path/to/rotasim/calibration
python calibrate_hybrid_50trials.py
```

### 2. run_trial49_50sims.py
**Purpose:** Run 50 independent simulations with Trial #49 parameters (best fit)

**What it does:**
- Runs 50 simulations using the best-fit parameters from trial #49
- Uses different random seeds for each simulation to assess uncertainty
- Computes incidence and age distribution for each run
- Provides summary statistics (mean, SD, min, max, quartiles)
- Each simulation takes ~2-3 minutes, total runtime ~1.5-2.5 hours

**Trial #49 Parameters:**
- reporting_rate: 0.003393
- base_beta: 4.5363
- Age model: beta0=-0.9843, beta1=0.2582, beta2=-0.0085
- Susceptibility: 0.7997/0.7137/0.6316 (after 1/2/3+ infections)
- Expected incidence: ~20.27 per 100k (74% of target 27.56)

**Outputs:**
- `trial49_50sims_results.csv` - Detailed results for all 50 simulations
- `trial49_50sims_summary.json` - Summary statistics (mean, SD, quartiles)
- Console output showing progress and summary

**How to run:**
```bash
cd /path/to/rotasim/calibration
python run_trial49_50sims.py
```

## Prerequisites

### Required Python Packages
Both scripts require:
- numpy
- pandas
- sciris
- starsim
- rotasim
- optuna (for calibration script only)

### Required Files in Same Directory
- `process_incidence_uk_age.py` - Data processing utilities
- `uk_age_data.csv` - UK age distribution data

## Running on VM

### Step 1: Transfer Files
Copy these files to your VM:
```bash
scp calibrate_hybrid_50trials.py user@vm:/path/to/rotasim/calibration/
scp run_trial49_50sims.py user@vm:/path/to/rotasim/calibration/
```

### Step 2: Verify Dependencies
```bash
# SSH into VM
ssh user@vm

# Check Python environment
python -c "import numpy, pandas, sciris, starsim, rotasim, optuna; print('All dependencies OK')"
```

### Step 3: Run Scripts

**Option A: Run calibration (50 trials, ~2.5-4 hours)**
```bash
cd /path/to/rotasim/calibration
python calibrate_hybrid_50trials.py 2>&1 | tee calibration_50trials_output.txt
```

**Option B: Run trial #49 simulations (50 runs, ~1.5-2.5 hours)**
```bash
cd /path/to/rotasim/calibration
python run_trial49_50sims.py 2>&1 | tee trial49_50sims_output.txt
```

**Option C: Run both in parallel (if VM has enough resources)**
```bash
# In separate terminal sessions:
# Session 1:
python calibrate_hybrid_50trials.py 2>&1 | tee calibration_50trials_output.txt

# Session 2:
python run_trial49_50sims.py 2>&1 | tee trial49_50sims_output.txt
```

### Step 4: Monitor Progress
Both scripts print progress to console:
- Calibration: Shows trial number and GOF after each trial
- Simulations: Shows simulation number and incidence after each run

To monitor from another terminal:
```bash
# For calibration:
tail -f calibration_50trials_output.txt

# For simulations:
tail -f trial49_50sims_output.txt
```

### Step 5: Retrieve Results
After completion, copy results back:
```bash
# Calibration results:
scp user@vm:/path/to/rotasim/calibration/rota_hybrid_50trials.db .
scp user@vm:/path/to/rotasim/calibration/calibration_50trials_results.json .

# Simulation results:
scp user@vm:/path/to/rotasim/calibration/trial49_50sims_results.csv .
scp user@vm:/path/to/rotasim/calibration/trial49_50sims_summary.json .
```

## Expected Outputs

### Calibration (calibrate_hybrid_50trials.py)
```
Best Trial: #XX
Best GOF: X.XXXX

Best Parameters:
  reporting_rate      : X.XXXXXX
  base_beta           : X.XXXX
  beta0               : X.XXXX
  beta1               : X.XXXX
  beta2               : X.XXXX
  sus_after_1         : X.XXXXXX (XX.XX% protection)
  sus_after_2         : X.XXXXXX (XX.XX% protection)
  sus_after_3plus     : X.XXXXXX (XX.XX% protection)

Top 5 Trials:
#XX: GOF = X.XXXX
...
```

### Simulations (run_trial49_50sims.py)
```
Incidence (per 100,000):
  Target: 27.56
  Mean:   XX.XX (SD: X.XX)
  Median: XX.XX
  Min:    XX.XX
  Max:    XX.XX

Age Distribution (mean proportions):
  0-11 months    :  XX.XX% (SD: X.XX%) [Target: 13.77%]
  12-23 months   :  XX.XX% (SD: X.XX%) [Target: 27.67%]
  24-59 months   :  XX.XX% (SD: X.XX%) [Target: 46.89%]
  5+ years       :  XX.XX% (SD: X.XX%) [Target: 11.67%]

GOF Distribution:
  Mean:   X.XXXX (SD: X.XXXX)
  Median: X.XXXX
  Min:    X.XXXX
  Max:    X.XXXX
```

## Troubleshooting

### Issue: "No module named 'optuna'"
```bash
pip install optuna
```

### Issue: "FileNotFoundError: process_incidence_uk_age.py"
Ensure you're running from the calibration directory and all required files are present.

### Issue: "FileNotFoundError: uk_age_data.csv"
Copy the UK age data file to the calibration directory.

### Issue: Script appears frozen
Both scripts can take several hours. Check if they're still running:
```bash
ps aux | grep python
```

Monitor output file for progress:
```bash
tail -f calibration_50trials_output.txt
# or
tail -f trial49_50sims_output.txt
```

## File Locations

After running the scripts, you'll find the output files in:
```
/path/to/rotasim/calibration/
├── calibrate_hybrid_50trials.py          # Calibration script
├── run_trial49_50sims.py                 # Simulation script
├── rota_hybrid_50trials.db               # Calibration database (after running)
├── calibration_50trials_results.json     # Calibration summary (after running)
├── trial49_50sims_results.csv            # Simulation results (after running)
├── trial49_50sims_summary.json           # Simulation summary (after running)
├── calibration_50trials_output.txt       # Console output (if using tee)
└── trial49_50sims_output.txt             # Console output (if using tee)
```

## Contact

If you encounter any issues, check:
1. All dependencies are installed
2. You're in the correct directory
3. Required data files are present
4. VM has sufficient memory (recommend 8GB+ RAM)
5. VM has sufficient disk space (recommend 10GB+ free)
