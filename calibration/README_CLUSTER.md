# Running Calibration on Cluster

## Parallelization Strategy

The calibration script now supports **two levels of parallelization**:

### 1. Across Trials (via Optuna's `n_jobs`)
- Multiple calibration trials run simultaneously
- Controlled by `--n-jobs` parameter
- Best for: Taking advantage of multiple CPUs on a single node

### 2. Within Trials (via MultiSim)
- The 20 simulation replicates within each trial run in parallel
- Controlled by `--n-cpus-per-trial` parameter (default: all available)
- Always active by default

## Usage Examples

### Single Node with Parallel Trials

Run 50 trials with 5 trials in parallel:
```bash
python calibrate_hybrid_multisim.py --n-trials 50 --n-jobs 5 --n-reps 20
```

Run 50 trials using all available CPUs for parallel trials:
```bash
python calibrate_hybrid_multisim.py --n-trials 50 --n-jobs -1 --n-reps 20
```

### SLURM Cluster

Submit the job:
```bash
sbatch run_calibration_cluster.sh
```

Or customize:
```bash
sbatch --cpus-per-task=32 run_calibration_cluster.sh
```

### Multi-Node (Advanced)

For running across multiple nodes, launch separate workers that share the same database:

**Node 1:**
```bash
python calibrate_hybrid_multisim.py --n-trials 25 --n-jobs 5 --worker-id node1
```

**Node 2:**
```bash
python calibrate_hybrid_multisim.py --n-trials 25 --n-jobs 5 --worker-id node2
```

All workers will coordinate via the shared SQLite database.

## Resource Recommendations

### Conservative (Avoid Resource Contention)
```bash
python calibrate_hybrid_multisim.py \
    --n-trials 50 \
    --n-jobs 4 \
    --n-reps 20 \
    --n-cpus-per-trial 5
```
- 4 trials run in parallel
- Each trial uses 5 CPUs for its 20 replicates
- Total: ~20 CPUs utilized

### Aggressive (Maximum Parallelism)
```bash
python calibrate_hybrid_multisim.py \
    --n-trials 50 \
    --n-jobs -1 \
    --n-reps 20
```
- Uses all CPUs for parallel trials
- Each trial uses all CPUs for replicates
- May cause resource contention but maximizes throughput

### Recommended for 32-CPU Node
```bash
python calibrate_hybrid_multisim.py \
    --n-trials 50 \
    --n-jobs 8 \
    --n-reps 20 \
    --n-cpus-per-trial 4
```
- 8 trials in parallel
- Each trial uses 4 CPUs for simulations
- Total: 32 CPUs fully utilized with good balance

## Monitoring Progress

Check log files:
```bash
tail -f calibrate_hybrid_multisim_worker*_*.log
```

Check database status:
```bash
sqlite3 rota_hybrid_multisim.db "SELECT COUNT(*) FROM trials WHERE state='COMPLETE';"
```

## Command Line Options

- `--n-trials`: Number of trials to run (default: 50)
- `--n-jobs`: Parallel trials via Optuna (default: 1, use -1 for all CPUs)
- `--n-reps`: Simulation replicates per trial (default: 20)
- `--n-cpus-per-trial`: CPUs for MultiSim within trial (default: all available)
- `--worker-id`: Identifier for this worker (default: auto-generated)
- `--db-path`: Path to Optuna database (default: rota_hybrid_multisim.db)
