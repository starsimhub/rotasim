"""Exp 59 orchestrator -- runs `run_one_seed.py <seed>` as a fresh OS process
per seed (sequentially; each already saturates all cores via workers=-1),
same pattern as exp58 (avoids the open-file-descriptor exhaustion looping
differential_evolution in one long-lived process caused there, and isolates
a crash in one seed from the others). Same 6 seeds as exp58 for a
directly-comparable stability check between age_binned and infnum.
"""
import pathlib, subprocess, sys, time, json

HERE = pathlib.Path(__file__).resolve().parent
WORKER = HERE / 'run_one_seed.py'
RUNS_FILE = HERE / 'outputs' / 'seed_runs.jsonl'

SEEDS = [20260817, 1, 2, 3, 4, 5]  # same seeds as exp58, for a like-for-like comparison

if __name__ == '__main__':
    done = set()
    if RUNS_FILE.exists():
        done = {json.loads(l)['seed'] for l in RUNS_FILE.open()}
        print(f"Resuming: seeds already completed = {sorted(done)}")

    for seed in SEEDS:
        if seed in done:
            print(f"seed {seed}: already in {RUNS_FILE.name}, skipping")
            continue
        t0 = time.time()
        r = subprocess.run([sys.executable, str(WORKER), str(seed)])
        dt = time.time() - t0
        if r.returncode != 0:
            print(f"\n!!! seed {seed} FAILED (rc={r.returncode}) after {dt:.0f}s -- "
                  f"continuing to next seed, this one is just missing from the results.")
        else:
            print(f"\nseed {seed} subprocess finished cleanly [{dt:.0f}s]")

    print("\nAll seeds attempted.")
