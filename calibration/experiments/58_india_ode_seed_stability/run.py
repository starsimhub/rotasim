"""Exp 58 part 1 orchestrator -- runs `run_one_seed.py <seed>` as a fresh OS
process per seed (sequentially; each already saturates all cores via
workers=-1). See run_one_seed.py's docstring for why this is a separate
process per seed rather than one loop: repeated in-process Pool creation
exhausted zebra's open-file-descriptor limit by the 4th seed, and a crash in
one seed shouldn't take down the others.
"""
import pathlib, subprocess, sys, time, json

HERE = pathlib.Path(__file__).resolve().parent
WORKER = HERE / 'run_one_seed.py'
RUNS_FILE = HERE / 'outputs' / 'seed_runs.jsonl'

SEEDS = [20260817, 1, 2, 3, 4, 5]  # first = exp57's original seed, repeated for consistent diagnostics

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
