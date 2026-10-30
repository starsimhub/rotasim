"""Exp 67 orchestrator -- runs run_age_binned.py and run_infnum.py as fresh OS
processes per seed (each already saturates all cores via workers=-1), same
subprocess-per-seed pattern as exp58/59/60/61/64/65. Resumable: skips seeds
already present in the corresponding outputs/seed_runs_*.jsonl.
"""
import pathlib, subprocess, sys, time, json

HERE = pathlib.Path(__file__).resolve().parent
SEEDS = [0, 1, 2, 3, 4, 5]

WORKERS = {
    'age_binned': (HERE / 'run_age_binned.py', HERE / 'outputs' / 'seed_runs_age_binned.jsonl'),
    'infnum': (HERE / 'run_infnum.py', HERE / 'outputs' / 'seed_runs_infnum.jsonl'),
}

if __name__ == '__main__':
    for model, (worker, runs_file) in WORKERS.items():
        done = set()
        if runs_file.exists():
            done = {json.loads(l)['seed'] for l in runs_file.open()}
            print(f"[{model}] Resuming: seeds already completed = {sorted(done)}")

        for seed in SEEDS:
            if seed in done:
                print(f"[{model}] seed {seed}: already in {runs_file.name}, skipping")
                continue
            t0 = time.time()
            r = subprocess.run([sys.executable, str(worker), str(seed)])
            dt = time.time() - t0
            if r.returncode != 0:
                print(f"\n!!! [{model}] seed {seed} FAILED (rc={r.returncode}) after {dt:.0f}s -- "
                      f"continuing to next seed, this one is just missing from the results.")
            else:
                print(f"\n[{model}] seed {seed} subprocess finished cleanly [{dt:.0f}s]")

    print("\nAll seeds attempted for both models.")
