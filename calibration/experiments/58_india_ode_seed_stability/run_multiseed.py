"""Exp 58 part 1 -- repeat exp57's ODE direct-fit (differential_evolution, 12
params, BDF solver) across multiple seeds to check MLE stability. Re-runs the
original seed (20260817) too so every run captures the same extra diagnostic:
not just the single best point, but the top-K members of the final population,
for pooling into exp58 part 2's equilibrium re-analysis.

Imports exp57's composite_logL/BOUNDS by explicit file path, not by module
name -- ode_model_age.py's own import inserts 54_india_ode_approximation/ at
sys.path[0] as a side effect, which would otherwise shadow exp57's run.py
(both dirs happen to contain a file named run.py).
"""
import sys, pathlib, time, json, importlib.util
import numpy as np
from scipy.optimize import differential_evolution

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
EXP57 = HERE.parents[0] / '57_india_ode_direct_fit'
sys.path.insert(0, str(CALIB))
sys.path.insert(0, str(HERE.parents[0] / '54_india_ode_approximation'))
sys.path.insert(0, str(HERE.parents[0] / '55_india_age_structured_ode'))
sys.path.insert(0, str(EXP57))

_spec = importlib.util.spec_from_file_location("exp57_run", str(EXP57 / "run.py"))
exp57 = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(exp57)

OUT_DIR = HERE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
RUNS_FILE = OUT_DIR / 'seed_runs.jsonl'
POOL_FILE = OUT_DIR / 'pooled_candidates.jsonl'
TOP_K_PER_RUN = 8  # keep more than needed so part-2's pooled top-10 has headroom

SEEDS = [20260817, 1, 2, 3, 4, 5]  # first = exp57's original seed, repeated for consistent diagnostics

if __name__ == '__main__':
    for seed in SEEDS:
        print(f"\n=== seed {seed} ===")
        t0 = time.time()
        result = differential_evolution(
            exp57.neg_logL, exp57.BOUNDS, maxiter=60, popsize=15, tol=1e-6,
            seed=seed, workers=-1, updating='deferred', polish=True,
        )
        dt = time.time() - t0
        best_logL = -result.fun
        best_params = dict(zip(exp57.PARAM_NAMES, result.x))
        print(f"seed {seed}: best logL={best_logL:.2f}  nit={result.nit}  nfev={result.nfev}  [{dt:.0f}s]")

        with RUNS_FILE.open('a') as f:
            f.write(json.dumps(dict(seed=seed, best_logL=best_logL, best_params=best_params,
                                     nit=int(result.nit), nfev=int(result.nfev), wall_s=dt)) + '\n')

        # pool the top-K members of this run's FINAL population, not just the single best
        energies = np.asarray(result.population_energies)
        order = np.argsort(energies)[:TOP_K_PER_RUN]
        with POOL_FILE.open('a') as f:
            for rank, i in enumerate(order):
                params = dict(zip(exp57.PARAM_NAMES, result.population[i]))
                f.write(json.dumps(dict(seed=seed, rank=rank, logL=float(-energies[i]),
                                         params=params)) + '\n')
        print(f"  appended to {RUNS_FILE.name} and {POOL_FILE.name}")

    print("\nAll seeds done.")
