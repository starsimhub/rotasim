"""Exp 58 part 1, single-seed worker. Runs exp57's ODE direct-fit
(differential_evolution, 12 params, BDF solver) for ONE seed and appends its
result to the shared output files. Invoked as its own OS process (from
run_multiseed.py, one `python3 run_one_seed.py <seed>` per seed) rather than
looped in-process: repeatedly creating a ~160-worker multiprocessing.Pool
inside one long-lived process exhausted zebra's 1024 open-file-descriptor
limit by the 4th seed (`OSError: [Errno 24] Too many open files`) -- a fresh
process per seed gets a fresh fd table and avoids the accumulation entirely,
and isolates a crash in one seed from the others.

Imports exp57's composite_logL/BOUNDS by explicit file path + sys.modules
registration (not by bare module name) -- see run_multiseed.py's docstring
for why (ode_model_age.py's own sys.path.insert side effect would otherwise
shadow exp57's run.py; multiprocessing's fork-based Pool needs the module
name to resolve via sys.modules for pickling neg_logL).
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
sys.modules["exp57_run"] = exp57
_spec.loader.exec_module(exp57)

OUT_DIR = HERE / 'outputs'
OUT_DIR.mkdir(exist_ok=True)
RUNS_FILE = OUT_DIR / 'seed_runs.jsonl'
POOL_FILE = OUT_DIR / 'pooled_candidates.jsonl'
TOP_K_PER_RUN = 8

if __name__ == '__main__':
    seed = int(sys.argv[1])
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

    energies = np.asarray(result.population_energies)
    order = np.argsort(energies)[:TOP_K_PER_RUN]
    with POOL_FILE.open('a') as f:
        for rank, i in enumerate(order):
            params = dict(zip(exp57.PARAM_NAMES, result.population[i]))
            f.write(json.dumps(dict(seed=seed, rank=rank, logL=float(-energies[i]),
                                     params=params)) + '\n')
    print(f"  appended to {RUNS_FILE.name} and {POOL_FILE.name}")
