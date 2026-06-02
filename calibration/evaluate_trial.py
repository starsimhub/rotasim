"""
Re-run a specific Optuna trial's parameters and print a per-bin fit breakdown.

Diagnostic tool — does NOT modify the study database. Useful when the original
calibration log didn't dump per-age-bin model output, or when you want to look
at a specific past trial in more detail.

Reuses the spawn-pool worker function from calibrate_maled.py.

Usage:
  python evaluate_trial.py --site bangladesh --trial-number 3 --n-reps 5
"""
import argparse
from multiprocessing import get_context

import numpy as np
import optuna

# Pull worker + config from the main calibration module. Importing it triggers
# only its module-level definitions (the main() body is __main__-guarded).
from calibrate_maled import (
    _run_one_replicate,
    SITE_DEMOGRAPHICS,
    FIXED_REPORTING_RATE,
    FIXED_CONSTANT_SEVERITY,
    process_incidence_maled,
    thisdir,
)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--site', required=True, choices=list(SITE_DEMOGRAPHICS.keys()))
    p.add_argument('--trial-number', type=int, required=True)
    p.add_argument('--n-reps', type=int, default=5)
    p.add_argument('--db-path', type=str, default=None)
    args = p.parse_args()

    db = args.db_path or f'rota_maled_{args.site}.db'
    study = optuna.load_study(study_name=f'rota_maled_{args.site}',
                               storage=f'sqlite:///{db}')
    trial = next((t for t in study.trials if t.number == args.trial_number), None)
    if trial is None:
        raise SystemExit(f"Trial #{args.trial_number} not found in {db}")
    sim_pars = dict(trial.params)

    print("=" * 70)
    print(f"Re-running Trial #{args.trial_number} for {args.site} "
          f"(n_reps={args.n_reps})")
    print("=" * 70)
    print(f"Stored GOF: {trial.value:.4f}")
    print("Parameters:")
    for k, v in sim_pars.items():
        print(f"  {k:<20s} = {v:.6f}")

    targets = process_incidence_maled.load_targets(args.site)
    demo = SITE_DEMOGRAPHICS[args.site]
    cal_window = (5.0, 10.0)
    sim_config = dict(
        n_agents=100_000,
        start='2003-01-01',
        stop='2013-01-01',
        n_contacts=7,
        birth_rate=demo['birth_rate'],
        death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY,
        reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'),
    )

    seeds = np.random.randint(0, int(1e6), args.n_reps).tolist()
    args_list = [(sim_config, sim_pars, int(s), cal_window) for s in seeds]

    print(f"\nSpawning {args.n_reps} worker(s)...")
    ctx = get_context('spawn')
    with ctx.Pool(processes=args.n_reps) as pool:
        model_outs = pool.map(_run_one_replicate, args_list)
    print(f"Done. Aggregating results across {len(model_outs)} replicates.\n")

    bins = process_incidence_maled.MALED_AGE_BINS
    print("=== Incidence by age (per 100 person-months) ===")
    header = f"{'bin':<8} {'target':>8}  {'model_med':>10} {'model_min':>10} {'model_max':>10}  {'log_diff':>10}"
    print(header)
    print('-' * len(header))
    for b in bins:
        t_ir  = targets['ir_by_age'].loc[b, 'IR']
        m_irs = np.array([mo['ir_by_age'].loc[b, 'IR'] for mo in model_outs])
        m_med = float(np.median(m_irs))
        # log_diff = log(target+eps) - log(model+eps), same eps as GOF
        eps = process_incidence_maled.LOG_EPS
        log_diff = float(np.log(t_ir + eps) - np.log(m_med + eps))
        print(f"{b:<8} {t_ir:>8.3f}  {m_med:>10.3f} {m_irs.min():>10.3f} {m_irs.max():>10.3f}  {log_diff:>+10.3f}")

    print("\n=== Age at first infection (months) ===")
    fi_t = targets['first_infection']
    for key in ('q25', 'median', 'q75'):
        m_vals = np.array([mo['first_infection'][key] for mo in model_outs])
        diff = float(np.median(m_vals)) - fi_t[key]
        print(f"  {key:<7} target={fi_t[key]:>6.2f}  "
              f"model median={float(np.median(m_vals)):>6.2f}  "
              f"(range [{m_vals.min():.2f}, {m_vals.max():.2f}], diff={diff:+.2f})")

    print("\n=== GOF breakdown ===")
    gofs = [process_incidence_maled.gof(mo, targets) for mo in model_outs]
    print(f"  median total           = {np.median([g['gof'] for g in gofs]):.4f}")
    print(f"  median GOF_incidence   = {np.median([g['gof_incidence'] for g in gofs]):.4f}")
    print(f"  median GOF_first_inf   = {np.median([g['gof_first_infection'] for g in gofs]):.4f}")
    print(f"  range of GOF totals    = [{min(g['gof'] for g in gofs):.4f}, "
          f"{max(g['gof'] for g in gofs):.4f}]")


if __name__ == '__main__':
    main()
