"""
Re-run a specific Optuna trial's parameters and print a per-bin fit breakdown.

Diagnostic tool — does NOT modify the study database. Useful when the original
calibration log didn't dump per-age-bin model output, or when you want to look
at a specific past trial in more detail, possibly under a different fit-target
interpretation than the one it was originally calibrated against.

Reuses the spawn-pool worker function from calibrate_maled.py.

Usage:
  # Look at trial 3 from the default 'joint' DB
  python evaluate_trial.py --site bangladesh --trial-number 3 --n-reps 5

  # Pull trial 7 from the symptomatic-IR-only study
  python evaluate_trial.py --site bangladesh --trial-number 7 --fit-target symptomatic_ir

  # Re-evaluate trial 3 with no detection filter (matches old code semantics)
  python evaluate_trial.py --site bangladesh --trial-number 3 --p-asymp-detect 1.0

  # Explicit DB path overrides everything
  python evaluate_trial.py --site bangladesh --trial-number 3 \\
    --db-path rota_maled_bangladesh_v9.db
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


def _default_db_and_study(site, fit_target):
    """Match the convention in calibrate_maled.main()."""
    suffix = '' if fit_target == 'joint' else f'_{fit_target}'
    return f'rota_maled_{site}{suffix}.db', f'rota_maled_{site}{suffix}'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--site', required=True, choices=list(SITE_DEMOGRAPHICS.keys()))
    p.add_argument('--trial-number', type=int, required=True)
    p.add_argument('--n-reps', type=int, default=5)
    p.add_argument('--db-path', type=str, default=None,
                   help='Override the default DB path. Default follows the same naming '
                        'convention as calibrate_maled.py (suffix by fit-target).')
    p.add_argument('--study-name', type=str, default=None,
                   help='Override the default study name. Default follows the same '
                        'convention as calibrate_maled.py.')
    p.add_argument('--fit-target', default='joint',
                   choices=['joint', 'symptomatic_ir', 'first_infection'],
                   help='Which study to load from. Also picks the headline GOF in the '
                        'output, but the per-component breakdown shows all three views.')
    p.add_argument('--p-asymp-detect', type=float, default=0.4,
                   help='P(asymp infection detected by MAL-ED). Set to 1.0 to disable '
                        'the detection filter (matches pre-40b9743 semantics).')
    args = p.parse_args()

    db_default, study_default = _default_db_and_study(args.site, args.fit_target)
    db = args.db_path or db_default
    study_name = args.study_name or study_default

    print(f"Loading trial #{args.trial_number} from study '{study_name}' in {db}")
    study = optuna.load_study(study_name=study_name, storage=f'sqlite:///{db}')
    trial = next((t for t in study.trials if t.number == args.trial_number), None)
    if trial is None:
        raise SystemExit(f"Trial #{args.trial_number} not found in {db}")
    sim_pars = dict(trial.params)

    print("=" * 70)
    print(f"Re-running Trial #{args.trial_number} for {args.site} "
          f"(n_reps={args.n_reps}, fit_target={args.fit_target}, "
          f"p_asymp_detect={args.p_asymp_detect})")
    print("=" * 70)
    print(f"Stored GOF: {trial.value:.4f}")
    print("Parameters:")
    for k, v in sim_pars.items():
        print(f"  {k:<35s} = {v:.6f}")
    # Flag any params the worker expects but the stored trial lacks (pre-feature trials).
    expected = {'beta0', 'beta1', 'beta2',
                'sus_after_1', 'sus_after_2', 'sus_after_3plus',
                'maternal_immunity_efficacy', 'maternal_immunity_half_life_days',
                'base_beta'}
    missing = sorted(expected - set(sim_pars))
    if missing:
        print(f"\nNote: trial predates these parameters: {missing}")
        print(f"      Worker will use safe defaults (maternal=0.0 = OFF).")

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
        p_asymp_detect=args.p_asymp_detect,
    )

    seeds = np.random.randint(0, int(1e6), args.n_reps).tolist()
    args_list = [(sim_config, sim_pars, int(s), cal_window) for s in seeds]

    print(f"\nSpawning {args.n_reps} worker(s)...")
    ctx = get_context('spawn')
    with ctx.Pool(processes=args.n_reps) as pool:
        model_outs = pool.map(_run_one_replicate, args_list)
    print(f"Done. Aggregating results across {len(model_outs)} replicates.\n")

    # ---- Per-bin IR ----
    bins = process_incidence_maled.MALED_AGE_BINS
    print("=== Incidence by age (per 100 person-months) ===")
    header = f"{'bin':<8} {'target':>8}  {'model_med':>10} {'model_min':>10} {'model_max':>10}  {'log_diff':>10}"
    print(header)
    print('-' * len(header))
    for b in bins:
        t_ir  = targets['ir_by_age'].loc[b, 'IR']
        m_irs = np.array([mo['ir_by_age'].loc[b, 'IR'] for mo in model_outs])
        m_med = float(np.median(m_irs))
        eps = process_incidence_maled.LOG_EPS
        log_diff = float(np.log(t_ir + eps) - np.log(m_med + eps))
        print(f"{b:<8} {t_ir:>8.3f}  {m_med:>10.3f} {m_irs.min():>10.3f} {m_irs.max():>10.3f}  {log_diff:>+10.3f}")

    # ---- Per-quartile first-detected infection age ----
    print("\n=== Age at first DETECTED infection (months) ===")
    fi_t = targets['first_infection']
    for key in ('q25', 'median', 'q75'):
        m_vals = np.array([mo['first_infection'][key] for mo in model_outs])
        diff = float(np.median(m_vals)) - fi_t[key]
        print(f"  {key:<7} target={fi_t[key]:>6.2f}  "
              f"model median={float(np.median(m_vals)):>6.2f}  "
              f"(range [{m_vals.min():.2f}, {m_vals.max():.2f}], diff={diff:+.2f})")

    # ---- GOF under all three interpretations ----
    print("\n=== GOF breakdown (per-replicate medians) ===")
    print(f"{'view':<18} {'GOF':>10}  {'GOF_inc':>10}  {'GOF_first':>10}")
    for view in ('joint', 'symptomatic_ir', 'first_infection'):
        gofs = [process_incidence_maled.gof(mo, targets, fit_target=view) for mo in model_outs]
        m_total = float(np.median([g['gof']                  for g in gofs]))
        m_inc   = float(np.median([g['gof_incidence']        for g in gofs]))
        m_first = float(np.median([g['gof_first_infection'] for g in gofs]))
        marker = '  *' if view == args.fit_target else ''
        print(f"{view:<18} {m_total:>10.4f}  {m_inc:>10.4f}  {m_first:>10.4f}{marker}")
    print("  (* = the view that matches --fit-target; "
          "the GOF_inc and GOF_first columns are unchanged across views, "
          "just the headline 'GOF' weights them differently)")


if __name__ == '__main__':
    main()
