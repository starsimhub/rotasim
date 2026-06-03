"""
Side-by-side comparison of multiple Optuna trials against MAL-ED targets.

Runs each specified trial with fresh seeds and reports:
  - Symptomatic IR by age bin: target vs each model
  - Pooled symptomatic IR (0-36 months): target vs each model
  - Age at first DETECTED infection (Q25 / median / Q75): target vs each model

Each --trial argument is of the form "<fit_target>:<trial_number>", where
<fit_target> is one of joint / symptomatic_ir / first_infection. Repeat
--trial to compare multiple trials.

Usage:
  python compare_trials.py --site bangladesh \\
      --trial symptomatic_ir:19 --trial first_infection:17 --n-reps 10
"""
import argparse
from multiprocessing import get_context

import numpy as np
import optuna

from calibrate_maled import (
    _run_one_replicate,
    SITE_DEMOGRAPHICS,
    FIXED_REPORTING_RATE,
    FIXED_CONSTANT_SEVERITY,
    process_incidence_maled,
    thisdir,
)


def parse_trial_spec(spec):
    if ':' not in spec:
        raise argparse.ArgumentTypeError(
            f"--trial must be '<fit_target>:<number>', got {spec!r}")
    target, num = spec.split(':', 1)
    if target not in ('joint', 'symptomatic_ir', 'first_infection'):
        raise argparse.ArgumentTypeError(
            f"fit_target must be joint|symptomatic_ir|first_infection, got {target!r}")
    return target, int(num)


def load_trial(site, fit_target, trial_num):
    suffix = '' if fit_target == 'joint' else f'_{fit_target}'
    db = f'rota_maled_{site}{suffix}.db'
    name = f'rota_maled_{site}{suffix}'
    study = optuna.load_study(study_name=name, storage=f'sqlite:///{db}')
    trial = next((t for t in study.trials if t.number == trial_num), None)
    if trial is None:
        raise SystemExit(f"trial {trial_num} not found in {db}")
    return trial, db, name


def run_one_trial(sim_pars, sim_config, cal_window, n_reps):
    seeds = np.random.randint(0, int(1e6), n_reps).tolist()
    args_list = [(sim_config, sim_pars, int(s), cal_window) for s in seeds]
    ctx = get_context('spawn')
    with ctx.Pool(processes=n_reps) as pool:
        return pool.map(_run_one_replicate, args_list)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--site', required=True, choices=list(SITE_DEMOGRAPHICS.keys()))
    parser.add_argument('--trial', action='append', required=True,
                        type=parse_trial_spec, dest='trials',
                        help='Trial spec, e.g. "symptomatic_ir:19". Repeat to compare.')
    parser.add_argument('--n-reps', type=int, default=10)
    parser.add_argument('--p-asymp-detect', type=float, default=0.4)
    args = parser.parse_args()

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

    # Run each trial in turn (each uses n_reps cores at peak)
    results = []
    for fit_target, num in args.trials:
        trial, db, study_name = load_trial(args.site, fit_target, num)
        label = f"{fit_target}#{num}"
        sim_pars = dict(trial.params)
        print(f"\nRunning {label} (stored GOF={trial.value:.4f}, db={db})")
        print(f"  params: {sim_pars}")
        outs = run_one_trial(sim_pars, sim_config, cal_window, args.n_reps)
        results.append((label, trial, outs))

    bins = process_incidence_maled.MALED_AGE_BINS
    cols = ['target'] + [r[0] for r in results]

    # --- 1. Symptomatic IR by age ---
    print("\n" + "=" * 78)
    print("Symptomatic IR by age (per 100 person-months) — model median across reps")
    print("=" * 78)
    print(f"{'age bin':<10}" + ''.join(f"{c:>20}" for c in cols))
    for b in bins:
        t_ir = float(targets['ir_by_age'].loc[b, 'IR'])
        row = [f"{t_ir:>20.3f}"]
        for _, _, outs in results:
            irs = np.array([mo['ir_by_age'].loc[b, 'IR'] for mo in outs])
            row.append(f"{float(np.median(irs)):>20.3f}")
        print(f"{b:<10}" + ''.join(row))

    # --- 2. Pooled IR ---
    print("\n" + "=" * 78)
    print("Pooled symptomatic IR across 0-36 months (cases / PT * 100)")
    print("=" * 78)
    t_cases = int(targets['ir_by_age']['cases'].sum())
    t_pt    = int(targets['ir_by_age']['PT'].sum())
    t_pool  = t_cases / t_pt * 100
    pooled_row = [f"{t_pool:>20.3f}"]
    for _, _, outs in results:
        m = []
        for mo in outs:
            cases = float(mo['ir_by_age']['cases'].sum())
            pt    = float(mo['ir_by_age']['PT'].sum())
            if pt > 0:
                m.append(cases / pt * 100)
        pooled_row.append(f"{float(np.median(m)):>20.3f}" if m else f"{'nan':>20}")
    print(f"{'IR/100PM':<10}" + ''.join(f"{c:>20}" for c in cols))
    print(f"{'pooled':<10}" + ''.join(pooled_row))
    print(f"  (target: {t_cases} cases / {t_pt} PT)")

    # --- 3. Age at first DETECTED infection ---
    print("\n" + "=" * 78)
    print("Age at first DETECTED infection (months) — model median across reps")
    print(f"(p_asymp_detect={args.p_asymp_detect})")
    print("=" * 78)
    print(f"{'quartile':<10}" + ''.join(f"{c:>20}" for c in cols))
    for key in ('q25', 'median', 'q75'):
        t_val = float(targets['first_infection'][key])
        row = [f"{t_val:>20.2f}"]
        for _, _, outs in results:
            vals = np.array([mo['first_infection'][key] for mo in outs])
            row.append(f"{float(np.median(vals)):>20.2f}")
        print(f"{key:<10}" + ''.join(row))


if __name__ == '__main__':
    main()
