"""Exp 41 — India VE validation from exp39 (age_binned fixed-psymp) posterior.

Forward-prediction validation: take top-K best-fitting India draws from exp39
(age_binned + biweekly-fixed p_symp), add Rotavac (3 doses at 6/10/14 weeks),
compute test-negative-analog direct VE (1 - IRR_vax/unvax) by age group.

Targets from Nair et al. Nature Medicine 2025 (31 hospitals, 9 states, 2016-2020,
test-negative case-control, severe rotavirus gastroenteritis):
  - 6-11 months: 59% (47-68%)
  - 12-23 months: ~51-54%
  - 6-59 months overall: 54% (45-62%)

User ecologic estimate (all symptomatic, includes mild cases):
  - 6-11 months: 52.4%
  - 6-59 months: ~35-40% (by eye; model expected to overshoot without waning)

Take rates from Patel et al. JID 2009:
  - 0.63 (conservative)
  - 0.74 (middle-income country seroconversion rate)

Coverage: 0.90 (Nair final programmatic coverage)
MoA: infection_blocking (advances infection-equivalent counter; feeds susceptibility)
Dosing: Rotavac at 6, 10, 14 weeks of age (India Universal Immunization Program)

Model note: no waning modelled → expect VE to be accurate at 6-11m but
overshoot at older ages (12-35m). This is informative — overshoot = waning signal.

Run on zebra:
  tmux new-session -s india41 \\
    "cd ~/rotasim/rotasim/calibration && \\
     MALED_SITE=india NEO_PRIME=1 \\
     ~/ukvenv/bin/python experiments/41_india_ve_validation/run.py \\
     2>&1 | tee experiments/41_india_ve_validation/india41.log"
"""
import os, sys, json, pathlib
os.environ.setdefault('MALED_SITE', 'india')
os.environ.setdefault('NEO_PRIME', '1')
from multiprocessing import get_context
import numpy as np, pandas as pd

HERE  = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))

import calibrate_maled as cm
from hm_calibrate import build_sim_config, untransform

N_WORKERS      = int(os.environ.get('HM_WORKERS', str(os.cpu_count() or 40)))
N_AGENTS       = 80_000
N_TOPK         = 40
MIN_NOVAX_CASES = 100   # drop near-extinct draws
# 0.05 coverage: minimal herd immunity so within-sim vaccinated vs unvaccinated
# comparison approximates direct (individual-level) VE. At 50% coverage in a
# well-mixed ABM, herd immunity reduces unvaccinated IR by ~25%, collapsing the
# test-negative VE to near-zero even though the underlying direct VE is ~57%.
# Low coverage eliminates this artefact. Parallel (vax sim vs novax sim) impact
# reported separately for the Nair et al. positivity-drop comparison.
COVERAGE        = 0.05
COVERAGE_IMPACT = 0.90   # high coverage for parallel population-impact comparison
TAKE_RATES      = [0.63, 0.74]   # Patel JID 2009: conservative vs middle-income
# Rotavac: 3 doses at 6, 10, 14 weeks of age (Universal Immunization Program India)
DOSE_AGES_Y    = [6/52, 10/52, 14/52]

# Exp39 outputs
EXP39_TS   = CALIB / 'experiments' / '39_india_age_binned_fixed' / 'outputs' / 'ts'
EXP39_HM   = (CALIB / 'experiments' / '39_india_age_binned_fixed' / 'outputs' / 'hm'
               / 'maled_age_binned_titer_fixedagepsymp')

# Surveillance bins: 0-6m, 6-12m, 12-24m, 24-36m, 36-48m, 48-60m
BIN_EDGES_M = (0.0, 6.0, 12.0, 24.0, 36.0, 48.0, 60.0)
BIN_LABELS  = ['<6 m', '6-11 m', '12-23 m', '24-35 m', '36-47 m', '48-59 m']
CAP_AGE_M   = 60.0

# Calibration window: years 5-10 of a 10-year sim (equilibrium portion, post-burn-in)
CAL_WINDOW  = (5.0, 10.0)


def _read_jsonl(path):
    rows = []
    for line in open(path, errors='ignore'):
        line = line.replace('\x00', '').strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def draw_topk(k):
    """Top-k India draws by TS logL score from exp39."""
    recs  = _read_jsonl(EXP39_TS / 'sir_results.jsonl')
    nroy  = pd.read_csv(EXP39_TS / 'nroy_draw.csv').reset_index(drop=True)
    logls = {r['idx']: r['logL'] for r in recs if np.isfinite(r.get('logL', np.nan))}
    scored = sorted(logls.items(), key=lambda x: -x[1])[:k]
    idxs  = [i for i, _ in scored]
    return nroy.iloc[idxs].reset_index(drop=True)


def _run(args):
    sp, vax_cfg, seed = args
    sc = build_sim_config('age_binned', N_AGENTS, 'titer')
    # Switch to surveillance mode for direct VE split (cases_vax/unvax by age bin)
    sc['observation']  = 'surveillance'
    sc['bin_edges_m']  = BIN_EDGES_M
    sc['cap_age_m']    = CAP_AGE_M
    if vax_cfg is not None:
        sc['vaccine'] = vax_cfg
    mo = cm._run_one_replicate((sc, sp, int(seed), CAL_WINDOW))
    return dict(
        total_cases = mo.get('total_cases', 0),
        ir          = mo.get('ir_per_100cy', [0.0] * len(BIN_LABELS)),
        cases_vax   = mo.get('cases_vax',   [0] * len(BIN_LABELS)),
        cases_unvax = mo.get('cases_unvax', [0] * len(BIN_LABELS)),
        py_vax      = mo.get('py_vax',      [0.0] * len(BIN_LABELS)),
        py_unvax    = mo.get('py_unvax',    [0.0] * len(BIN_LABELS)),
        ve_direct   = mo.get('ve_direct',   [float('nan')] * len(BIN_LABELS)),
    )


def direct_ve_pooled(draws_results, lo_bin, hi_bin):
    """Pool cases and person-time across draws, compute VE = 1 - IRR."""
    cv = cu = pv = pu = 0.0
    for r in draws_results:
        cv += sum(r['cases_vax'][lo_bin:hi_bin])
        cu += sum(r['cases_unvax'][lo_bin:hi_bin])
        pv += sum(r['py_vax'][lo_bin:hi_bin])
        pu += sum(r['py_unvax'][lo_bin:hi_bin])
    ve = (1.0 - (cv / pv) / (cu / pu)) if (pv > 0 and pu > 0 and cu > 0) else float('nan')
    return dict(ve=ve, cases_vax=cv, cases_unvax=cu, py_vax=pv, py_unvax=pu)


def main():
    print(f"India VE validation | exp39 top-{N_TOPK} draws | coverage={COVERAGE} | take={TAKE_RATES}")
    print(f"Rotavac 3-dose schedule: {[round(d*52, 1) for d in DOSE_AGES_Y]} weeks | {N_AGENTS} agents | {N_WORKERS} workers")

    draws = draw_topk(N_TOPK)
    print(f"Loaded {len(draws)} top draws from exp39 TS")

    tasks, meta = [], []
    for i, (_, row) in enumerate(draws.iterrows()):
        sp = untransform(row, 'age_binned', 'titer', fix_age_psymp=True)
        seed = 70000 + i
        tasks.append((sp, None, seed));  meta.append((i, 'novax', None))
        for take in TAKE_RATES:
            # Low-coverage (5%): direct VE, minimal herd immunity
            vax_low = dict(response_prob=take, coverage=COVERAGE,
                           dose_ages_y=DOSE_AGES_Y, moa='infection_blocking')
            tasks.append((sp, vax_low, seed)); meta.append((i, 'vax_low', take))
            # High-coverage (90%): population impact for positivity-drop comparison
            vax_hi = dict(response_prob=take, coverage=COVERAGE_IMPACT,
                          dose_ages_y=DOSE_AGES_Y, moa='infection_blocking')
            tasks.append((sp, vax_hi, seed)); meta.append((i, 'vax_hi', take))

    with get_context('spawn').Pool(processes=min(N_WORKERS, len(tasks)), maxtasksperchild=4) as pool:
        results = pool.map(_run, tasks)

    by = {(i, k, t): r for r, (i, k, t) in zip(results, meta)}

    healthy = [i for i in range(len(draws))
               if by.get((i, 'novax', None), {}).get('total_cases', 0) >= MIN_NOVAX_CASES]
    print(f"\n{len(healthy)}/{len(draws)} draws healthy (>={MIN_NOVAX_CASES} novax cases)")
    VAX_KEY = 'vax_low'

    # Age groups for output (bin index ranges into BIN_LABELS)
    AGE_GROUPS = {
        '<6 m':     (0, 1),
        '6-11 m':   (1, 2),
        '12-23 m':  (2, 3),
        '24-35 m':  (3, 4),
        '6-35 m':   (1, 4),   # comparable to model's calibration range
        '6-59 m':   (1, 6),   # Nair paper range
    }

    out = {'n_topk': len(draws), 'n_healthy': len(healthy),
           'take_rates': TAKE_RATES, 'coverage': COVERAGE,
           'dose_ages_weeks': [round(d * 52, 1) for d in DOSE_AGES_Y],
           'bin_labels': BIN_LABELS}

    # No-vax IR by age (check calibration)
    novax_ir = [[by[(i, 'novax', None)]['ir'][b] for b in range(len(BIN_LABELS))]
                for i in healthy]
    out['novax_ir_med'] = np.median(novax_ir, 0).tolist()

    print(f"\nNo-vax symptomatic IR (per 100 child-yr) by age bin:")
    for b, label in enumerate(BIN_LABELS):
        print(f"  {label}: {out['novax_ir_med'][b]:.2f}")

    print("\nDirect VE (1 - IRR, pooled across draws):")
    print(f"{'Age group':<12}  {'Target (ecol/Nair)':<22}", end='')
    for take in TAKE_RATES:
        print(f"  take={take}", end='')
    print()

    targets = {
        '6-11 m':  '52% ecol / 59% Nair',
        '12-23 m': '~45-54% Nair',
        '6-35 m':  '35-40% ecol (est)',
        '6-59 m':  '54% Nair (severe)',
    }
    out['ve'] = {}
    for label, (lo, hi) in AGE_GROUPS.items():
        row_str = f"{label:<12}  {targets.get(label, ''):<22}"
        out['ve'][label] = {}
        for take in TAKE_RATES:
            vax_res = [by[(i, VAX_KEY, take)] for i in healthy]
            dv = direct_ve_pooled(vax_res, lo, hi)
            out['ve'][label][str(take)] = dv
            row_str += f"  {dv['ve']:.3f}"
        print(row_str)

    # Per-draw VE for uncertainty bands
    out['ve_perdraw'] = {}
    for label, (lo, hi) in AGE_GROUPS.items():
        out['ve_perdraw'][label] = {}
        for take in TAKE_RATES:
            ve_list = []
            for i in healthy:
                r = by[(i, VAX_KEY, take)]
                cv = sum(r['cases_vax'][lo:hi]); cu = sum(r['cases_unvax'][lo:hi])
                pv = sum(r['py_vax'][lo:hi]);   pu = sum(r['py_unvax'][lo:hi])
                if pv > 0 and pu > 0 and cu > 0:
                    ve_list.append(1.0 - (cv / pv) / (cu / pu))
            out['ve_perdraw'][label][str(take)] = ve_list

    # Population impact (high coverage): parallel novax vs vax-hi
    print(f"\nPopulation impact (coverage={COVERAGE_IMPACT}, parallel novax vs vax-hi):")
    print(f"  (Compare to Nair: rotavirus positivity 40%→20%, ~50% reduction)")
    out['impact'] = {}
    for label, (lo, hi) in AGE_GROUPS.items():
        out['impact'][label] = {}
        for take in TAKE_RATES:
            novax_ir_list = [sum(by[(i,'novax',None)]['ir'][lo:hi]) for i in healthy]
            hi_ir_list    = [sum(by[(i,'vax_hi',take)]['ir'][lo:hi]) for i in healthy]
            ratios = [1 - h/n if n > 0 else float('nan') for h,n in zip(hi_ir_list, novax_ir_list)]
            finite = [r for r in ratios if not (r != r)]
            med = float(np.median(finite)) if finite else float('nan')
            out['impact'][label][str(take)] = dict(median=med, n=len(finite))
        if label in ('6-11 m', '6-35 m', '6-59 m'):
            print(f"  {label}: take=0.63: {out['impact'][label]['0.63']['median']:.3f}  "
                  f"take=0.74: {out['impact'][label]['0.74']['median']:.3f}")

    (HERE / 'outputs').mkdir(exist_ok=True)
    json.dump(out, open(HERE / 'outputs' / 'india_ve.json', 'w'), indent=2)
    print(f"\nwrote outputs/india_ve.json")


if __name__ == '__main__':
    main()
