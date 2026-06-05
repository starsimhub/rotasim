"""
Exp 03 — model verification (see README.md). Bug vs. genuine mixing limitation.

Reuses calibrate_maled._run_one_replicate (no model rebuild):
  Prong 1 (denominator): run the reference fit (#19) under the UK and Bangladesh
    pyramids; recompute symptomatic IR-by-age with the current CROSS-SECTIONAL
    person-time vs a BIRTH-COHORT person-time (PT proportional to bin width).
    Does the shape move toward the data?  (cases & cross-sectional PT come
    straight from the worker's output.)
  Prong 2 (peak): all-infection incidence by age, obtained by running the
    infection-number symptom model with p_symp_1=p_symp_2=p_symp_3plus=1 (every
    infection counted), across a base_beta sweep. Surge-then-decline or monotone?
    A low <6m here also confirms maternal immunity is suppressing early infection.

Run on the VM:  python run.py
"""
import sys
import json
import pathlib
import argparse
from multiprocessing import get_context

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR))
from calibrate_maled import (   # noqa: E402
    _run_one_replicate, SITE_DEMOGRAPHICS,
    FIXED_REPORTING_RATE, FIXED_CONSTANT_SEVERITY,
    process_incidence_maled, thisdir,
)

BINS = process_incidence_maled.MALED_AGE_BINS
WIDTHS = np.array([6., 6., 12., 12.])         # MAL-ED bin widths (months)
WINDOW = (5.0, 10.0)
WINDOW_MO = (WINDOW[1] - WINDOW[0]) * 12.0


def base_config(site, n_agents, age_data_path, symptom_model, n_stages=1):
    demo = SITE_DEMOGRAPHICS[site]
    return dict(n_agents=n_agents, start='2003-01-01', stop='2013-01-01', n_contacts=7,
                birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
                constant_severity=FIXED_CONSTANT_SEVERITY, reporting_rate=FIXED_REPORTING_RATE,
                age_data_path=age_data_path, p_asymp_detect=0.4,
                symptom_model=symptom_model, maternal_n_stages=n_stages)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--out-root', default=str(HERE))
    ap.add_argument('--n-agents', type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.n_agents:
        cfg['n_agents'] = args.n_agents
    N = cfg['n_agents']; site = cfg['site']
    out_root = pathlib.Path(args.out_root)
    (out_root / 'outputs').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    uk = str(thisdir / 'uk_age_data.csv')
    bgd = str(thisdir / 'bangladesh_age_data.csv')
    ref = dict(cfg['ref_params'])

    # ---- assemble runs: (label, sim_config, sim_pars) ----
    runs = [
        ('symp_UK',  base_config(site, N, uk,  'age_only', 1), ref),
        ('symp_BGD', base_config(site, N, bgd, 'age_only', 1), ref),
    ]
    for b in cfg['base_beta_sweep']:
        pars = dict(base_beta=b,
                    sus_after_1=ref['sus_after_1'], sus_after_2=ref['sus_after_2'],
                    sus_after_3plus=ref['sus_after_3plus'],
                    maternal_immunity_efficacy=ref['maternal_immunity_efficacy'],
                    maternal_immunity_half_life_days=ref['maternal_immunity_half_life_days'],
                    p_symp_1=1.0, p_symp_2=1.0, p_symp_3plus=1.0)   # every infection counted
        runs.append((f'allinf_beta{b}', base_config(site, N, uk, 'infection_number', 1), pars))

    rng = np.random.default_rng(cfg['seed'])
    seeds = rng.integers(0, 1_000_000, len(runs)).tolist()
    tasks = [(rc, rp, int(s), WINDOW) for (_, rc, rp), s in zip(runs, seeds)]
    print(f"Running {len(runs)} diagnostic sims ({N} agents each): {[r[0] for r in runs]}")
    with get_context('spawn').Pool(processes=len(runs)) as pool:
        outs = pool.map(_run_one_replicate, tasks)
    res = {lbl: mo for (lbl, _, _), mo in zip(runs, outs)}

    # persist the reduced outputs
    with (out_root / 'outputs' / 'results.jsonl').open('w') as f:
        for lbl, mo in res.items():
            f.write(json.dumps({'label': lbl,
                                'cases': mo['ir_by_age']['cases'].tolist(),
                                'PT': mo['ir_by_age']['PT'].tolist(),
                                'IR': mo['ir_by_age']['IR'].tolist(),
                                'first_inf': mo['first_infection']}) + '\n')

    targets = process_incidence_maled.load_targets(site)
    tgt_ir = np.array([targets['ir_by_age'].loc[b, 'IR'] for b in BINS])

    # ---- Prong 1: cross-sectional vs cohort denominator ----
    def cohort_and_xsec(mo):
        cases = mo['ir_by_age']['cases'].values.astype(float)
        xsec_pt = mo['ir_by_age']['PT'].values.astype(float)
        cohort_pt = xsec_pt.sum() * WIDTHS / WIDTHS.sum()   # redistribute total PT by bin width
        ir_xsec = np.where(xsec_pt > 0, cases / xsec_pt * 100, 0.0)
        ir_cohort = np.where(cohort_pt > 0, cases / cohort_pt * 100, 0.0)
        return ir_xsec, ir_cohort, xsec_pt / WINDOW_MO   # last = headcount per bin

    print("\n=== Prong 1: person-time denominator (does cohort PT reshape toward data?) ===")
    print(f"  {'DATA':<10} IR={[round(v,2) for v in tgt_ir]}")
    p1 = {}
    for lbl in ('symp_UK', 'symp_BGD'):
        ir_x, ir_c, hc = cohort_and_xsec(res[lbl])
        p1[lbl] = (ir_x, ir_c)
        print(f"  {lbl}:")
        print(f"    headcount/bin = {[int(v) for v in hc]}  | headcount:width ratio = "
              f"{[round(h/w, 1) for h, w in zip(hc, WIDTHS)]}  (flat ratio => xsec==cohort shape)")
        print(f"    IR cross-sectional (current) = {[round(v,2) for v in ir_x]}")
        print(f"    IR birth-cohort (PT∝width)   = {[round(v,2) for v in ir_c]}")

    # ---- Prong 2: all-infection incidence by age (peak vs monotone) ----
    print("\n=== Prong 2: all-infection incidence by age (peak? maternal <6m suppression?) ===")
    allinf = {}
    for b in cfg['base_beta_sweep']:
        ir = res[f'allinf_beta{b}']['ir_by_age']['IR'].values
        allinf[b] = ir
        shape = "PEAK@6-11" if (ir[1] > ir[0] and ir[1] >= ir[2]) else ("monotone-decr" if ir[0] >= ir[1] >= ir[2] else "other")
        print(f"  beta={b}: all-inf IR by age = {[round(v,2) for v in ir]}  -> {shape}")

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))
    x = np.arange(len(BINS)); w = 0.15
    axA.bar(x - 2 * w, tgt_ir, w, color='red', label='MAL-ED data')
    styles = [('symp_UK', 'cross', '#1f77b4'), ('symp_UK', 'cohort', '#7fb1d6'),
              ('symp_BGD', 'cross', '#2ca02c'), ('symp_BGD', 'cohort', '#9bd49b')]
    for i, (lbl, kind, c) in enumerate(styles):
        ir = p1[lbl][0 if kind == 'cross' else 1]
        axA.bar(x + (i - 1) * w, ir, w, color=c, label=f'{lbl} {kind}')
    axA.set_xticks(x); axA.set_xticklabels(BINS); axA.set_ylabel('Symptomatic IR /100 PM')
    axA.set_title('(A) Prong 1: cross-sectional vs birth-cohort denominator\n(reference fit #19, UK vs BGD pyramid)')
    axA.legend(frameon=False, fontsize=7.5); axA.spines[['top', 'right']].set_visible(False)

    axB.plot(x, tgt_ir, 'r*-', ms=14, lw=2, label='MAL-ED data (symptomatic)')
    for b in cfg['base_beta_sweep']:
        axB.plot(x, allinf[b], 'o-', label=f'all-infection, beta={b}')
    axB.set_xticks(x); axB.set_xticklabels(BINS); axB.set_ylabel('Incidence /100 PM')
    axB.set_title('(B) Prong 2: all-infection incidence by age\n(can homogeneous mixing peak at 6-11mo?)')
    axB.legend(frameon=False, fontsize=8); axB.spines[['top', 'right']].set_visible(False)

    fig.suptitle('Exp 03 — model verification (MAL-ED Bangladesh)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_root / 'figures' / 'model_verification.png', dpi=150)
    print(f"\nwrote {out_root / 'figures' / 'model_verification.png'}")


if __name__ == '__main__':
    main()
