"""
Exp 05 — non-linear (peaked) age-symptom curve + strong maternal feasibility (see README.md).

Reuses calibrate_maled._run_one_replicate (RandomNet homogeneous mixing, age_only
quadratic symptom model, Erlang maternal). Forward-runs peaked age-symptom beta sets
x base_beta, plots symptomatic IR by age vs data, and checks for the low-<6m /
6-11mo-peak / declining-tail shape. No calibration.

  python run.py
"""
import sys
import json
import itertools
import pathlib
import argparse
from multiprocessing import get_context

import numpy as np
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
WINDOW = (5.0, 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--out-root', default=str(HERE))
    ap.add_argument('--n-agents', type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.n_agents:
        cfg['n_agents'] = args.n_agents
    out_root = pathlib.Path(args.out_root)
    (out_root / 'outputs').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    demo = SITE_DEMOGRAPHICS[cfg['site']]
    sim_config = dict(
        n_agents=cfg['n_agents'], start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY, reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'), p_asymp_detect=0.4,
        symptom_model='age_only', maternal_n_stages=cfg['maternal_n_stages'],
    )

    combos = list(itertools.product(cfg['beta_sets'].items(), cfg['base_beta_sweep']))
    rng = np.random.default_rng(cfg['seed'])
    seeds = rng.integers(0, 1_000_000, len(combos)).tolist()
    tasks, labels = [], []
    for ((name, betas), bb), s in zip(combos, seeds):
        pars = dict(base_beta=bb, beta0=betas[0], beta1=betas[1], beta2=betas[2],
                    sus_after_1=cfg['sus_after_1'], sus_after_2=cfg['sus_after_2'],
                    sus_after_3plus=cfg['sus_after_3plus'],
                    maternal_immunity_efficacy=cfg['maternal_immunity_efficacy'],
                    maternal_immunity_mean_duration_days=cfg['maternal_mean_duration_days'])
        tasks.append((sim_config, pars, int(s), WINDOW))
        labels.append((name, bb))

    tgt_ir = np.array([process_incidence_maled.load_targets(cfg['site'])['ir_by_age'].loc[b, 'IR'] for b in BINS])
    print(f"Forward-running {len(combos)} (beta-set x base_beta) combos, {cfg['n_agents']} agents each.")
    with get_context('spawn').Pool(processes=min(len(tasks), 8)) as pool:
        outs = pool.map(_run_one_replicate, tasks)

    rows, jsonl = [], (out_root / 'outputs' / 'sweep.jsonl')
    with jsonl.open('w') as f:
        for (name, bb), mo in zip(labels, outs):
            ir = [float(mo['ir_by_age'].loc[b, 'IR']) for b in BINS]
            r = dict(beta_set=name, base_beta=bb, symp_ir=ir,
                     first_med=float(mo['first_infection']['median']))
            rows.append(r); f.write(json.dumps(r) + '\n')

    print("\n=== symptomatic IR by age — low-<6m / 6-11mo-peak / declining-tail? ===")
    print(f"  {'DATA':<26} {[round(v,2) for v in tgt_ir]}")
    any_peak = False
    for r in rows:
        ir = np.array(r['symp_ir'])
        peaked = (ir[1] > ir[0]) and (ir[1] > ir[2]) and (ir[2] > ir[3])  # rise to 6-11, then monotone decline
        any_peak |= peaked
        print(f"  {r['beta_set']:<16} beta={r['base_beta']:.2f}: {[round(v,2) for v in r['symp_ir']]}  "
              f"first-med={r['first_med']:.1f} {'<-- PEAK+decline' if peaked else ''}")
    print(f"\n  FEASIBILITY: {'SHAPE REPRODUCED — peaked age curve + maternal works under homogeneous mixing' if any_peak else 'shape not yet reproduced in this grid'}")

    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BINS))
    ax.plot(x, tgt_ir, 'r*-', ms=16, lw=2.5, label='MAL-ED data', zorder=5)
    for r in rows:
        ax.plot(x, r['symp_ir'], 'o-', alpha=0.7, label=f"{r['beta_set']}, beta={r['base_beta']}")
    ax.set_xticks(x); ax.set_xticklabels(BINS); ax.set_ylabel('Symptomatic IR /100 PM')
    ax.set_title('Exp 05 — peaked age-symptom curve + strong maternal (homogeneous mixing)\n'
                 'does it reproduce the data shape? (data in red)')
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_root / 'figures' / 'feasibility_shape.png', dpi=150)
    print(f"\nwrote {out_root / 'figures' / 'feasibility_shape.png'}")


if __name__ == '__main__':
    main()
