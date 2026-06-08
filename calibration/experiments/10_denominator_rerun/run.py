"""
Exp 10 — analysis of the corrected-denominator re-run (see README.md).

Reads the two *_ptfix Poisson studies (no sims re-run), pulls each best trial, and
reports the same scorecard as exp 09 (Poisson deviance, fitted level vs observed 161,
peak bin, shape L1/cosine/multinomial-LL, first-infection median), overlaying the
exp-09 (old-denominator) peaked fit so the denominator-driven shift is visible.

  python run.py
"""
import sys
import json
import pathlib

import numpy as np
import pandas as pd
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR))
import process_incidence_maled as P  # noqa: E402

BINS = P.MALED_AGE_BINS
MODELS = {
    'peaked_age':       dict(db='rota_maled_bangladesh_erlang6_poisson_ptfix.db',
                             study='rota_maled_bangladesh_erlang6_poisson',
                             label='Peaked age (exp 10, PT-fix)', color='#1b7837'),
    'infection_number': dict(db='rota_maled_bangladesh_infnum_erlang6_poisson_ptfix.db',
                             study='rota_maled_bangladesh_infnum_erlang6_poisson',
                             label='Infection-number (exp 10, PT-fix)', color='#762a83'),
}
EXP09_PEAKED_IR = [1.1, 6.46, 2.0, 0.0]   # exp-09 peaked best (old denominator), for shift ref


def multinomial_shape_ll(obs_counts, p_model):
    p = np.clip(np.asarray(p_model, float), 1e-9, None); p = p / p.sum()
    return float(np.sum(np.asarray(obs_counts, float) * np.log(p)))


def main():
    targets = P.load_targets('bangladesh')
    tir = targets['ir_by_age']
    tgt_ir = tir['IR'].values
    tgt_cases = tir['cases'].values.astype(float)
    tgt_pt = tir['PT'].values.astype(float)
    tgt_p = tgt_ir / tgt_ir.sum()
    tgt_fi = targets['first_infection']

    results = {}
    for key, m in MODELS.items():
        study = optuna.load_study(study_name=m['study'],
                                  storage=f"sqlite:///{CALIB_DIR / m['db']}")
        bt = study.best_trial
        ir = np.array(bt.user_attrs['ir_by_age_per_rep'])
        fi = np.array(bt.user_attrs['first_inf_per_rep'])
        med, lo, hi = np.median(ir, 0), *np.percentile(ir, [5, 95], 0)
        med_fi = np.median(fi, 0)
        model_ir_df = pd.DataFrame({'cases': [0]*4, 'PT': tgt_pt, 'IR': med}, index=BINS)
        dev = P.gof_incidence_poisson(model_ir_df, tir)
        exp_total = float(np.sum(med / 100.0 * tgt_pt))
        p = med / med.sum()
        results[key] = dict(
            label=m['label'], color=m['color'], best_trial=bt.number,
            gof_total=float(bt.value), poisson_dev=dev, exp_total_cases=exp_total,
            med_ir=med.tolist(), lo_ir=lo.tolist(), hi_ir=hi.tolist(),
            peak_bin=BINS[int(np.argmax(med))],
            norm_l1=float(np.sum(np.abs(p - tgt_p))),
            cosine=float(np.dot(p, tgt_p) / (np.linalg.norm(p) * np.linalg.norm(tgt_p))),
            shape_loglik=multinomial_shape_ll(tgt_cases, p),
            first_med=float(med_fi[1]), params=bt.params,
        )

    pk, inf = results['peaked_age'], results['infection_number']
    print(f"\n  TARGET: IR {[round(v,2) for v in tgt_ir]}  total {int(tgt_cases.sum())}  "
          f"peak {BINS[int(np.argmax(tgt_ir))]}  first-inf {tgt_fi['median']:.2f}mo")
    print(f"{'':22}{'PEAK-AGE':>14}{'INFECTION-#':>14}{'   exp09 peaked':>16}")
    def row(name, k, fmt, e09=''):
        print(f"{name:22}{fmt.format(pk[k]):>14}{fmt.format(inf[k]):>14}{e09:>16}")
    row('poisson deviance',  'poisson_dev', '{:.2f}', '19.30')
    row('exp. total cases',  'exp_total_cases', '{:.0f}', '155')
    print(f"{'peak bin':22}{pk['peak_bin']:>14}{inf['peak_bin']:>14}{'6-11 m':>16}")
    row('shape L1 (norm)',   'norm_l1',     '{:.3f}', '0.251')
    row('shape cosine',      'cosine',      '{:.4f}', '0.982')
    row('first-inf med (mo)','first_med',   '{:.2f}', '8.22')
    for r in (pk, inf):
        print(f"  {r['label']} IR {[round(v,2) for v in r['med_ir']]}")

    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, tgt_ir, 'r*-', ms=18, lw=2.5, label='MAL-ED data', zorder=6)
    ax.plot(x, EXP09_PEAKED_IR, 'o--', color='#1b7837', alpha=0.4, lw=1.5,
            label='Peaked age (exp 09, old denominator)', zorder=3)
    for r in (pk, inf):
        ax.plot(x, r['med_ir'], 'o-', color=r['color'], lw=2,
                label=f"{r['label']}  (dev {r['poisson_dev']:.1f})", zorder=4)
        ax.fill_between(x, r['lo_ir'], r['hi_ir'], color=r['color'], alpha=0.15, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM'); ax.set_xlabel('Age bin')
    ax.set_title('Exp 10 — corrected-denominator re-run vs MAL-ED Bangladesh\n'
                 '(dashed = exp-09 peaked fit, old denominator)')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'figures' / 'compare_ptfix_fits.png', dpi=150)
    json.dump(results, (HERE / 'outputs' / 'scorecard.json').open('w'), indent=2)
    print(f"\nwrote {HERE/'figures'/'compare_ptfix_fits.png'} + scorecard.json")


if __name__ == '__main__':
    main()
