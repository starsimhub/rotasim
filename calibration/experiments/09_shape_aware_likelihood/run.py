"""
Exp 09 — head-to-head of the two models re-calibrated under the per-bin Poisson
(shape-aware) likelihood (see README.md).

Calibrations run on the VM via:
  cd calibration
  python experiments/09_shape_aware_likelihood/enqueue_seeds.py   # peaked shape ladder
  python calibrate_maled.py --site bangladesh --symptom-model age_only \
      --maternal-n-stages 6 --fit-target poisson --n-trials 40 --n-reps 20 --n-jobs 1
  python calibrate_maled.py --site bangladesh --symptom-model infection_number \
      --maternal-n-stages 6 --fit-target poisson --n-trials 40 --n-reps 20 --n-jobs 1

This script reads both _poisson studies (no sims re-run), pulls each best trial, and
reports: Poisson deviance, fitted level (total expected cases vs observed 161), the
shape scorecard (peak bin, normalized L1, cosine, multinomial shape-LL), and the
age-at-first-infection median as a held-out diagnostic (it is only lightly weighted
under the Poisson objective). Overlays both best fits + the old exp-07 squared-log
peaked fit (to show whether the level overshoot shrank).

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
    'peaked_age':       dict(db='rota_maled_bangladesh_erlang6_poisson.db',
                             study='rota_maled_bangladesh_erlang6_poisson',
                             label='Peaked age (exp 09)', color='#1b7837'),
    'infection_number': dict(db='rota_maled_bangladesh_infnum_erlang6_poisson.db',
                             study='rota_maled_bangladesh_infnum_erlang6_poisson',
                             label='Infection-number (exp 09)', color='#762a83'),
}
OLD_PEAKED_IR = [6.09, 11.48, 5.72, 0.35]   # exp-07 squared-log best (for overshoot ref)


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
        exp_total = float(np.sum(med / 100.0 * tgt_pt))     # total expected cases
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
    print(f"\n  TARGET: IR {[round(v,2) for v in tgt_ir]}  total cases {int(tgt_cases.sum())}  "
          f"peak {BINS[int(np.argmax(tgt_ir))]}  first-inf med {tgt_fi['median']:.2f}mo")
    print(f"{'':22}{'PEAK-AGE':>14}{'INFECTION-#':>14}")
    def row(name, k, fmt):
        print(f"{name:22}{fmt.format(pk[k]):>14}{fmt.format(inf[k]):>14}")
    row('objective (poisson)', 'gof_total',   '{:.2f}')
    row('  poisson deviance',  'poisson_dev', '{:.2f}')
    row('  exp. total cases',  'exp_total_cases', '{:.0f}')
    print(f"{'  peak bin':22}{pk['peak_bin']:>14}{inf['peak_bin']:>14}")
    row('  shape L1 (norm)',   'norm_l1',     '{:.3f}')
    row('  shape cosine',      'cosine',      '{:.4f}')
    row('  multinom shape-LL', 'shape_loglik','{:.1f}')
    row('  first-inf med (mo)','first_med',   '{:.2f}')
    for r in (pk, inf):
        print(f"  {r['label']} IR {[round(v,2) for v in r['med_ir']]}")

    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, tgt_ir, 'r*-', ms=18, lw=2.5, label='MAL-ED data', zorder=6)
    ax.plot(x, OLD_PEAKED_IR, 'o--', color='#1b7837', alpha=0.4, lw=1.5,
            label='Peaked age (exp 07, squared-log)', zorder=3)
    for r in (pk, inf):
        ax.plot(x, r['med_ir'], 'o-', color=r['color'], lw=2,
                label=f"{r['label']}  (dev {r['poisson_dev']:.1f})", zorder=4)
        ax.fill_between(x, r['lo_ir'], r['hi_ir'], color=r['color'], alpha=0.15, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM'); ax.set_xlabel('Age bin')
    ax.set_title('Exp 09 — Poisson-recalibrated fits vs MAL-ED Bangladesh\n'
                 '(dashed = exp-07 squared-log peaked fit, for level reference)')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'figures' / 'compare_poisson_fits.png', dpi=150)
    json.dump(results, (HERE / 'outputs' / 'scorecard.json').open('w'), indent=2)
    print(f"\nwrote {HERE/'figures'/'compare_poisson_fits.png'} + scorecard.json")


if __name__ == '__main__':
    main()
