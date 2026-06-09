"""
Exp 11 — analysis: did titer maternal let the infection-number model make the peak?

Reads the infnum+titer Poisson study (no sims re-run), pulls the best trial, and reports
the shape scorecard vs the MAL-ED target, overlaying the exp 09/10 infection-number
(Erlang) best fits (which were <6m-high / flat) so the effect of erlang->titer is visible.

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
DB = CALIB_DIR / 'rota_maled_bangladesh_infnum_titer_poisson.db'
STUDY = 'rota_maled_bangladesh_infnum_titer_poisson'
INFNUM_ERLANG_EXP09 = [4.3, 3.8, 1.8, 1.3]    # exp-09 infnum (Erlang) best: <6m-high
INFNUM_ERLANG_EXP10 = [2.5, 3.1, 1.6, 1.1]    # exp-10 infnum (Erlang, PT-fix) best: flat


def multinomial_shape_ll(obs, p):
    p = np.clip(np.asarray(p, float), 1e-9, None); p = p / p.sum()
    return float(np.sum(np.asarray(obs, float) * np.log(p)))


def main():
    targets = P.load_targets('bangladesh')
    tir = targets['ir_by_age']
    tgt_ir = tir['IR'].values; tgt_cases = tir['cases'].values.astype(float)
    tgt_pt = tir['PT'].values.astype(float); tgt_p = tgt_ir / tgt_ir.sum()
    tgt_fi = targets['first_infection']

    study = optuna.load_study(study_name=STUDY, storage=f'sqlite:///{DB}')
    bt = study.best_trial
    ir = np.array(bt.user_attrs['ir_by_age_per_rep'])
    fi = np.array(bt.user_attrs['first_inf_per_rep'])
    med, lo, hi = np.median(ir, 0), *np.percentile(ir, [5, 95], 0)
    med_fi = np.median(fi, 0)
    model_ir_df = pd.DataFrame({'cases': [0]*4, 'PT': tgt_pt, 'IR': med}, index=BINS)
    dev = P.gof_incidence_poisson(model_ir_df, tir)
    p = med / med.sum()
    res = dict(
        best_trial=bt.number, gof_total=float(bt.value), poisson_dev=dev,
        med_ir=med.tolist(), lo_ir=lo.tolist(), hi_ir=hi.tolist(),
        peak_bin=BINS[int(np.argmax(med))],
        norm_l1=float(np.sum(np.abs(p - tgt_p))),
        cosine=float(np.dot(p, tgt_p) / (np.linalg.norm(p) * np.linalg.norm(tgt_p))),
        shape_loglik=multinomial_shape_ll(tgt_cases, p),
        first_med=float(med_fi[1]), params=bt.params,
    )
    flipped = (res['peak_bin'] == '6-11 m') and (res['cosine'] > 0.97)
    print(f"\n  TARGET IR {[round(v,2) for v in tgt_ir]}  peak 6-11m  first-inf {tgt_fi['median']:.2f}mo")
    print(f"  infnum+TITER best (trial {bt.number}): dev {dev:.1f}")
    print(f"    IR {[round(v,2) for v in med]}  peak {res['peak_bin']}  "
          f"L1 {res['norm_l1']:.3f}  cosine {res['cosine']:.4f}  first-inf {res['first_med']:.2f}mo")
    print(f"  (exp09 infnum-Erlang {INFNUM_ERLANG_EXP09} peak <6m;  exp10 {INFNUM_ERLANG_EXP10} flat)")
    print(f"\n  VERDICT: titer maternal {'FLIPS it -- infection-number makes the peak' if flipped else 'does NOT flip it -- still no real peak'}")
    print(f"  titer params: median={bt.params.get('maternal_titer_median'):.1f} "
          f"gsd={bt.params.get('maternal_titer_gsd'):.2f} "
          f"half_life={bt.params.get('maternal_titer_half_life_days'):.1f}d "
          f"hill={bt.params.get('maternal_hill_slope'):.2f}; base_beta={bt.params.get('base_beta'):.3f}")

    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, tgt_ir, 'r*-', ms=18, lw=2.5, label='MAL-ED data', zorder=6)
    ax.plot(x, INFNUM_ERLANG_EXP09, 'o--', color='#762a83', alpha=0.4, lw=1.5,
            label='infnum + Erlang (exp 09): <6m-high', zorder=3)
    ax.plot(x, med, 'o-', color='#2166ac', lw=2,
            label=f'infnum + TITER (exp 11)  (dev {dev:.1f})', zorder=4)
    ax.fill_between(x, lo, hi, color='#2166ac', alpha=0.15, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM'); ax.set_xlabel('Age bin')
    ax.set_title('Exp 11 — infection-number + titer maternal (homogeneous mixing) vs MAL-ED\n'
                 'does titer alone make the peak that Erlang could not?')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'figures' / 'infnum_titer_fit.png', dpi=150)
    json.dump(res, (HERE / 'outputs' / 'scorecard.json').open('w'), indent=2)
    print(f"\nwrote {HERE/'figures'/'infnum_titer_fit.png'} + scorecard.json")


if __name__ == '__main__':
    main()
