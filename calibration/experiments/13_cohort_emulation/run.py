"""
Exp 13 — analysis: infection-number + titer maternal under the MAL-ED cohort observation.

Reads the cohort study, pulls the best trial, and reports the symptomatic-IR shape, the
KM age-at-first-DETECTION median, and the repeat-detected fraction vs their targets, with
the IR overlay. (Objective: Poisson IR + median-only KM first-detection + repeat-fraction.)

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
DB = CALIB_DIR / 'rota_maled_bangladesh_infnum_titer_cohort.db'
STUDY = 'rota_maled_bangladesh_infnum_titer_cohort'


def main():
    targets = P.load_targets('bangladesh')
    tir = targets['ir_by_age']
    tgt_ir = tir['IR'].values; tgt_p = tgt_ir / tgt_ir.sum()
    km = targets['first_infection_km']; rf = targets['repeat_frac']

    study = optuna.load_study(study_name=STUDY, storage=f'sqlite:///{DB}')
    bt = study.best_trial
    ir = np.array(bt.user_attrs['ir_by_age_per_rep'])
    fi = np.array(bt.user_attrs['first_inf_per_rep'])
    rep = bt.user_attrs.get('repeat_frac_per_rep')
    med, lo, hi = np.median(ir, 0), *np.percentile(ir, [5, 95], 0)
    med_first = float(np.nanmedian(fi[:, 1]))
    med_rep = float(np.nanmedian([r for r in rep if r is not None])) if rep else float('nan')
    p = med / med.sum()
    cosine = float(np.dot(p, tgt_p) / (np.linalg.norm(p) * np.linalg.norm(tgt_p)))

    print(f"\n  TARGET: IR {[round(v,2) for v in tgt_ir]} (peak 6-11m)  "
          f"KM first-detect med {km['median']:.2f}mo  repeat-frac {rf['frac']:.3f}")
    print(f"  best trial {bt.number}: objective {bt.value:.2f}")
    print(f"    IR {[round(v,2) for v in med]}  peak {BINS[int(np.argmax(med))]}  cosine {cosine:.4f}")
    print(f"    KM first-detect med {med_first:.2f}mo (target {km['median']:.2f})")
    print(f"    repeat-frac {med_rep:.3f} (target {rf['frac']:.3f}; se {rf['se']:.3f})")
    print(f"    base_beta {bt.params.get('base_beta'):.3f}  titer median {bt.params.get('maternal_titer_median'):.1f} "
          f"half_life {bt.params.get('maternal_titer_half_life_days'):.1f}d hill {bt.params.get('maternal_hill_slope'):.2f}")

    x = np.arange(len(BINS))
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, tgt_ir, 'r*-', ms=18, lw=2.5, label='MAL-ED data', zorder=6)
    ax.plot(x, med, 'o-', color='#0571b0', lw=2,
            label=f'infnum+titer, cohort obs (exp 13)', zorder=4)
    ax.fill_between(x, lo, hi, color='#0571b0', alpha=0.15, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM'); ax.set_xlabel('Age bin')
    ax.set_title('Exp 13 — infection-number + titer maternal under MAL-ED cohort observation\n'
                 f'(cosine {cosine:.3f}; KM first-detect {med_first:.1f}mo vs {km["median"]:.1f}; '
                 f'repeat {med_rep:.2f} vs {rf["frac"]:.2f})')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'figures' / 'cohort_fit.png', dpi=150)
    json.dump(dict(best_trial=bt.number, objective=float(bt.value), med_ir=med.tolist(),
                   cosine=cosine, km_first_med=med_first, repeat_frac=med_rep,
                   params=bt.params), (HERE / 'outputs' / 'scorecard.json').open('w'), indent=2)
    print(f"\nwrote {HERE/'figures'/'cohort_fit.png'} + scorecard.json")


if __name__ == '__main__':
    main()
