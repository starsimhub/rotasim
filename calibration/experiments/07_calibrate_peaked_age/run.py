"""
Exp 07 — analysis of the calibrated peaked-age model (see README.md).

The calibration itself ran on the covaguest VM via the repo's calibrate_maled.py:

  cd calibration
  python experiments/07_calibrate_peaked_age/enqueue_seed.py   # peaked starting trial
  python calibrate_maled.py --site bangladesh --symptom-model age_only \
      --maternal-n-stages 6 --fit-target joint --n-trials 40 --n-reps 20 --n-jobs 1
  # -> writes study 'rota_maled_bangladesh_erlang6' to rota_maled_bangladesh_erlang6.db

This script reads that study (no sims re-run), pulls the best trial, and plots its
symptomatic-IR-by-age best fit (median + 5-95% rep band) against the MAL-ED target.

  python run.py
"""
import sys
import json
import pathlib

import numpy as np
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR))
import process_incidence_maled  # noqa: E402

BINS = process_incidence_maled.MALED_AGE_BINS
DB = CALIB_DIR / 'rota_maled_bangladesh_erlang6.db'
STUDY = 'rota_maled_bangladesh_erlang6'


def main():
    targets = process_incidence_maled.load_targets('bangladesh')
    tgt_ir = np.array([targets['ir_by_age'].loc[b, 'IR'] for b in BINS])
    tgt_fi = targets['first_infection']
    tgt_p = tgt_ir / tgt_ir.sum()

    study = optuna.load_study(study_name=STUDY, storage=f'sqlite:///{DB}')
    bt = study.best_trial
    ir = np.array(bt.user_attrs['ir_by_age_per_rep'])     # n_reps x 4
    fi = np.array(bt.user_attrs['first_inf_per_rep'])      # n_reps x 3
    med, lo, hi = np.median(ir, 0), *np.percentile(ir, [5, 95], 0)
    med_fi = np.median(fi, 0)
    p = med / med.sum()

    fit = dict(
        best_trial=bt.number, gof_total=float(bt.value),
        gof_inc=float(np.median(bt.user_attrs['per_rep_gof_inc'])),
        gof_first=float(np.median(bt.user_attrs['per_rep_gof_first'])),
        med_ir=med.tolist(), peak_bin=BINS[int(np.argmax(med))],
        norm_l1=float(np.sum(np.abs(p - tgt_p))),
        cosine=float(np.dot(p, tgt_p) / (np.linalg.norm(p) * np.linalg.norm(tgt_p))),
        first_med=float(med_fi[1]), params=bt.params,
    )
    print(f"Best trial {bt.number}: GOF {fit['gof_total']:.3f} "
          f"(inc {fit['gof_inc']:.3f}, first {fit['gof_first']:.3f})")
    print(f"  best-fit IR {[round(v,2) for v in med]} vs target {[round(v,2) for v in tgt_ir]}")
    print(f"  peak bin {fit['peak_bin']} (target {BINS[int(np.argmax(tgt_ir))]}), "
          f"shape L1 {fit['norm_l1']:.3f}, cosine {fit['cosine']:.4f}")
    print(f"  first-inf median {fit['first_med']:.2f}mo (target {tgt_fi['median']:.2f})")
    print(f"  params: {json.dumps(fit['params'], indent=0)}")

    x = np.arange(len(BINS))
    f, a = plt.subplots(figsize=(8, 5.5))
    a.plot(x, tgt_ir, 'r*-', ms=16, lw=2, label='MAL-ED data', zorder=5)
    a.plot(x, med, 'o-', color='#1b7837', lw=2, label='Peaked age best fit', zorder=4)
    a.fill_between(x, lo, hi, color='#1b7837', alpha=0.18, label='5-95% reps')
    a.set_xticks(x); a.set_xticklabels(BINS); a.set_ylabel('Symptomatic IR /100 PM')
    a.set_title(f"Exp 07 peaked-age best fit (GOF {fit['gof_total']:.2f}, peak {fit['peak_bin']})\n"
                f"shape: cosine {fit['cosine']:.3f}, L1 {fit['norm_l1']:.3f}")
    a.legend(frameon=False); a.spines[['top', 'right']].set_visible(False)
    f.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    f.savefig(HERE / 'figures' / 'best_fit.png', dpi=150)
    json.dump(fit, (HERE / 'outputs' / 'fit.json').open('w'), indent=2)
    print(f"wrote {HERE/'figures'/'best_fit.png'}")


if __name__ == '__main__':
    main()
