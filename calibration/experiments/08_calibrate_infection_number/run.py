"""
Exp 07 vs 08 — head-to-head analysis of the two calibrated models (see README.md).

Loads both Optuna studies (already run on the VM; no sims re-run here), pulls the
best trial of each, and compares them on:
  - scalar joint GOF (the calibration objective) + its inc/first breakdown
  - the actual symptomatic-IR-by-age profile (median + 5-95% rep band) vs MAL-ED
  - a SHAPE scorecard that the scalar GOF does not directly capture:
      * peak-bin location (should be 6-11mo)
      * L1 distance between level-normalized profiles
      * multinomial shape log-likelihood of the observed case counts under each
        model's predicted age-proportions (higher = better)
      * cosine similarity of normalized profiles
Writes per-model + comparison figures and a scorecard JSON.

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
EXP07 = HERE.parent / '07_calibrate_peaked_age'
CALIB_DIR = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR))
import process_incidence_maled  # noqa: E402

BINS = process_incidence_maled.MALED_AGE_BINS
SITE = 'bangladesh'
MODELS = {
    'peaked_age':       dict(db='rota_maled_bangladesh_erlang6.db',
                             study='rota_maled_bangladesh_erlang6',
                             label='Peaked age (exp 07)'),
    'infection_number': dict(db='rota_maled_bangladesh_infnum_erlang6.db',
                             study='rota_maled_bangladesh_infnum_erlang6',
                             label='Infection-number (exp 08)'),
}


def multinomial_shape_ll(obs_counts, p_model):
    """Sum_k obs_k * log(p_model_k): how well the model's age-PROPORTIONS explain
    the observed case distribution. Level-independent (pure shape). Higher better."""
    p = np.clip(np.asarray(p_model, float), 1e-9, None)
    p = p / p.sum()
    return float(np.sum(np.asarray(obs_counts, float) * np.log(p)))


def main():
    targets = process_incidence_maled.load_targets(SITE)
    tgt_ir = np.array([targets['ir_by_age'].loc[b, 'IR'] for b in BINS])
    tgt_cases = np.array([targets['ir_by_age'].loc[b, 'cases'] for b in BINS], float)
    tgt_fi = targets['first_infection']
    tgt_p = tgt_ir / tgt_ir.sum()

    results = {}
    for key, m in MODELS.items():
        study = optuna.load_study(study_name=m['study'],
                                  storage=f"sqlite:///{CALIB_DIR / m['db']}")
        bt = study.best_trial
        ir_per_rep = np.array(bt.user_attrs['ir_by_age_per_rep'])   # n_reps x 4
        fi_per_rep = np.array(bt.user_attrs['first_inf_per_rep'])    # n_reps x 3
        med_ir = np.median(ir_per_rep, axis=0)
        lo_ir, hi_ir = np.percentile(ir_per_rep, [5, 95], axis=0)
        med_fi = np.median(fi_per_rep, axis=0)
        gof_inc = float(np.median(bt.user_attrs['per_rep_gof_inc']))
        gof_first = float(np.median(bt.user_attrs['per_rep_gof_first']))

        p_model = med_ir / med_ir.sum()
        peak_bin = BINS[int(np.argmax(med_ir))]
        l1 = float(np.sum(np.abs(p_model - tgt_p)))
        cos = float(np.dot(p_model, tgt_p) /
                    (np.linalg.norm(p_model) * np.linalg.norm(tgt_p)))
        shape_ll = multinomial_shape_ll(tgt_cases, p_model)

        results[key] = dict(
            label=m['label'], best_trial=bt.number, gof_total=float(bt.value),
            gof_inc=gof_inc, gof_first=gof_first,
            med_ir=med_ir.tolist(), lo_ir=lo_ir.tolist(), hi_ir=hi_ir.tolist(),
            med_first=dict(q25=float(med_fi[0]), median=float(med_fi[1]), q75=float(med_fi[2])),
            peak_bin=peak_bin, norm_l1=l1, cosine=cos, shape_loglik=shape_ll,
            params=bt.params,
        )

    # ---- scorecard ----
    print(f"\n{'':22}{'PEAK-AGE':>16}{'INFECTION-#':>16}{'  (target/best)':>16}")
    def row(name, k, fmt, best='lo'):
        a, b = results['peaked_age'][k], results['infection_number'][k]
        star = '  <-' + ('A' if (a < b) == (best == 'lo') else 'B')
        print(f"{name:22}{fmt.format(a):>16}{fmt.format(b):>16}{star:>16}")
    print(f"  TARGET IR by age: {[round(v,2) for v in tgt_ir]}  peak={BINS[int(np.argmax(tgt_ir))]}")
    row('joint GOF (scalar)',  'gof_total',   '{:.3f}', 'lo')
    row('  gof_incidence',     'gof_inc',     '{:.3f}', 'lo')
    row('  gof_first_inf',     'gof_first',   '{:.3f}', 'lo')
    print(f"{'peak bin':22}{results['peaked_age']['peak_bin']:>16}"
          f"{results['infection_number']['peak_bin']:>16}{'  6-11mo':>16}")
    row('shape L1 (norm)',     'norm_l1',     '{:.3f}', 'lo')
    row('shape cosine',        'cosine',      '{:.4f}', 'hi')
    row('multinomial shapeLL', 'shape_loglik','{:.2f}', 'hi')
    for k in ('peaked_age', 'infection_number'):
        r = results[k]
        print(f"\n  {r['label']} best-fit IR: {[round(v,2) for v in r['med_ir']]}  "
              f"first-inf med={r['med_first']['median']:.2f}mo (target {tgt_fi['median']:.2f})")

    # ---- figures ----
    x = np.arange(len(BINS))
    colors = {'peaked_age': '#1b7837', 'infection_number': '#762a83'}
    # combined comparison
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(x, tgt_ir, 'r*-', ms=18, lw=2.5, label='MAL-ED data', zorder=6)
    for k, r in results.items():
        ax.plot(x, r['med_ir'], 'o-', color=colors[k], lw=2,
                label=f"{r['label']}  (GOF {r['gof_total']:.2f})", zorder=4)
        ax.fill_between(x, r['lo_ir'], r['hi_ir'], color=colors[k], alpha=0.15, zorder=2)
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM'); ax.set_xlabel('Age bin')
    ax.set_title('Exp 07 vs 08 — calibrated best fits vs MAL-ED Bangladesh\n'
                 '(band = 5-95% across replicates)')
    ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    (HERE / 'figures').mkdir(parents=True, exist_ok=True)
    fig.savefig(HERE / 'figures' / 'compare_best_fits.png', dpi=150)

    # infection-number's own best-fit figure (for this experiment's SUMMARY)
    r = results['infection_number']
    f, a = plt.subplots(figsize=(8, 5.5))
    a.plot(x, tgt_ir, 'r*-', ms=16, lw=2, label='MAL-ED data', zorder=5)
    a.plot(x, r['med_ir'], 'o-', color=colors['infection_number'], lw=2, label=r['label'], zorder=4)
    a.fill_between(x, r['lo_ir'], r['hi_ir'], color=colors['infection_number'], alpha=0.18)
    a.set_xticks(x); a.set_xticklabels(BINS); a.set_ylabel('Symptomatic IR /100 PM')
    a.set_title(f"Exp 08 infection-number best fit (GOF {r['gof_total']:.2f}, peak {r['peak_bin']})\n"
                f"shape: cosine {r['cosine']:.3f}, L1 {r['norm_l1']:.3f} (flatter than data)")
    a.legend(frameon=False); a.spines[['top', 'right']].set_visible(False)
    f.tight_layout()
    f.savefig(HERE / 'figures' / 'best_fit.png', dpi=150)

    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    with (HERE / 'outputs' / 'scorecard.json').open('w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {HERE/'figures'/'compare_best_fits.png'} + best_fit.png + scorecard.json")


if __name__ == '__main__':
    main()
