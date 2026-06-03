"""
Compare the two best-fit MAL-ED Bangladesh models against their two HEADLINE
targets (not the per-age-bin detail):

  1. Overall (pooled, 0-36 mo) symptomatic incidence  [per 100 person-months]
  2. Age at first (detected) infection                [Q25 / median / Q75, months]

Runs the best symptomatic-IR trial (#19) and best first-infection trial (#17)
from the single-phase baseline studies with fresh seeds, then plots each model
vs its explicit target. Error bars = 5-95th percentile across replicates.

Run on the VM (needs the baseline study DBs + cores):
  python plot_explicit_targets.py --n-reps 20
"""
import argparse
from multiprocessing import get_context

import numpy as np
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from calibrate_maled import (
    _run_one_replicate, SITE_DEMOGRAPHICS,
    FIXED_REPORTING_RATE, FIXED_CONSTANT_SEVERITY,
    process_incidence_maled, thisdir,
)

SITE = 'bangladesh'
CAL_WINDOW = (5.0, 10.0)
# (label, study_db, study_name, trial_number, color)
MODELS = [
    ('Symptomatic-IR fit (#19)',  'rota_maled_bangladesh_symptomatic_ir.db',
     'rota_maled_bangladesh_symptomatic_ir',  19, '#1f77b4'),
    ('First-infection fit (#17)', 'rota_maled_bangladesh_first_infection.db',
     'rota_maled_bangladesh_first_infection', 17, '#d62728'),
]
TARGET_COLOR = '#444444'


def run_model(sim_pars, sim_config, n_reps, seed0):
    seeds = (np.arange(n_reps) + seed0 * 1000 + 1).tolist()
    args_list = [(sim_config, sim_pars, int(s), CAL_WINDOW) for s in seeds]
    with get_context('spawn').Pool(processes=n_reps) as pool:
        return pool.map(_run_one_replicate, args_list)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-reps', type=int, default=20)
    ap.add_argument('--p-asymp-detect', type=float, default=0.4)
    args = ap.parse_args()

    targets = process_incidence_maled.load_targets(SITE)
    demo = SITE_DEMOGRAPHICS[SITE]
    sim_config = dict(
        n_agents=100_000, start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY, reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'), p_asymp_detect=args.p_asymp_detect,
    )

    # --- Targets ---
    t_cases = float(targets['ir_by_age']['cases'].sum())
    t_pt    = float(targets['ir_by_age']['PT'].sum())
    t_pool  = t_cases / t_pt * 100.0
    t_fi    = targets['first_infection']

    # --- Run each model, collect per-rep pooled IR + first-inf quartiles ---
    results = {}
    for i, (label, db, name, num, color) in enumerate(MODELS):
        study = optuna.load_study(study_name=name, storage='sqlite:///' + db)
        trial = next(t for t in study.trials if t.number == num)
        print(f"Running {label}: GOF={trial.value:.4f}")
        outs = run_model(dict(trial.params), sim_config, args.n_reps, i)
        pooled = np.array([mo['ir_by_age']['cases'].sum() / mo['ir_by_age']['PT'].sum() * 100.0
                           for mo in outs])
        q = {k: np.array([mo['first_infection'][k] for mo in outs])
             for k in ('q25', 'median', 'q75')}
        results[label] = dict(color=color, pooled=pooled, q=q)

    def stat(arr):
        return float(np.median(arr)), float(np.percentile(arr, 5)), float(np.percentile(arr, 95))

    # ---------------- Figure: 2 panels ----------------
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))

    # Panel A: overall symptomatic incidence
    labels = ['MAL-ED\ntarget'] + [lab.split(' (')[0] for lab, *_ in MODELS]
    colors = [TARGET_COLOR] + [results[lab]['color'] for lab, *_ in MODELS]
    vals, err_lo, err_hi = [t_pool], [0], [0]
    for lab, *_ in MODELS:
        m, lo, hi = stat(results[lab]['pooled'])
        vals.append(m); err_lo.append(m - lo); err_hi.append(hi - m)
    xA = np.arange(len(labels))
    axA.bar(xA, vals, color=colors, width=0.6,
            yerr=[err_lo, err_hi], capsize=5, ecolor='#222222')
    for x, v in zip(xA, vals):
        axA.text(x, v + 0.05, f'{v:.2f}', ha='center', va='bottom', fontsize=10)
    axA.set_xticks(xA); axA.set_xticklabels(labels)
    axA.set_ylabel('Symptomatic incidence (per 100 person-months)')
    axA.set_title('(1) Overall symptomatic incidence, 0-36 mo\n'
                  f'target = {t_pool:.2f}  ({int(t_cases)} cases / {int(t_pt)} PT)')
    axA.spines[['top', 'right']].set_visible(False)

    # Panel B: age at first infection, Q25 / median / Q75
    quarts = ['q25', 'median', 'q75']
    qlabels = ['Q25', 'Median', 'Q75']
    series = [('MAL-ED target', TARGET_COLOR, [t_fi[k] for k in quarts], None)]
    for lab, *_ in MODELS:
        meds = [stat(results[lab]['q'][k]) for k in quarts]
        series.append((lab.split(' (')[0], results[lab]['color'],
                       [m[0] for m in meds],
                       ([m[0] - m[1] for m in meds], [m[2] - m[0] for m in meds])))
    xB = np.arange(len(quarts)); w = 0.26
    for j, (slab, scol, svals, serr) in enumerate(series):
        off = (j - 1) * w
        axB.bar(xB + off, svals, w, color=scol, label=slab,
                yerr=serr, capsize=4, ecolor='#222222')
    axB.set_xticks(xB); axB.set_xticklabels(qlabels)
    axB.set_ylabel('Age at first detected infection (months)')
    axB.set_title('(2) Age at first infection\n'
                  f'target Q25/med/Q75 = {t_fi["q25"]:.1f}/{t_fi["median"]:.1f}/{t_fi["q75"]:.1f}')
    axB.legend(frameon=False)
    axB.spines[['top', 'right']].set_visible(False)

    fig.suptitle('MAL-ED Bangladesh: each best-fit model vs. its explicit headline target',
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig('explicit_targets_bangladesh.png', dpi=150)
    print('wrote explicit_targets_bangladesh.png')

    # ---------------- Text table ----------------
    print('\n' + '=' * 70)
    print(f"{'metric':<34}{'target':>10}{'#19 (IR)':>13}{'#17 (first)':>13}")
    print('-' * 70)
    mp = {lab.split(' (')[0]: results[lab] for lab, *_ in MODELS}
    s19 = stat(mp['Symptomatic-IR fit']['pooled'])
    s17 = stat(mp['First-infection fit']['pooled'])
    print(f"{'Overall sympt. IR (/100 PM)':<34}{t_pool:>10.2f}{s19[0]:>13.2f}{s17[0]:>13.2f}")
    for k, kl in zip(quarts, qlabels):
        a = stat(mp['Symptomatic-IR fit']['q'][k])
        b = stat(mp['First-infection fit']['q'][k])
        print(f"{'First-inf ' + kl + ' (mo)':<34}{t_fi[k]:>10.2f}{a[0]:>13.2f}{b[0]:>13.2f}")
    print('=' * 70)


if __name__ == '__main__':
    main()
