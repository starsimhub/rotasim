"""
TUTORIAL DEMO — Predicted vaccine impact in Bangladesh under two fitted immunity structures.

The point this demo makes (for a non-modeller audience):
    Two models calibrated to the SAME pre-vaccine MAL-ED data — one where SYMPTOMS depend on
    AGE, one where symptoms depend on INFECTION NUMBER — fit the data equally well, yet they
    predict DIFFERENT vaccine impact. So the assumed disease mechanism matters for policy.

It reuses the real fitted parameters and the real single-strain simulation engine:
  - fitted parameters  : experiments/25_age_binned... and 27_infnum... posteriors
  - simulation engine  : experiments/15_vaccine_toy/vaccine_toy.py  (_build_run)
  - VE definition      : 1 - symptomatic_incidence(vaccinated) / symptomatic_incidence(no vaccine),
                         children <=36 months, over the calibration window (a TOTAL effect).

Vaccine: 2 doses at 2 & 4 months; each seroconversion (prob = --response) advances the agent one
"infection-equivalent". That counter feeds BOTH acquisition (everyone) AND, in the infection-number
model, symptom probability — which is why the two structures diverge.

Run:
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python vaccine_demo_bangladesh.py            # ~few min
  PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python vaccine_demo_bangladesh.py --quick     # fast/rough
"""
import sys, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
vt = sc.importbypath(HERE / 'experiments' / '15_vaccine_toy' / 'vaccine_toy.py')   # the single-strain engine

# Convert a fitted-posterior row (transformed space) into the sim's parameter dict. Inlined here
# (instead of importing hm_calibrate) so the tutorial only needs rotasim, not the historymatching
# package. Maternal shape is held at the identified Bangladesh curve (fix_titer_shape).
FIXED_TITER = dict(median=20.0, gsd=2.3, half_life_days=50.0, hill=4.7)
def untransform(row, model):
    s1 = float(row['sus_after_1']); s2 = s1 * float(row['sus_r2']); s3 = s2 * float(row['sus_r3'])
    p = dict(base_beta=float(np.exp(row['log_base_beta'])),
             sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
             maternal_immunity_efficacy=float(row['maternal_efficacy']),
             maternal_titer_median=FIXED_TITER['median'], maternal_titer_gsd=FIXED_TITER['gsd'],
             maternal_titer_half_life_days=FIXED_TITER['half_life_days'], maternal_hill_slope=FIXED_TITER['hill'])
    if model == 'age_binned':
        p.update(p_symp_age_0_6=float(row['p_symp_age_0_6']), p_symp_age_6_11=float(row['p_symp_age_6_11']),
                 p_symp_age_12plus=float(row['p_symp_age_12plus']))
    else:  # infnum
        p1 = float(row['p_symp_1']); p2 = p1 * float(row['p_r2']); p3 = p2 * float(row['p_r3'])
        p.update(p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)
    return p

# The two fitted immunity structures (calibrated to the same MAL-ED Bangladesh data).
MODELS = {
    'age (symptoms ~ age)':        ('age_binned', HERE / 'experiments' / '25_age_binned_titer_fixedshape'      / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'),
    'infnum (symptoms ~ # infections)': ('infnum',   HERE / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'),
}
SEED_BASE = 90000


def run_model(label, key, post_csv, n_draws, n_agents, response, n_workers):
    """Run no-vaccine vs vaccine for n_draws posterior parameter sets; return per-draw VE + incidence."""
    post = pd.read_csv(post_csv).drop_duplicates().reset_index(drop=True)
    if len(post) > n_draws:
        post = post.sample(n_draws, random_state=0).reset_index(drop=True)
    tasks, meta = [], []
    for i, (_, row) in enumerate(post.iterrows()):
        p = untransform(row, key)
        seed = SEED_BASE + i                                  # SAME seed for the vax/novax pair (variance reduction)
        # _build_run args: (model, params, base_beta, response, vaccinate, seed, n_agents)
        tasks.append((key, p, p['base_beta'], 0.0,      False, seed, n_agents)); meta.append((i, 'novax'))
        tasks.append((key, p, p['base_beta'], response, True,  seed, n_agents)); meta.append((i, 'vax'))
    with get_context('spawn').Pool(processes=min(n_workers, len(tasks)), maxtasksperchild=4) as pool:
        outs = pool.map(vt._build_run, tasks)
    nov, vax = {}, {}
    for (i, kind), o in zip(meta, outs):
        (nov if kind == 'novax' else vax)[i] = o
    rows = []
    for i in nov:
        if i not in vax or nov[i]['overall'] <= 0:
            continue
        ve = 1.0 - vax[i]['overall'] / nov[i]['overall']
        rows.append(dict(draw=i, ve=ve, ir_novax=nov[i]['overall'], ir_vax=vax[i]['overall']))
    df = pd.DataFrame(rows)
    print(f"\n{label}:  VE median {df.ve.median():.2f}  (95% range {df.ve.quantile(.025):.2f}-{df.ve.quantile(.975):.2f})"
          f"   |  symptomatic IR/100cy: no-vax {df.ir_novax.median():.1f} -> vax {df.ir_vax.median():.1f}")
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=20, help='posterior parameter sets per model')
    ap.add_argument('--n-agents', type=int, default=20000)
    ap.add_argument('--response', type=float, default=0.75, help='per-dose seroconversion probability ("efficacy")')
    ap.add_argument('--n-workers', type=int, default=10)
    ap.add_argument('--quick', action='store_true', help='fast/rough: 6 draws, 10k agents')
    a = ap.parse_args()
    if a.quick:
        a.n_draws, a.n_agents = 6, 10000
    print(f"Bangladesh vaccine-impact demo: {a.n_draws} draws/model, {a.n_agents} agents, seroconversion={a.response}")

    results = {lab: run_model(lab, key, csv, a.n_draws, a.n_agents, a.response, a.n_workers)
               for lab, (key, csv) in MODELS.items()}

    # ---- Figure: incidence no-vax vs vax (left) + VE distribution (right), by immunity structure ----
    labels = list(results); colors = ['#2c7fb8', '#c0392b']
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(12, 5))
    x = np.arange(len(labels)); w = 0.36
    nov = [results[l].ir_novax.median() for l in labels]
    vax = [results[l].ir_vax.median() for l in labels]
    axA.bar(x - w/2, nov, w, color='lightgray', label='no vaccine')
    axA.bar(x + w/2, vax, w, color=colors, label='with vaccine')
    ymax = max(nov + vax)
    for i, l in enumerate(labels):
        axA.text(i, max(nov[i], vax[i]) + 0.04*ymax, f"VE {results[l].ve.median()*100:.0f}%",
                 ha='center', va='bottom', fontweight='bold')
    axA.set_xticks(x); axA.set_xticklabels(labels, fontsize=9)
    axA.set_ylim(0, ymax*1.18)
    axA.set_ylabel('symptomatic incidence /100 child-yr (<=36 mo)'); axA.set_title('Predicted vaccine impact')
    axA.legend(frameon=False, loc='center right')
    axB.boxplot([results[l].ve.values for l in labels], labels=[l.split(' ')[0] for l in labels])
    axB.set_ylabel('achieved VE (1 - vax/novax incidence)'); axB.set_title(f'VE across posterior draws (efficacy={a.response})')
    axB.set_ylim(0, 1)
    fig.suptitle('Same pre-vaccine data, two immunity structures -> different predicted vaccine impact (Bangladesh)')
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    out = HERE / 'vaccine_demo_bangladesh.png'
    fig.savefig(out, dpi=140); print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
