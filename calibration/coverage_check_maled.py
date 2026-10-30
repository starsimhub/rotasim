"""
Prior-predictive coverage check (calibration-workflow step 3) for the MAL-ED
Bangladesh PRE-VACCINE fit.

Decisive question: can the model JOINTLY reach both targets -- symptomatic IR
by age AND age-at-first-(detected)-infection -- under the prior? Draws
INDEPENDENT prior samples (NOT Optuna/TPE), 1 replicate each (the binary
"can it reach the data" question), and reports:
  - MARGINAL coverage: is each data target inside the envelope of draws?
  - JOINT coverage: is there a SINGLE draw close to BOTH targets at once?

Marginal-pass + joint-fail is the signature of a structural Pareto tension:
no single parameter set can produce both, so no symptom-model/maternal tweak
or better optimizer will fix it -> structural change needed (e.g. age-on-
infection). Joint-pass would instead point at a search/likelihood problem.

Default symptom model is age_and_infection_offsets (most flexible: age logistic
+ categorical per-infection offsets, nests age_only). If even that fails joint
coverage, the limit is in the infection dynamics, not the symptom model.

Run on the VM (reuses the calibrate_maled spawn-pool worker):
  python coverage_check_maled.py --n-draws 50
  python coverage_check_maled.py --n-draws 200 --symptom-model age_only
"""
import argparse
from multiprocessing import get_context

import numpy as np
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
BINS = process_incidence_maled.MALED_AGE_BINS
QUARTS = ('q25', 'median', 'q75')

# "Acceptable fit" thresholds, ~ our best calibrated fits, used only to label
# the joint-coverage box (not to drive the marginal-coverage verdict).
INC_OK = 5.0     # gof_incidence (sum sq log-IR over bins) of a good fit
FI_OK  = 0.30    # gof_first_infection of a good fit


def sample_prior(rng, symptom_model):
    """One INDEPENDENT prior draw, matching calibrate_maled's ranges (NOT TPE)."""
    p = dict(
        base_beta=float(np.exp(rng.uniform(np.log(0.05), np.log(0.5)))),  # log-uniform
        sus_after_3plus=(s3 := rng.uniform(0.1, 1.0)),
        sus_after_2=(s2 := rng.uniform(s3, 1.0)),
        sus_after_1=rng.uniform(s2, 1.0),
        maternal_immunity_efficacy=rng.uniform(0.5, 0.99),
        maternal_immunity_mean_duration_days=rng.uniform(30.0, 300.0),
    )
    if symptom_model == 'infection_number':
        p1 = rng.uniform(0.0, 1.0); p2 = rng.uniform(0.0, p1); p3 = rng.uniform(0.0, p2)
        p.update(p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)
    else:
        p.update(beta0=rng.uniform(-5, 2), beta1=rng.uniform(-1, 1), beta2=rng.uniform(-0.5, 0.5))
        if symptom_model == 'age_and_infection':
            p['beta3'] = rng.uniform(-2.0, 0.5)
        elif symptom_model == 'age_and_infection_offsets':
            g2 = rng.uniform(-6.0, 0.0); g3 = rng.uniform(-10.0, g2)
            p.update(gamma_2=g2, gamma_3plus=g3)
    return p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=50)
    ap.add_argument('--symptom-model', default='age_and_infection_offsets',
                    choices=['age_only', 'infection_number',
                             'age_and_infection', 'age_and_infection_offsets'])
    ap.add_argument('--maternal-n-stages', type=int, default=6)
    ap.add_argument('--n-agents', type=int, default=100_000)
    ap.add_argument('--n-workers', type=int, default=40)
    ap.add_argument('--seed', type=int, default=20260604)
    args = ap.parse_args()

    targets = process_incidence_maled.load_targets(SITE)
    demo = SITE_DEMOGRAPHICS[SITE]
    sim_config = dict(
        n_agents=args.n_agents, start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY, reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'), p_asymp_detect=0.4,
        symptom_model=args.symptom_model, maternal_n_stages=args.maternal_n_stages,
    )

    rng = np.random.default_rng(args.seed)
    draws = [sample_prior(rng, args.symptom_model) for _ in range(args.n_draws)]
    seeds = rng.integers(0, 1_000_000, args.n_draws).tolist()  # 1 rep / draw
    arglist = [(sim_config, draws[i], int(seeds[i]), CAL_WINDOW) for i in range(args.n_draws)]

    print(f"Coverage check: {args.n_draws} prior draws x 1 rep | symptom_model={args.symptom_model} "
          f"| Erlang n={args.maternal_n_stages} | n_agents={args.n_agents}")
    with get_context('spawn').Pool(processes=min(args.n_workers, args.n_draws)) as pool:
        outs = pool.map(_run_one_replicate, arglist)

    ir = np.array([[mo['ir_by_age'].loc[b, 'IR'] for b in BINS] for mo in outs])      # n x 4
    fi = np.array([[mo['first_infection'][k] for k in QUARTS] for mo in outs])         # n x 3
    tgt_ir = np.array([targets['ir_by_age'].loc[b, 'IR'] for b in BINS])
    tgt_fi = np.array([targets['first_infection'][k] for k in QUARTS])
    gof_inc = np.array([process_incidence_maled.gof(mo, targets, fit_target='symptomatic_ir')['gof_incidence'] for mo in outs])
    gof_fi  = np.array([process_incidence_maled.gof(mo, targets, fit_target='first_infection')['gof_first_infection'] for mo in outs])

    print("\n=== MARGINAL coverage (is each data target inside [min,max] of draws?) ===")
    marg = True
    for j, b in enumerate(BINS):
        lo, hi = ir[:, j].min(), ir[:, j].max(); inside = lo <= tgt_ir[j] <= hi; marg &= inside
        print(f"  IR {b:<8}: data={tgt_ir[j]:6.2f}  draws[{lo:6.2f},{hi:6.2f}]  {'IN ' if inside else 'OUT'}")
    for j, k in enumerate(QUARTS):
        lo, hi = fi[:, j].min(), fi[:, j].max(); inside = lo <= tgt_fi[j] <= hi; marg &= inside
        print(f"  first-inf {k:<6}: data={tgt_fi[j]:6.2f}  draws[{lo:6.2f},{hi:6.2f}]  {'IN ' if inside else 'OUT'}")

    joint = np.where((gof_inc <= INC_OK) & (gof_fi <= FI_OK))[0]
    print("\n=== JOINT coverage (a SINGLE draw close to BOTH targets) ===")
    print(f"  draws with gof_inc<={INC_OK} AND gof_first<={FI_OK}: {len(joint)} / {args.n_draws}")
    print(f"  best single-target across draws: min gof_inc={gof_inc.min():.2f}, min gof_first={gof_fi.min():.3f}")
    # Pareto-closest draw to the origin (both good), normalized
    norm = (gof_inc / INC_OK) ** 2 + (gof_fi / FI_OK) ** 2
    bi = int(np.argmin(norm))
    print(f"  closest-to-both draw #{bi}: gof_inc={gof_inc[bi]:.2f}, gof_first={gof_fi[bi]:.3f}, "
          f"IR={[round(float(v),2) for v in ir[bi]]}")
    verdict = ("JOINTLY COVERED -- a single draw reaches both targets "
               "(tension is search/likelihood, not structural)") if len(joint) > 0 else \
              ("NOT JOINTLY COVERED -- no single prior draw reaches both targets "
               "(Pareto tension is STRUCTURAL; symptom-model/optimizer tweaks won't fix it)")
    print(f"\n  VERDICT: {verdict}")
    if marg and len(joint) == 0:
        print("  NOTE: marginal coverage passes but joint fails -> classic structural-tension signature.")

    # ---- figure ----
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
    x = np.arange(len(BINS))
    for j in range(len(BINS)):
        axA.scatter(np.full(args.n_draws, x[j]) + rng.uniform(-0.12, 0.12, args.n_draws),
                    ir[:, j], s=10, color='#bbbbbb', alpha=0.55)
    axA.scatter(x, tgt_ir, color='red', marker='_', s=600, lw=3, zorder=5, label='MAL-ED data')
    axA.set_yscale('symlog', linthresh=0.1)
    axA.set_xticks(x); axA.set_xticklabels(BINS)
    axA.set_ylabel('Symptomatic IR (per 100 PM)')
    axA.set_title(f'(A) Marginal coverage by age bin\n{args.n_draws} prior draws (grey) vs data (red)')
    axA.legend(frameon=False)
    axA.spines[['top', 'right']].set_visible(False)

    axB.scatter(gof_inc, np.maximum(gof_fi, 1e-3), s=22, color='#2ca02c', alpha=0.6, label='prior draws')
    axB.axvline(INC_OK, color='k', ls=':', alpha=0.5); axB.axhline(FI_OK, color='k', ls=':', alpha=0.5)
    axB.scatter([gof_inc[bi]], [max(gof_fi[bi], 1e-3)], s=160, marker='*',
                color='#2ca02c', edgecolors='k', zorder=6, label='closest to both')
    axB.set_xscale('log'); axB.set_yscale('log')
    axB.set_xlabel(r'GOF$_{\mathrm{incidence}}$ (lower=better)')
    axB.set_ylabel(r'GOF$_{\mathrm{first\ infection}}$ (lower=better)')
    axB.set_title(f'(B) Joint coverage\nlower-left box = both acceptable; {len(joint)} draw(s) inside')
    axB.legend(frameon=False, loc='upper right')
    axB.grid(True, which='both', ls=':', alpha=0.3)
    fig.suptitle(f'Prior-predictive coverage check — MAL-ED Bangladesh — {args.symptom_model}', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out = f'coverage_check_{args.symptom_model}.png'
    fig.savefig(out, dpi=150)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
