"""
Exp 03 — Prior Predictive / Coverage Check (parallel, VM-ready).

Draws parameter sets from the MAL-ED calibration prior, runs the full ABM under
the site's demographics, and records the model's endemic prevalence + IR-by-age
+ first-infection quartiles. Compares the observed MAL-ED targets to the
simulated ensemble (does the data fall inside what the model can produce?).

MEMORY DESIGN: instead of logging every infection event (InfectedStrainStats)
and post-processing into targets -- which holds O(events) in RAM and blows up at
high prevalence -- the `MALEDTargets` analyzer folds the detection + binning
INTO the step loop and keeps only tiny summaries: 4 IR bin counters, person-time
per bin, and one first-detected age per agent. Memory is O(agents), independent
of prevalence and duration. This reproduces process_incidence_maled.process_model
(symptom_model='age_only' logistic, symptomatic IR, first-DETECTED quartiles with
asymptomatic detection at p_asymp_detect).

Writes incrementally to outputs/results.jsonl (resumable / extendable).

Usage:
  uv run python experiments/03_coverage_check/run.py                  # full run
  uv run python experiments/03_coverage_check/run.py --smoke          # 3 draws, 5k agents
  uv run python experiments/03_coverage_check/run.py --n-draws 1000 --n-agents 20000 --n-workers 24
"""
import sys
import os
import json
import argparse
from pathlib import Path
from multiprocessing import get_context

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'
FIGDIR = HERE / 'figures'
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

REPO = HERE.parent.parent
CALIB_DIR = REPO / 'calibration'
AGE_DATA = CALIB_DIR / 'uk_age_data.csv'

CAL_WINDOW = (5.0, 10.0)  # years from start
SITE_DEMOGRAPHICS = {
    'bangladesh': dict(birth_rate=19, death_rate=6),
    'pakistan':   dict(birth_rate=27, death_rate=7),
}
P_ASYMP_DETECT = 0.4
CENSOR_MONTHS = 36.0
# MAL-ED bin labels (the analyzer owns the edges/bins); kept here for output cols.
MALED_LABELS = rs.MALEDTargets.LABELS


def draw_prior(rng):
    """One draw from the 9-parameter MAL-ED calibration prior (matches
    calibrate_maled._trial_to_sim_pars ranges; sus ladder is monotone)."""
    base_beta = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))  # log-uniform
    beta0 = float(rng.uniform(-5.0, 2.0))
    beta1 = float(rng.uniform(-1.0, 1.0))
    beta2 = float(rng.uniform(-0.5, 0.5))
    sus_after_3plus = float(rng.uniform(0.1, 1.0))
    sus_after_2     = float(rng.uniform(sus_after_3plus, 1.0))
    sus_after_1     = float(rng.uniform(sus_after_2, 1.0))
    maternal_immunity_efficacy       = float(rng.uniform(0.5, 0.99))
    maternal_immunity_half_life_days = float(rng.uniform(30.0, 365.0))
    return dict(base_beta=base_beta, beta0=beta0, beta1=beta1, beta2=beta2,
                sus_after_1=sus_after_1, sus_after_2=sus_after_2,
                sus_after_3plus=sus_after_3plus,
                maternal_immunity_efficacy=maternal_immunity_efficacy,
                maternal_immunity_half_life_days=maternal_immunity_half_life_days)


def _run_one(args):
    """Worker: build + run one sim; return tiny target dict. Self-sufficient for spawn."""
    draw_id, params, site, n_agents, seed = args
    demo = SITE_DEMOGRAPHICS[site]
    try:
        targets = rs.MALEDTargets(calibration_window=CAL_WINDOW,
                                  beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'],
                                  reporting_rate=1.0, constant_severity=1.0,
                                  p_asymp_detect=P_ASYMP_DETECT, censor_at_months=CENSOR_MONTHS, seed=seed)
        immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
        people = ss.People(n_agents=n_agents, age_data=str(AGE_DATA))
        sim = rs.Sim(
            n_agents=n_agents, start='2003-01-01', stop='2013-01-01', dt=ss.days(1),
            verbose=False, scenario='single', people=people,
            analyzers=[targets],
            networks=ss.RandomNet(n_contacts=7),
            demographics=[ss.Births(birth_rate=ss.peryear(demo['birth_rate'])),
                          ss.Deaths(death_rate=ss.peryear(demo['death_rate']))],
            connectors=[immunity_connector], rand_seed=seed,
        )
        sim.pars.base_beta = params['base_beta']
        for disease in sim.pars.diseases:
            if isinstance(disease, rs.Rotavirus):
                disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)
        sim.init()
        ic = sim.connectors.rotaimmunityconnector
        ic.pars['use_fixed_susceptibility'] = True
        ic.pars['sus_after_1']     = params['sus_after_1']
        ic.pars['sus_after_2']     = params['sus_after_2']
        ic.pars['sus_after_3plus'] = params['sus_after_3plus']
        ic.pars['maternal_immunity_efficacy']  = params['maternal_immunity_efficacy']
        ic.pars['maternal_immunity_half_life'] = ss.days(params['maternal_immunity_half_life_days'])
        ic.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
        sim.run()

        out = sim.analyzers['maledtargets'].results_dict()
        rec = dict(draw_id=draw_id, site=site, seed=seed, ok=True,
                   prev_mean=round(out['prev_mean'], 5), prev_drift=round(out['prev_drift'], 5),
                   fi_q25=round(out['fi_q25'], 4), fi_median=round(out['fi_median'], 4),
                   fi_q75=round(out['fi_q75'], 4), fi_n=out['fi_n'],
                   **{f'par_{k}': round(v, 5) for k, v in params.items()})
        for b in MALED_LABELS:
            rec[f'ir_{b}'] = round(out['ir'][b], 5)
        return rec
    except Exception as e:
        import traceback
        return dict(draw_id=draw_id, site=site, seed=seed, ok=False,
                    error=f'{e!r} | {traceback.format_exc()[-400:]}',
                    **{f'par_{k}': round(v, 5) for k, v in params.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=1000)
    ap.add_argument('--n-agents', type=int, default=20_000)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--site', default='bangladesh')
    ap.add_argument('--seed', type=int, default=20260605)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=str(OUTDIR / 'results.jsonl'))
    args = ap.parse_args()

    if args.smoke:
        args.n_draws, args.n_agents = 4, 5_000
        args.out = str(OUTDIR / 'results_smoke.jsonl')

    rng = np.random.default_rng(args.seed)
    draws = [draw_prior(rng) for _ in range(args.n_draws)]
    tasks = [(i, p, args.site, args.n_agents, args.seed + i) for i, p in enumerate(draws)]

    n_workers = args.n_workers or os.cpu_count()
    print(f'Coverage check: {args.n_draws} draws, site={args.site}, '
          f'{args.n_agents} agents, {n_workers} workers', flush=True)

    outpath = Path(args.out)
    if outpath.exists():
        outpath.unlink()

    t0 = sc.tic()
    ctx = get_context('spawn')
    done = n_fail = 0
    with ctx.Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(_run_one, tasks):
            with outpath.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            done += 1
            n_fail += (not rec.get('ok'))
            if done % 50 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)} ({n_fail} failed), {sc.toc(t0, output=True):.0f}s', flush=True)

    print(f'\nDone: {done} sims, {n_fail} failed, {sc.toc(t0, output=True):.0f}s total')
    print(f'Results: {outpath}')


if __name__ == '__main__':
    main()
