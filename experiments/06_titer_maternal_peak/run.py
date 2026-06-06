"""
Exp 06 — Titer-based maternal immunity + low young_reservoir, sharpening the IR peak.

Same cohort emulation (MixingPools 3-group, infection-number severity, surveillance
detection + individual dropout) as exp 05, but:
  - young_reservoir swept LOW (1-15) to stay off the knife-edge;
  - maternal immunity uses the IBM titer model (per-infant log-normal titer, common
    decay, Hill protection) -- peakiness from titer gsd + Hill slope, drop-age from
    median + half-life;
  - acquired-immunity ladder (sus_after_*) produces the 12-23m decline.

Score primarily on IR-by-age (peak bin AND magnitude/sharpness).

Usage:
  uv run python experiments/06_titer_maternal_peak/run.py --smoke
  uv run python experiments/06_titer_maternal_peak/run.py --n-draws 1000 --n-agents 40000 --n-workers 118
"""
import os, json, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, sciris as sc, starsim as ss, rotasim as rs

HERE = Path(__file__).resolve().parent
OUTDIR = HERE / 'outputs'; OUTDIR.mkdir(exist_ok=True)
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
REPO = HERE.parent.parent
AGE_DATA = REPO / 'calibration' / 'uk_age_data.csv'
# Reuse the validated cohort analyzer + matrix builder from exp 05.
exp05 = sc.importbypath(HERE.parent / '05_structured_mixing_cohort' / 'run.py')
MALEDCohort = exp05.MALEDCohort
DEMO = dict(birth_rate=19, death_rate=6)
LABELS = exp05.LABELS


def make_matrix(infant_exposure, young_reservoir, cross, adult_contacts):
    """3x3 contact matrix, groups [inf, young, rest], rows=src cols=dst.
    young->young = young_reservoir; young->inf = infant_exposure (low);
    rest->rest = adult_contacts (a large, permanent reservoir for persistence,
    decoupled from the child targets); everything else = cross background."""
    c = cross
    return np.array([[c, c, c],
                     [infant_exposure, young_reservoir, c],
                     [c, c, adult_contacts]], dtype=float)


def draw_prior(rng):
    base_beta = float(np.exp(rng.uniform(np.log(0.10), np.log(0.6))))   # higher: low reservoir needs more beta
    young_reservoir = float(rng.uniform(1.0, 15.0))                     # THE focus: swept low
    infant_exposure = float(rng.uniform(0.3, 4.0))
    adult_contacts = float(rng.uniform(0.5, 2.5))                       # adult reservoir for persistence (decoupled from targets)
    s3 = float(rng.uniform(0.1, 1.0)); s2 = float(rng.uniform(s3, 1.0)); s1 = float(rng.uniform(s2, 1.0))
    # titer-based maternal: median+half_life -> drop age; gsd+hill_slope -> sharpness
    mat_eff = float(rng.uniform(0.7, 0.99))
    titer_median = float(np.exp(rng.uniform(np.log(4.0), np.log(60.0))))   # IC50 units
    titer_gsd = float(rng.uniform(1.3, 3.5))
    titer_half_life_days = float(rng.uniform(25.0, 70.0))
    hill_slope = float(rng.uniform(1.5, 8.0))
    p1 = float(rng.uniform(0.4, 1.0)); p2 = float(rng.uniform(0.05, p1)); p3 = float(rng.uniform(0.0, p2))
    return dict(base_beta=base_beta, young_reservoir=young_reservoir, infant_exposure=infant_exposure,
                adult_contacts=adult_contacts,
                sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
                maternal_efficacy=mat_eff, titer_median=titer_median, titer_gsd=titer_gsd,
                titer_half_life_days=titer_half_life_days, hill_slope=hill_slope,
                p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)


def _run_one(args):
    draw_id, params, n_agents, seed, censoring_ages, cross = args
    try:
        cohort = MALEDCohort(p_symp_1=params['p_symp_1'], p_symp_2=params['p_symp_2'],
                             p_symp_3plus=params['p_symp_3plus'], censoring_ages=censoring_ages, seed=seed)
        ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
        people = ss.People(n_agents=n_agents, age_data=str(AGE_DATA))
        ag = {'inf': ss.AgeGroup(0, 1), 'young': ss.AgeGroup(1, 5), 'rest': ss.AgeGroup(5, None)}
        matrix = make_matrix(params['infant_exposure'], params['young_reservoir'], cross, params['adult_contacts'])
        net = ss.MixingPools(diseases='G1P8', beta=1.0, src=ag, dst=ag, n_contacts=matrix)
        sim = rs.Sim(n_agents=n_agents, start='2003-01-01', stop='2013-01-01', dt=ss.days(1),
                     verbose=False, scenario='single', people=people, analyzers=[cohort], networks=net,
                     demographics=[ss.Births(birth_rate=ss.peryear(DEMO['birth_rate'])),
                                   ss.Deaths(death_rate=ss.peryear(DEMO['death_rate']))],
                     connectors=[ic], rand_seed=seed)
        sim.pars.base_beta = params['base_beta']
        for d in sim.pars.diseases:
            if isinstance(d, rs.Rotavirus):
                d.pars.beta = ss.perday(sim.pars.base_beta * d.pars.fitness)
        sim.init()
        icc = sim.connectors.rotaimmunityconnector
        icc.pars['use_fixed_susceptibility'] = True
        for k in ('sus_after_1', 'sus_after_2', 'sus_after_3plus'):
            icc.pars[k] = params[k]
        # titer-based maternal model
        icc.pars['maternal_immunity_model'] = 'titer'
        icc.pars['maternal_immunity_efficacy'] = params['maternal_efficacy']
        icc.pars['maternal_titer_median'] = params['titer_median']
        icc.pars['maternal_titer_gsd'] = params['titer_gsd']
        icc.pars['maternal_titer_half_life'] = ss.days(params['titer_half_life_days'])
        icc.pars['maternal_hill_slope'] = params['hill_slope']
        icc.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
        sim.run()
        out = sim.analyzers['maledcohort'].results_dict()
        sim.shrink(die=False)  # release leaked sim refs (starsim#1343); die=False: MixingPools
        #                        won't shrink below size_limit -> warn instead of raise. Results extracted.
        rec = dict(draw_id=draw_id, seed=seed, ok=True, n_enrolled=out['n_enrolled'],
                   frac_ever_detected=round(out['frac_ever_detected'], 4),
                   frac_ever_infected=round(out['frac_ever_infected'], 4),
                   repeat_detected_frac=round(out['repeat_detected_frac'], 4),
                   true_first_median=round(out['true_first_median'], 3),
                   km_time=[round(x, 3) for x in out['km_time']], km_observed=out['km_observed'],
                   **{f'par_{k}': round(v, 5) for k, v in params.items()})
        for b in LABELS:
            rec[f'ir_symp_{b}'] = round(out['ir_symp'][b], 5)
            rec[f'ir_all_{b}'] = round(out['ir_all'][b], 5)
        return rec
    except Exception as e:
        import traceback
        return dict(draw_id=draw_id, seed=seed, ok=False, error=f'{e!r} | {traceback.format_exc()[-300:]}',
                    **{f'par_{k}': round(v, 5) for k, v in params.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=1000)
    ap.add_argument('--n-agents', type=int, default=40_000)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--cross-contacts', type=float, default=0.5)
    ap.add_argument('--seed', type=int, default=20260605)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=str(OUTDIR / 'results.jsonl'))
    args = ap.parse_args()
    if args.smoke:
        args.n_draws, args.n_agents = 6, 10_000
        args.out = str(OUTDIR / 'results_smoke.jsonl')

    import pandas as pd
    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    cens = fi.loc[fi['event_observed'] == 0, 'age_event_months'].dropna().values
    cens = cens[cens > 0]
    rng = np.random.default_rng(args.seed)
    draws = [draw_prior(rng) for _ in range(args.n_draws)]
    tasks = [(i, p, args.n_agents, args.seed + i, cens, args.cross_contacts) for i, p in enumerate(draws)]
    nw = args.n_workers or os.cpu_count()
    print(f'Exp 06: {args.n_draws} draws, {args.n_agents} agents, {nw} workers (titer maternal, young_reservoir 1-15)', flush=True)
    outpath = Path(args.out)
    if outpath.exists(): outpath.unlink()
    t0 = sc.tic(); ctx = get_context('spawn'); done = nfail = 0
    with ctx.Pool(processes=nw) as pool:
        for rec in pool.imap_unordered(_run_one, tasks):
            with outpath.open('a') as f: f.write(json.dumps(rec) + '\n')
            done += 1; nfail += (not rec.get('ok'))
            if done % 40 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)} ({nfail} failed), {sc.toc(t0, output=True):.0f}s', flush=True)
    print(f'\nDone: {done} sims, {nfail} failed, {sc.toc(t0, output=True):.0f}s\nResults: {outpath}')


if __name__ == '__main__':
    main()
