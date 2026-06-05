"""
Exp 05 — Generalizing mechanism: infection-number severity + Erlang maternal +
age-structured low-infant-exposure mixing, fit to MAL-ED Bangladesh.

Combines:
  - ss.MixingPools, 3 groups (inf 0-1 / young 1-5 / rest 5+): a high within-toddler
    reservoir, LOW young->infant exposure (the key lever controlling infant FOI),
    background cross_contacts elsewhere. (Alicia's exp 04 structure.)
  - infection-number symptom severity: p_symp_1 >= p_symp_2 >= p_symp_3+ (first
    infection most symptomatic) -- an INTRINSIC per-infection conditional chosen to
    generalize across the FOI gradient (the age of the peak then emerges from
    timing, not a site-tuned age curve).
  - strong Erlang maternal immunity (sharp plateau-then-drop) sets the floor age by
    delaying the first infection past ~6mo.
  - cohort observation: birth cohort to 24mo, schedule-based surveillance detection,
    individual data-driven dropout (from MAL-ED Bangladesh censoring ages).

Targets (joint): symptomatic IR-by-age (6-11mo peak), age-at-first-detection (KM),
repeat-infection fraction (~10%).

Usage:
  uv run python experiments/05_structured_mixing_cohort/run.py --smoke
  uv run python experiments/05_structured_mixing_cohort/run.py --n-draws 1000 --n-agents 40000 --n-workers 118
"""
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
OUTDIR = HERE / 'outputs'; OUTDIR.mkdir(exist_ok=True)
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
REPO = HERE.parent.parent
AGE_DATA = REPO / 'calibration' / 'uk_age_data.csv'

DEMO = dict(birth_rate=19, death_rate=6)   # Bangladesh
ENROLL_WINDOW = (5.0, 7.5)
CAPTURE = 0.79
EIA_SENSITIVITY = 0.85                      # PIN
SHED_DAYS = 13.0                            # PIN
MATERNAL_N_STAGES = 6                       # sharp Erlang (fixed)
MONTHLY_INTERVAL_D = 30.4375
QUARTERLY_INTERVAL_D = 91.3125
LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
EDGES_M = np.array([0.0, 6.0, 12.0, 24.0, 36.0])
BINS_M = {'<6 m': (0, 6), '6-11 m': (6, 12), '12-23 m': (12, 24), '24-35 m': (24, 36)}


class MALEDCohort(ss.Analyzer):
    """Cohort emulation with INFECTION-NUMBER symptom severity (vs exp 04's age curve)."""

    def __init__(self, p_symp_1, p_symp_2, p_symp_3plus, censoring_ages,
                 enroll_window=ENROLL_WINDOW, capture=CAPTURE,
                 eia_sensitivity=EIA_SENSITIVITY, shed_days=SHED_DAYS, seed=0, **kw):
        super().__init__(**kw)
        self.p_symp = [p_symp_1, p_symp_2, p_symp_3plus]  # by infection order (1,2,3+)
        self.enroll = enroll_window
        self.capture = capture
        self.eia = eia_sensitivity
        self.shed = shed_days
        self.censoring_ages = np.asarray(censoring_ages, float)
        self.rng = np.random.default_rng(seed)
        self.uid2idx = {}
        self.exit_age_m = []
        self.last_age_m = []
        self.n_inf = []          # TRUE cumulative infection count (sets symptom order)
        self.true_first_m = []
        self.det_first_m = []
        self.n_det = []
        self.person_years = {b: 0.0 for b in LABELS}
        self.cases_all = {b: 0 for b in LABELS}
        self.cases_symp = {b: 0 for b in LABELS}

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        self._dty = self.dt.years
        self._dtm = self.dt.years * 12.0

    def init_results(self):
        super().init_results()
        self._diseases = [d for d in self.sim.diseases.values() if hasattr(d, 'G')]
        self._prev_infected = {d.name: d.infected.uids for d in self._diseases}

    def _symp_prob(self, order):
        return self.p_symp[min(order, 3) - 1]

    def _p_surv(self, age_m):
        interval = MONTHLY_INTERVAL_D if age_m < 12.0 else QUARTERLY_INTERVAL_D
        return min(1.0, self.shed / interval)

    def _enroll(self, uid):
        self.uid2idx[uid] = len(self.last_age_m)
        self.exit_age_m.append(float(self.rng.choice(self.censoring_ages)))
        self.last_age_m.append(0.0); self.n_inf.append(0)
        self.true_first_m.append(np.nan); self.det_first_m.append(np.nan); self.n_det.append(0)

    def step(self):
        sim = self.sim
        t = sim.t.relvec[sim.ti].years
        alive_uids = sim.people.alive.uids
        ages_m = np.asarray(sim.people.age[alive_uids]) * 12.0

        if self.enroll[0] <= t < self.enroll[1]:
            for u in np.asarray(alive_uids)[ages_m <= self._dtm]:
                if int(u) not in self.uid2idx:
                    self._enroll(int(u))
        if not self.uid2idx:
            for d in self._diseases:
                self._prev_infected[d.name] = d.infected.uids
            return

        eu = np.fromiter(self.uid2idx.keys(), dtype=np.int64)
        eu_idx = np.fromiter(self.uid2idx.values(), dtype=np.int64)
        eu_age_m = np.asarray(sim.people.age[ss.uids(eu)]) * 12.0
        eu_alive = np.asarray(sim.people.alive[ss.uids(eu)])
        eu_exit = np.array(self.exit_age_m)[eu_idx]
        in_fu = eu_alive & (eu_age_m <= eu_exit)
        for b, (lo_m, hi_m) in BINS_M.items():
            self.person_years[b] += int((in_fu & (eu_age_m >= lo_m) & (eu_age_m < hi_m)).sum()) * self._dty
        la = np.array(self.last_age_m)
        la[eu_idx[in_fu]] = np.maximum(la[eu_idx[in_fu]], eu_age_m[in_fu])
        self.last_age_m = la.tolist()

        for d in self._diseases:
            cur = d.infected.uids
            new = cur - self._prev_infected[d.name]
            for u in np.asarray(new):
                idx = self.uid2idx.get(int(u))
                if idx is None:
                    continue
                a = float(sim.people.age[ss.uids(np.array([int(u)]))][0]) * 12.0
                if a > self.exit_age_m[idx]:
                    continue
                self.n_inf[idx] += 1
                order = self.n_inf[idx]
                if np.isnan(self.true_first_m[idx]):
                    self.true_first_m[idx] = a
                symp = self.rng.random() < self._symp_prob(order)
                if symp:
                    detected = self.rng.random() < (self.capture * self.eia)
                else:
                    detected = self.rng.random() < (self._p_surv(a) * self.eia)
                if detected:
                    self.n_det[idx] += 1
                    k = int(np.digitize(a, EDGES_M) - 1)
                    if 0 <= k < 4:
                        self.cases_all[LABELS[k]] += 1
                        if symp:
                            self.cases_symp[LABELS[k]] += 1
                    if np.isnan(self.det_first_m[idx]):
                        self.det_first_m[idx] = a
            self._prev_infected[d.name] = cur

    def results_dict(self):
        n = len(self.last_age_m)
        det = np.array(self.det_first_m); last = np.array(self.last_age_m)
        exitage = np.array(self.exit_age_m); true_first = np.array(self.true_first_m)
        ndet = np.array(self.n_det)
        observed = ~np.isnan(det)
        time = np.where(observed, det, np.minimum(last, exitage))
        pm = {b: self.person_years[b] * 12.0 for b in LABELS}
        ir_symp = {b: (self.cases_symp[b] / pm[b] * 100.0 if pm[b] > 0 else 0.0) for b in LABELS}
        ir_all = {b: (self.cases_all[b] / pm[b] * 100.0 if pm[b] > 0 else 0.0) for b in LABELS}
        ever = ~np.isnan(true_first)
        return dict(n_enrolled=int(n), km_time=time.tolist(), km_observed=observed.astype(int).tolist(),
                    ir_symp=ir_symp, ir_all=ir_all,
                    frac_ever_detected=float(observed.mean()) if n else float('nan'),
                    frac_ever_infected=float(ever.mean()) if n else float('nan'),
                    repeat_detected_frac=float((ndet[observed] >= 2).mean()) if observed.any() else 0.0,
                    true_first_median=float(np.nanmedian(true_first)) if ever.any() else float('nan'))


def make_matrix(infant_exposure, young_reservoir, cross):
    """3x3 contact matrix, groups [inf, young, rest], rows=src cols=dst.
    young->inf = infant_exposure (LOW, key), young->young = young_reservoir (HIGH)."""
    c = cross
    return np.array([[c, c, c],
                     [infant_exposure, young_reservoir, c],
                     [c, c, c]], dtype=float)


def draw_prior(rng):
    base_beta = float(np.exp(rng.uniform(np.log(0.05), np.log(0.6))))   # pool scale (~Alicia 0.25)
    young_reservoir = float(rng.uniform(10.0, 40.0))
    infant_exposure = float(rng.uniform(0.5, 12.0))                     # key low lever
    s3 = float(rng.uniform(0.1, 1.0)); s2 = float(rng.uniform(s3, 1.0)); s1 = float(rng.uniform(s2, 1.0))
    mat_eff = float(rng.uniform(0.5, 0.99))
    mat_dur = float(rng.uniform(120.0, 300.0))                          # days; sets drop/peak age
    # infection-number symptom severity, monotone decreasing
    p1 = float(rng.uniform(0.4, 1.0)); p2 = float(rng.uniform(0.05, p1)); p3 = float(rng.uniform(0.0, p2))
    return dict(base_beta=base_beta, young_reservoir=young_reservoir, infant_exposure=infant_exposure,
                sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
                maternal_immunity_efficacy=mat_eff, maternal_mean_duration_days=mat_dur,
                p_symp_1=p1, p_symp_2=p2, p_symp_3plus=p3)


def _run_one(args):
    draw_id, params, n_agents, seed, censoring_ages, cross = args
    try:
        cohort = MALEDCohort(p_symp_1=params['p_symp_1'], p_symp_2=params['p_symp_2'],
                             p_symp_3plus=params['p_symp_3plus'], censoring_ages=censoring_ages, seed=seed)
        ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
        people = ss.People(n_agents=n_agents, age_data=str(AGE_DATA))
        ag = {'inf': ss.AgeGroup(0, 1), 'young': ss.AgeGroup(1, 5), 'rest': ss.AgeGroup(5, None)}
        matrix = make_matrix(params['infant_exposure'], params['young_reservoir'], cross)
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
        icc.pars['maternal_immunity_efficacy'] = params['maternal_immunity_efficacy']
        icc.pars['maternal_immunity_n_stages'] = MATERNAL_N_STAGES
        icc.pars['maternal_immunity_mean_duration'] = ss.days(params['maternal_mean_duration_days'])
        icc.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
        sim.run()
        out = sim.analyzers['maledcohort'].results_dict()
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
    censoring_ages = fi.loc[fi['event_observed'] == 0, 'age_event_months'].dropna().values
    censoring_ages = censoring_ages[censoring_ages > 0]

    rng = np.random.default_rng(args.seed)
    draws = [draw_prior(rng) for _ in range(args.n_draws)]
    tasks = [(i, p, args.n_agents, args.seed + i, censoring_ages, args.cross_contacts) for i, p in enumerate(draws)]
    n_workers = args.n_workers or os.cpu_count()
    print(f'Exp 05: {args.n_draws} draws, {args.n_agents} agents, {n_workers} workers, '
          f'cross_contacts={args.cross_contacts}', flush=True)

    outpath = Path(args.out)
    if outpath.exists():
        outpath.unlink()
    t0 = sc.tic()
    ctx = get_context('spawn')
    done = nfail = 0
    with ctx.Pool(processes=n_workers) as pool:
        for rec in pool.imap_unordered(_run_one, tasks):
            with outpath.open('a') as f:
                f.write(json.dumps(rec) + '\n')
            done += 1; nfail += (not rec.get('ok'))
            if done % 40 == 0 or done == len(tasks):
                print(f'  {done}/{len(tasks)} ({nfail} failed), {sc.toc(t0, output=True):.0f}s', flush=True)
    print(f'\nDone: {done} sims, {nfail} failed, {sc.toc(t0, output=True):.0f}s')
    print(f'Results: {outpath}')


if __name__ == '__main__':
    main()
