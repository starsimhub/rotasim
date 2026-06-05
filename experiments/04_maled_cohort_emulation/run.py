"""
Exp 04 — MAL-ED Cohort Emulation + Kaplan-Meier censoring.

Faithfully emulates the MAL-ED study structure inside the endemic ABM:
  - enroll a BIRTH COHORT (agents born during an enrollment window), follow each
    from birth to 24 months;
  - apply the real surveillance detection regime:
      * symptomatic infection -> diarrheal stool -> detected with prob
        (capture 0.79 x eia_sensitivity) at infection onset;
      * asymptomatic infection -> detected only via scheduled surveillance stool;
        prob = min(1, shed_days/interval) x eia_sensitivity, where the sampling
        interval is MONTHLY (~30.4d) for age <12mo and QUARTERLY (~91.3d) after
        (the 15/18/21/24-mo schedule) -> detection drops ~3x after 12 months;
  - record age-at-first-DETECTION per child with right-censoring at 24 months
    (censored = never detected during follow-up);
  - cohort-based person-time and IR-by-age (not a steady-state snapshot).

Outputs per draw: KM-able (first_detect_age, event_observed) summary stats,
IR-by-age (all-detected and symptomatic), repeat-infection fraction, prevalence.
Same prior + seed as exp 03, so draw indices align (draw #314 is the good fit).

PINS (tuning knobs, revisit): eia_sensitivity, shed_days.

Usage:
  uv run python experiments/04_maled_cohort_emulation/run.py --smoke
  uv run python experiments/04_maled_cohort_emulation/run.py --n-draws 480 --n-workers 118
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
ENROLL_WINDOW = (5.0, 7.5)                  # sim-years; 24mo follow-up completes by 9.5 (< stop=10)
FOLLOWUP_M = 24.0
CAPTURE = 0.79                              # fraction of diarrheal episodes with analyzed stool
EIA_SENSITIVITY = 0.85                      # PIN: EIA antigen sensitivity (tuning knob, revisit)
SHED_DAYS = 13.0                            # PIN: detectable shedding window (~model mean dur_inf)
MONTHLY_INTERVAL_D = 30.4375
QUARTERLY_INTERVAL_D = 91.3125

LABELS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']
EDGES_M = np.array([0.0, 6.0, 12.0, 24.0, 36.0])
BINS_M = {'<6 m': (0, 6), '6-11 m': (6, 12), '12-23 m': (12, 24), '24-35 m': (24, 36)}


class MALEDCohort(ss.Analyzer):
    """Emulate a MAL-ED birth cohort with surveillance detection + censoring."""

    def __init__(self, beta0, beta1, beta2, censoring_ages, enroll_window=ENROLL_WINDOW,
                 followup_months=FOLLOWUP_M, capture=CAPTURE,
                 eia_sensitivity=EIA_SENSITIVITY, shed_days=SHED_DAYS, seed=0, **kw):
        super().__init__(**kw)
        self.beta0, self.beta1, self.beta2 = beta0, beta1, beta2
        self.enroll = enroll_window
        self.fu = followup_months
        self.capture = capture
        self.eia = eia_sensitivity
        self.shed = shed_days
        # Empirical per-child exit-age distribution (months), from the data's
        # observed censoring ages -> individual-level dropout (non-informative).
        self.censoring_ages = np.asarray(censoring_ages, dtype=float)
        self.rng = np.random.default_rng(seed)
        # per-enrolled-child state (parallel arrays; uid->index map)
        self.uid2idx = {}
        self.exit_age_m = []     # individual study-exit (dropout/administrative) age
        self.last_age_m = []     # max observed age (months) while alive & in follow-up
        self.true_first_m = []   # age at first infection (any), regardless of detection
        self.det_first_m = []    # age at first DETECTED infection
        self.n_det = []          # number of detected infections (for repeat fraction)
        # cohort accumulators
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

    def _symp_prob(self, age_m):
        ac = np.minimum(age_m, 60.0) - 12.0
        lp = self.beta0 + self.beta1 * ac + self.beta2 * ac * ac
        return 1.0 / (1.0 + np.exp(-lp))

    def _p_surv(self, age_m):
        """Per-asymptomatic-infection probability a surveillance stool catches it."""
        interval = MONTHLY_INTERVAL_D if age_m < 12.0 else QUARTERLY_INTERVAL_D
        return min(1.0, self.shed / interval)

    def _enroll(self, uid):
        self.uid2idx[uid] = len(self.last_age_m)
        # Draw this child's study-exit age from the data's censoring distribution.
        self.exit_age_m.append(float(self.rng.choice(self.censoring_ages)))
        self.last_age_m.append(0.0)
        self.true_first_m.append(np.nan)
        self.det_first_m.append(np.nan)
        self.n_det.append(0)

    def step(self):
        sim = self.sim
        t = sim.t.relvec[sim.ti].years
        alive_uids = sim.people.alive.uids
        ages_m = np.asarray(sim.people.age[alive_uids]) * 12.0

        # 1. Enroll newborns during the enrollment window.
        if self.enroll[0] <= t < self.enroll[1]:
            newborn = ages_m <= self._dtm  # born within ~one step
            for u in np.asarray(alive_uids)[newborn]:
                if int(u) not in self.uid2idx:
                    self._enroll(int(u))

        if not self.uid2idx:
            for d in self._diseases:
                self._prev_infected[d.name] = d.infected.uids
            return

        # 2. Cohort person-time + last-age update for enrolled, alive, in follow-up.
        eu = np.fromiter(self.uid2idx.keys(), dtype=np.int64)
        eu_idx = np.fromiter(self.uid2idx.values(), dtype=np.int64)
        eu_age_m = np.asarray(sim.people.age[ss.uids(eu)]) * 12.0
        eu_alive = np.asarray(sim.people.alive[ss.uids(eu)])
        eu_exit = np.array(self.exit_age_m)[eu_idx]   # individual study-exit age
        in_fu = eu_alive & (eu_age_m <= eu_exit)
        # person-time by bin
        for b, (lo, hi) in BINS_M.items():
            self.person_years[b] += int((in_fu & (eu_age_m >= lo) & (eu_age_m < hi)).sum()) * self._dty
        # update last observed age
        upd = in_fu
        la = np.array(self.last_age_m)
        la[eu_idx[upd]] = np.maximum(la[eu_idx[upd]], eu_age_m[upd])
        self.last_age_m = la.tolist()

        # 3. New infections among enrolled children in follow-up -> detection.
        for d in self._diseases:
            cur = d.infected.uids
            new = cur - self._prev_infected[d.name]
            for u in np.asarray(new):
                ui = int(u)
                idx = self.uid2idx.get(ui)
                if idx is None:
                    continue
                a = float(sim.people.age[ss.uids(np.array([ui]))][0]) * 12.0
                if a > self.exit_age_m[idx]:   # past this child's study exit -> unobserved
                    continue
                if np.isnan(self.true_first_m[idx]):
                    self.true_first_m[idx] = a
                symp = self.rng.random() < self._symp_prob(a)
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
        det = np.array(self.det_first_m)
        last = np.array(self.last_age_m)
        exitage = np.array(self.exit_age_m)
        true_first = np.array(self.true_first_m)
        ndet = np.array(self.n_det)
        # KM survival data: (time, event_observed). Censored at the child's study
        # exit (or earlier death, captured by last_age) when undetected.
        observed = ~np.isnan(det)
        time = np.where(observed, det, np.minimum(last, exitage))
        pm = {b: self.person_years[b] * 12.0 for b in LABELS}
        ir_all = {b: (self.cases_all[b] / pm[b] * 100.0 if pm[b] > 0 else 0.0) for b in LABELS}
        ir_symp = {b: (self.cases_symp[b] / pm[b] * 100.0 if pm[b] > 0 else 0.0) for b in LABELS}
        ever_inf = ~np.isnan(true_first)
        return dict(
            n_enrolled=int(n),
            km_time=time.tolist(), km_observed=observed.astype(int).tolist(),
            ir_all=ir_all, ir_symp=ir_symp, person_months=pm,
            cases_all=dict(self.cases_all), cases_symp=dict(self.cases_symp),
            frac_ever_detected=float(observed.mean()) if n else float('nan'),
            frac_ever_infected=float(ever_inf.mean()) if n else float('nan'),
            repeat_detected_frac=float((ndet[observed] >= 2).mean()) if observed.any() else 0.0,
            true_first_median=float(np.nanmedian(true_first)) if ever_inf.any() else float('nan'),
        )


def draw_prior(rng):
    base_beta = float(np.exp(rng.uniform(np.log(0.05), np.log(0.5))))
    beta0 = float(rng.uniform(-5.0, 2.0)); beta1 = float(rng.uniform(-1.0, 1.0))
    beta2 = float(rng.uniform(-0.5, 0.5))
    s3 = float(rng.uniform(0.1, 1.0)); s2 = float(rng.uniform(s3, 1.0)); s1 = float(rng.uniform(s2, 1.0))
    me = float(rng.uniform(0.5, 0.99)); mh = float(rng.uniform(30.0, 365.0))
    return dict(base_beta=base_beta, beta0=beta0, beta1=beta1, beta2=beta2,
                sus_after_1=s1, sus_after_2=s2, sus_after_3plus=s3,
                maternal_immunity_efficacy=me, maternal_immunity_half_life_days=mh)


def _run_one(args):
    draw_id, params, n_agents, seed, censoring_ages = args
    try:
        cohort = MALEDCohort(beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'],
                             censoring_ages=censoring_ages, seed=seed)
        ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
        people = ss.People(n_agents=n_agents, age_data=str(AGE_DATA))
        sim = rs.Sim(n_agents=n_agents, start='2003-01-01', stop='2013-01-01', dt=ss.days(1),
                     verbose=False, scenario='single', people=people, analyzers=[cohort],
                     networks=ss.RandomNet(n_contacts=7),
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
        icc.pars['maternal_immunity_half_life'] = ss.days(params['maternal_immunity_half_life_days'])
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
            rec[f'ir_all_{b}'] = round(out['ir_all'][b], 5)
            rec[f'ir_symp_{b}'] = round(out['ir_symp'][b], 5)
        return rec
    except Exception as e:
        import traceback
        return dict(draw_id=draw_id, seed=seed, ok=False, error=f'{e!r} | {traceback.format_exc()[-300:]}',
                    **{f'par_{k}': round(v, 5) for k, v in params.items()})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-draws', type=int, default=480)
    ap.add_argument('--n-agents', type=int, default=20_000)
    ap.add_argument('--n-workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=20260605)  # same as exp 03 -> draw indices align
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--out', default=str(OUTDIR / 'results.jsonl'))
    args = ap.parse_args()
    if args.smoke:
        args.n_draws, args.n_agents = 4, 8_000
        args.out = str(OUTDIR / 'results_smoke.jsonl')

    # Data-driven dropout: per-child exit ages drawn from the data's censoring ages.
    import pandas as pd
    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    censoring_ages = fi.loc[fi['event_observed'] == 0, 'age_event_months'].dropna().values
    censoring_ages = censoring_ages[censoring_ages > 0]
    print(f'Dropout model: {len(censoring_ages)} empirical exit ages '
          f'(median {np.median(censoring_ages):.1f}mo, {(censoring_ages<20).mean()*100:.0f}% <20mo)', flush=True)

    rng = np.random.default_rng(args.seed)
    draws = [draw_prior(rng) for _ in range(args.n_draws)]
    tasks = [(i, p, args.n_agents, args.seed + i, censoring_ages) for i, p in enumerate(draws)]
    n_workers = args.n_workers or os.cpu_count()
    print(f'MAL-ED cohort emulation: {args.n_draws} draws, {args.n_agents} agents, {n_workers} workers', flush=True)

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
