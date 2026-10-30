"""
Exp 15 — toy vaccine simulations on the exp-10 (age+Erlang) and exp-11 (infnum+titer)
fitted models. Vaccine mechanism (per A. Kraay's VIMC model SI): 2 doses at 2 & 4 months;
each dose a person SEROCONVERTS to (prob = response_prob) advances them ONE
infection-equivalent -> num_recovered_infections += 1 (up to +2 for a both-dose
responder). That single counter feeds BOTH susceptibility (sus_after_k, via the connector)
AND the infection-number symptom probability (p_symp, via the observer reading the
counter). So in the infection-number model the vaccine reduces symptom-given-infection;
in the age model symptoms are age-driven, so the counter only reduces acquisition --
the structural difference we want to compare.

VE (toy) = 1 - symptomatic-IR(fully-vaccinated world) / symptomatic-IR(unvaccinated world),
over the calibration window, age <=36 mo (total effect; same population both runs).

Run (on the VM):  python vaccine_toy.py --n-agents 50000 --reps 3
"""
import sys, json, argparse, pathlib
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc, starsim as ss, rotasim as rs, optuna

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
import process_incidence_maled as P  # noqa: E402

BINS = P.MALED_AGE_BINS
EDGES_M = np.array([0.0, 6.0, 12.0, 24.0, 36.0])
AGE_DATA = str(CALIB / 'bangladesh_age_data.csv')
DEMO = dict(birth_rate=19, death_rate=6)
WINDOW = (5.0, 10.0)
DOSE_AGES_Y = [2.0/12.0, 4.0/12.0]   # 2 and 4 months

MODELS = {
    'age':    dict(db='rota_maled_bangladesh_erlang6_poisson_ptfix.db',
                   study='rota_maled_bangladesh_erlang6_poisson', symptom_model='age_only'),
    'infnum': dict(db='rota_maled_bangladesh_infnum_titer_poisson.db',
                   study='rota_maled_bangladesh_infnum_titer_poisson', symptom_model='infection_number'),
    'age_binned': dict(symptom_model='age_binned'),   # corrected-pair age member (exp25)
}


class VaccinePrime(ss.Intervention):
    """2-dose vaccine; each seroconversion advances num_recovered_infections by 1 (up to +2).
    Each agent crosses each dose age once (per-step age slice), so no dose-state needed."""
    def __init__(self, response_prob, dose_ages_y=DOSE_AGES_Y, coverage=1.0, moa='infection_blocking', **kw):
        super().__init__(**kw)
        self.response_prob = response_prob
        self.dose_ages = list(dose_ages_y)
        self.coverage = coverage
        self.moa = moa                  # 'infection_blocking' (advance infection-equiv) | 'symptom_blocking'
        self.vaccinated_uids = set()    # symptom_blocking: seroconverted agents whose infections are attenuated

    def init_pre(self, sim):
        super().init_pre(sim)
        self._rng = np.random.default_rng(int(sim.pars.rand_seed or 0) + 7001)

    def step(self):
        sim = self.sim
        ic = sim.connectors.rotaimmunityconnector
        dt_y = self.dt.years
        alive_uids = sim.people.alive.uids
        if len(alive_uids) == 0:
            return
        au = np.asarray(alive_uids)
        ages = np.asarray(sim.people.age[ss.uids(au)])
        for d_age in self.dose_ages:                       # 2mo, 4mo -> up to +2
            cross = au[(ages >= d_age) & (ages < d_age + dt_y)]   # crossed this step (each agent once)
            if len(cross) == 0:
                continue
            recv = cross if self.coverage >= 1 else cross[self._rng.random(len(cross)) < self.coverage]
            if len(recv) == 0:
                continue
            sero = recv[self._rng.random(len(recv)) < self.response_prob]
            if len(sero):
                if self.moa == 'symptom_blocking':       # attenuate disease, do NOT block acquisition
                    self.vaccinated_uids.update(int(u) for u in sero)
                else:                                    # infection_blocking: advance infection-equivalents
                    ic.num_recovered_infections[ss.uids(sero)] += 1.0


class SympIRObserver(ss.Analyzer):
    """Symptomatic IR by age over the window. Symptom number for infnum reads the connector
    counter (num_recovered_infections + 1), so vaccine bumps flow through automatically."""
    def __init__(self, symptom_model, beta0=0, beta1=0, beta2=0,
                 p_symp_1=1.0, p_symp_2=1.0, p_symp_3plus=1.0,
                 p_symp_age_0_6=0.5, p_symp_age_6_11=0.5, p_symp_age_12plus=0.5,
                 symp_block_s=0.0, window=WINDOW, seed=0, **kw):
        super().__init__(**kw)
        self.sm = symptom_model
        self.b0, self.b1, self.b2 = beta0, beta1, beta2
        self.ps = [p_symp_1, p_symp_2, p_symp_3plus]
        self.pa = [p_symp_age_0_6, p_symp_age_6_11, p_symp_age_12plus]   # by age bin (<6, 6-11, >=12 mo)
        self.symp_block_s = symp_block_s    # symptom_blocking MOA: P(symp) *= (1-s) for vaccinated agents
        self.window = window
        self.rng = np.random.default_rng(seed)
        self.cases = {b: 0 for b in BINS}
        self.person_years = {b: 0.0 for b in BINS}
        self.first_age = {}                 # uid -> age (months) at first infection (age-of-infection mediator)

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force); self._dty = self.dt.years

    def init_results(self):
        super().init_results()
        self._dis = [d for d in self.sim.diseases.values() if hasattr(d, 'G')]
        self._prev = {d.name: d.infected.uids for d in self._dis}
        self._ic = self.sim.connectors.rotaimmunityconnector
        self._vax = next((iv for iv in self.sim.interventions.values() if isinstance(iv, VaccinePrime)), None)

    def _symp_prob(self, age_m, order):
        if self.sm == 'age_only':
            ac = np.minimum(age_m, 60.0) - 12.0
            return 1.0 / (1.0 + np.exp(-np.clip(self.b0 + self.b1*ac + self.b2*ac*ac, -30, 30)))
        if self.sm == 'age_binned':   # free P(symp) per age bin; ignores infection order (age-driven)
            return np.where(age_m < 6.0, self.pa[0], np.where(age_m < 12.0, self.pa[1], self.pa[2]))
        return np.array([self.ps[min(int(o), 3) - 1] for o in order])

    def step(self):
        sim = self.sim
        yr = sim.t.relvec[sim.ti].years
        inw = self.window[0] <= yr < self.window[1]
        ages_y = sim.people.age.values; alive = sim.people.alive.values
        if inw:
            for b, lo, hi in zip(BINS, EDGES_M[:-1]/12, EDGES_M[1:]/12):
                self.person_years[b] += int(((ages_y >= lo) & (ages_y < hi) & alive).sum()) * self._dty
        for d in self._dis:
            cur = d.infected.uids
            new = np.asarray(cur - self._prev[d.name])              # newly infected this step (all ages, all-time)
            if len(new):
                am = np.asarray(sim.people.age[ss.uids(new)]) * 12.0
                born = am <= (yr * 12.0 + 1e-6)                     # age <= elapsed sim time => born after start (cohort)
                for u, a in zip(new[born], am[born]):               # record age at FIRST infection (birth cohort)
                    iu = int(u)
                    if iu not in self.first_age:
                        self.first_age[iu] = float(a)
                if inw:
                    m = am <= 36.0
                    if m.any():
                        am2 = am[m]; uu = new[m]
                        order = self._ic.num_recovered_infections[ss.uids(uu)] + 1.0
                        psymp = self._symp_prob(am2, order)
                        if self.symp_block_s > 0 and self._vax is not None and self._vax.vaccinated_uids:
                            vmask = np.fromiter((int(u) in self._vax.vaccinated_uids for u in uu), bool, len(uu))
                            psymp = psymp * np.where(vmask, 1.0 - self.symp_block_s, 1.0)   # attenuate disease in vaccinated
                        symp = self.rng.random(len(am2)) < psymp
                        if symp.any():
                            idx = np.digitize(am2[symp], EDGES_M) - 1
                            for k in range(4):
                                self.cases[BINS[k]] += int((idx == k).sum())
            self._prev[d.name] = cur

    def ir(self):
        pm = {b: self.person_years[b] * 12.0 for b in BINS}
        by = {b: (self.cases[b] / pm[b] * 100.0 if pm[b] > 0 else 0.0) for b in BINS}
        tot_c = sum(self.cases.values()); tot_pm = sum(pm.values())
        fa = np.fromiter(self.first_age.values(), float)
        return dict(by_age=by, overall=(tot_c / tot_pm * 100.0 if tot_pm > 0 else 0.0),
                    first_inf_median=(float(np.median(fa)) if fa.size else float('nan')))


def _build_run(args):
    model, params, base_beta, response_prob, vaccinate, seed, n_agents = args[:7]
    moa = args[7] if len(args) > 7 else 'infection_blocking'   # 'infection_blocking' | 'symptom_blocking'
    s = args[8] if len(args) > 8 else 1.0                      # symptom-blocking attenuation (1.0 = full)
    m = MODELS[model]
    interventions = [VaccinePrime(response_prob=response_prob, moa=moa)] if vaccinate else []
    sb = s if (vaccinate and moa == 'symptom_blocking') else 0.0
    obs = SympIRObserver(symptom_model=m['symptom_model'], symp_block_s=sb,
                         beta0=params.get('beta0', 0), beta1=params.get('beta1', 0), beta2=params.get('beta2', 0),
                         p_symp_1=params.get('p_symp_1', 1.0), p_symp_2=params.get('p_symp_2', 1.0),
                         p_symp_3plus=params.get('p_symp_3plus', 0.0),
                         p_symp_age_0_6=params.get('p_symp_age_0_6', 0.5), p_symp_age_6_11=params.get('p_symp_age_6_11', 0.5),
                         p_symp_age_12plus=params.get('p_symp_age_12plus', 0.5), seed=seed)
    ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    people = ss.People(n_agents=n_agents, age_data=AGE_DATA)
    sim = rs.Sim(n_agents=n_agents, start='2003-01-01', stop='2013-01-01', verbose=False, scenario='single',
                 people=people, analyzers=[obs], networks=ss.RandomNet(n_contacts=7),
                 demographics=[ss.Births(birth_rate=ss.peryear(DEMO['birth_rate'])),
                               ss.Deaths(death_rate=ss.peryear(DEMO['death_rate']))],
                 interventions=interventions, connectors=[ic], rand_seed=seed)
    sim.pars.base_beta = base_beta
    for dis in sim.pars.diseases:
        if isinstance(dis, rs.Rotavirus):
            dis.pars.beta = ss.perday(base_beta * dis.pars.fitness)
    sim.init()
    ic = sim.connectors.rotaimmunityconnector
    ic.pars['use_fixed_susceptibility'] = True
    ic.pars['sus_after_1'] = params['sus_after_1']; ic.pars['sus_after_2'] = params['sus_after_2']
    ic.pars['sus_after_3plus'] = params['sus_after_3plus']
    ic.pars['maternal_immunity_efficacy'] = params['maternal_immunity_efficacy']
    if 'maternal_titer_median' in params:
        ic.pars['maternal_immunity_model'] = 'titer'
        ic.pars['maternal_titer_median'] = params['maternal_titer_median']
        ic.pars['maternal_titer_gsd'] = params['maternal_titer_gsd']
        ic.pars['maternal_titer_half_life'] = ss.days(params['maternal_titer_half_life_days'])
        ic.pars['maternal_hill_slope'] = params['maternal_hill_slope']
    else:
        ic.pars['maternal_immunity_model'] = 'erlang'
        ic.pars['maternal_immunity_n_stages'] = 6
        ic.pars['maternal_immunity_mean_duration'] = ss.days(params['maternal_immunity_mean_duration_days'])
    ic.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
    sim.run()
    res = sim.analyzers[0].ir()
    try: sim.shrink(die=False)
    except Exception: pass
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-agents', type=int, default=50000)
    ap.add_argument('--reps', type=int, default=3)
    ap.add_argument('--responses', type=float, nargs='+', default=[0.6, 0.75, 0.9])
    ap.add_argument('--beta-mults', type=float, nargs='+', default=[1.0, 0.5])  # fitted (LMIC) + lower (HIC proxy)
    ap.add_argument('--n-workers', type=int, default=16)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    if a.smoke:
        a.n_agents = 8000; a.reps = 1; a.responses = [0.75]; a.beta_mults = [1.0]

    pars = {}
    for model, m in MODELS.items():
        s = optuna.load_study(study_name=m['study'], storage=f"sqlite:///{CALIB / m['db']}")
        pars[model] = s.best_trial.params

    # Build task list: novax once per (model, beta); vax per (model, beta, response). reps each.
    tasks, meta = [], []
    for model in MODELS:
        bb0 = pars[model]['base_beta']
        for bm in a.beta_mults:
            bb = bb0 * bm
            for r in range(a.reps):
                tasks.append((model, pars[model], bb, 0.0, False, 1000 + r, a.n_agents)); meta.append((model, bm, bb, 'novax', None, r))
                for resp in a.responses:
                    tasks.append((model, pars[model], bb, resp, True, 2000 + r, a.n_agents)); meta.append((model, bm, bb, 'vax', resp, r))

    print(f"{len(tasks)} runs ({a.n_agents} agents): {len(MODELS)} models x {len(a.beta_mults)} betas x "
          f"({len(a.responses)} resp +novax) x {a.reps} reps", flush=True)
    with get_context('spawn').Pool(processes=min(a.n_workers, len(tasks)), maxtasksperchild=4) as pool:
        outs = pool.map(_build_run, tasks)

    rows = []
    for (model, bm, bb, kind, resp, rep), o in zip(meta, outs):
        rows.append(dict(model=model, beta_mult=bm, base_beta=bb, kind=kind, response=resp, rep=rep,
                         overall_ir=o['overall'], **{f'ir_{b}': o['by_age'][b] for b in BINS}))
    df = pd.DataFrame(rows)
    (HERE / 'outputs').mkdir(parents=True, exist_ok=True)
    df.to_csv(HERE / 'outputs' / 'toy_runs.csv', index=False)

    # VE = 1 - mean symp IR(vax) / mean symp IR(novax), per (model, beta, response)
    print("\n=== TOY VE (overall symptomatic IR, <=36mo) ===")
    print(f"{'model':8}{'beta_mult':>10}{'base_beta':>10}{'response':>10}{'novax_IR':>10}{'vax_IR':>9}{'VE':>8}")
    summ = []
    for model in MODELS:
        for bm in a.beta_mults:
            nov = df[(df.model == model) & (df.beta_mult == bm) & (df.kind == 'novax')].overall_ir.mean()
            for resp in a.responses:
                vax = df[(df.model == model) & (df.beta_mult == bm) & (df.kind == 'vax') & (df.response == resp)].overall_ir.mean()
                ve = 1 - vax / nov if nov > 0 else float('nan')
                summ.append(dict(model=model, beta_mult=bm, base_beta=df[(df.model==model)&(df.beta_mult==bm)].base_beta.iloc[0],
                                 response=resp, novax_ir=nov, vax_ir=vax, VE=ve))
                print(f"{model:8}{bm:>10}{summ[-1]['base_beta']:>10.3f}{resp:>10}{nov:>10.3f}{vax:>9.3f}{ve:>8.3f}")
    json.dump(summ, (HERE / 'outputs' / 've_summary.json').open('w'), indent=2)
    print(f"\nwrote outputs/toy_runs.csv + ve_summary.json")


if __name__ == '__main__':
    main()
