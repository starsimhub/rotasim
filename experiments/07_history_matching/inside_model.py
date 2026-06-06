"""
Exp 07 — look INSIDE the model on posterior-representative trajectories.

The stored SIR has only summary observables per sim. To see the mechanism (prevalence
over time, immunity build-up, susceptibility-by-age, maternal protection-by-age) we
RE-RUN high-posterior-weight trajectories with a read-only observer analyzer.

To recover the SAME latent trajectory we re-use the exact (params, seed) pair the SIR
used (seed = 20260605 + idx), the same cohort censoring array, and n_agents/cross. The
observer (InsideModel) is a normal ss.Analyzer that only READS state and does
deterministic math (recomputes maternal protection from the already-drawn titers) -- it
defines no ss.Dist and uses no np.random, so it should not perturb any RNG stream.

We do NOT assume that; we TEST it. `--verify` runs one trajectory three ways:
  stored (capy)  vs  ref (local, cohort only, sim.run)  vs  rec (local, + observer)
  - ref vs rec   -> does the observer perturb the trajectory?
  - ref vs stored-> does the platform/environment change the trajectory? (identical config)

Usage:
  uv run python experiments/07_history_matching/inside_model.py --verify
  uv run python experiments/07_history_matching/inside_model.py --k 3
"""
import json, argparse
from pathlib import Path
from multiprocessing import get_context
import numpy as np, pandas as pd, sciris as sc, starsim as ss, rotasim as rs
from scipy.stats import nbinom, betabinom

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
rw = sc.importbypath(HERE / 'run_wave.py')        # _untransform: NROY row -> exact params
exp06 = sc.importbypath(REPO / 'experiments' / '06_titer_maternal_peak' / 'run.py')
AGE_DATA = REPO / 'calibration' / 'uk_age_data.csv'
N_AGENTS, CROSS = 40_000, 0.5
NB = 37                                          # single-month age bins 0..36
ENDEMIC_FROM_REL = 5.0                           # years-since-start; average age-profiles over the endemic window
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]
REPEAT_OBS, REPEAT_N, EVER_OBS, EVER_N = 58, 136, 136, 213


# ----------------------------- observer analyzer (read-only, no RNG) -----------------------------
class InsideModel(ss.Analyzer):
    """Records prevalence/immunity time series + age-profiles. Reads state only; draws nothing."""
    def init_post(self):
        super().init_post()
        self.setattribute('rt', []); self.setattribute('series', {k: [] for k in
                          ('prev', 'prev_inf', 'prev_yng', 'prev_rest', 'msus')})
        self.setattribute('age', {k: np.zeros(NB) for k in
                          ('sus', 'mat', 'tot', 'prev', 'nrec', 'cnt')})

    def _rota(self):
        return [d for d in self.sim.diseases.values() if isinstance(d, rs.Rotavirus)][0]

    def step(self):
        sim = self.sim; ppl = sim.people
        alive = ppl.alive.values; age = ppl.age.values
        dis = self._rota(); ic = sim.connectors.rotaimmunityconnector
        inf = dis.infected.values; rel_sus = dis.rel_sus.values
        nrec = ic.num_recovered_infections.values
        mat = np.zeros(len(age))                                 # maternal protection (deterministic, no RNG)
        if ic.pars.maternal_immunity_model == 'titer' and float(ic.pars.maternal_immunity_efficacy) > 0:
            t0 = ic.maternal_titer0.values
            titer = t0 * np.exp(-np.log(2) * age / ic.pars.maternal_titer_half_life.years)
            th = np.power(np.maximum(titer, 0.0), float(ic.pars.maternal_hill_slope))
            mat = float(ic.pars.maternal_immunity_efficacy) * (th / (th + 1.0))
            mat[np.isnan(t0)] = 0.0
        a, i1, yo, re = alive, alive & (age < 1), alive & (age >= 1) & (age < 5), alive & (age >= 5)
        yr = sim.t.relvec[sim.ti].years
        self.rt.append(float(yr))
        for key, m in (('prev', a), ('prev_inf', i1), ('prev_yng', yo), ('prev_rest', re)):
            self.series[key].append(inf[m].mean() if m.any() else np.nan)
        self.series['msus'].append(rel_sus[a].mean() if a.any() else np.nan)
        if yr >= ENDEMIC_FROM_REL:
            mb = np.clip((age * 12).astype(int), 0, NB - 1); sel = alive & (age < NB / 12.0)
            np.add.at(self.age['cnt'], mb[sel], 1.0)
            np.add.at(self.age['sus'], mb[sel], rel_sus[sel])
            np.add.at(self.age['mat'], mb[sel], mat[sel])
            np.add.at(self.age['tot'], mb[sel], 1 - rel_sus[sel])
            np.add.at(self.age['prev'], mb[sel], inf[sel].astype(float))
            np.add.at(self.age['nrec'], mb[sel], nrec[sel])

    def results_dict(self):
        c = np.maximum(self.age['cnt'], 1e-9)
        out = dict(t=self.rt, age_months=list(range(NB)), cnt_by_age=self.age['cnt'].tolist())
        out.update(self.series)
        for k in ('sus', 'mat', 'tot', 'prev', 'nrec'):
            out[f'{k}_by_age'] = (self.age[k] / c).tolist()
        return out


def _build_sim(params, seed, cens, observer):
    """Identical to exp06._run_one; optionally append the read-only observer analyzer."""
    cohort = exp06.MALEDCohort(p_symp_1=params['p_symp_1'], p_symp_2=params['p_symp_2'],
                               p_symp_3plus=params['p_symp_3plus'], censoring_ages=cens, seed=seed)
    analyzers = [cohort] + ([InsideModel()] if observer else [])
    ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    people = ss.People(n_agents=N_AGENTS, age_data=str(AGE_DATA))
    ag = {'inf': ss.AgeGroup(0, 1), 'young': ss.AgeGroup(1, 5), 'rest': ss.AgeGroup(5, None)}
    matrix = exp06.make_matrix(params['infant_exposure'], params['young_reservoir'], CROSS, params['adult_contacts'])
    net = ss.MixingPools(diseases='G1P8', beta=1.0, src=ag, dst=ag, n_contacts=matrix)
    sim = rs.Sim(n_agents=N_AGENTS, start='2003-01-01', stop='2013-01-01', dt=ss.days(1), verbose=False,
                 scenario='single', people=people, analyzers=analyzers, networks=net,
                 demographics=[ss.Births(birth_rate=ss.peryear(exp06.DEMO['birth_rate'])),
                               ss.Deaths(death_rate=ss.peryear(exp06.DEMO['death_rate']))],
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
    icc.pars['maternal_immunity_model'] = 'titer'
    icc.pars['maternal_immunity_efficacy'] = params['maternal_efficacy']
    icc.pars['maternal_titer_median'] = params['titer_median']
    icc.pars['maternal_titer_gsd'] = params['titer_gsd']
    icc.pars['maternal_titer_half_life'] = ss.days(params['titer_half_life_days'])
    icc.pars['maternal_hill_slope'] = params['hill_slope']
    icc.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
    return sim


def _obs_from(sim):
    co = sim.analyzers['maledcohort'].results_dict()
    o = {f'ir_symp_{b}': round(co['ir_symp'][b], 5) for b, _, _ in IR}
    o.update(frac_ever_detected=round(co['frac_ever_detected'], 4),
             repeat_detected_frac=round(co['repeat_detected_frac'], 4))
    return o


def _run_inside(args):
    idx, params, seed, cens = args
    sim = _build_sim(params, seed, cens, observer=True)
    sim.run()
    out = dict(idx=idx, seed=seed, obs=_obs_from(sim), inside=sim.analyzers['insidemodel'].results_dict(), params=params)
    sim.shrink(die=False)
    return out


def _logL(r, phi=2.0, rho=0.05):
    if (not r.get('ok')) or (r.get('frac_ever_detected') or 0) < 0.05: return -np.inf
    ll = 0.0
    for b, c, PT in IR:
        mu = r[f'ir_symp_{b}'] / 100.0 * PT
        if mu <= 0: return -np.inf
        rr = mu / (phi - 1.0); ll += nbinom.logpmf(c, rr, rr / (rr + mu))
    for k, n, p in [(REPEAT_OBS, REPEAT_N, r['repeat_detected_frac']), (EVER_OBS, EVER_N, r['frac_ever_detected'])]:
        p = min(max(p, 1e-6), 1 - 1e-6); M = (1 - rho) / rho
        ll += betabinom.logpmf(k, n, p * M, (1 - p) * M)
    return ll


def _load():
    recs = [json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')]
    L = np.array([_logL(r) for r in recs]); fin = np.isfinite(L)
    w = np.zeros(len(recs)); w[fin] = np.exp(L[fin] - L[fin].max()); w /= w.sum()
    fi = pd.read_csv(REPO / 'calibration' / 'maled_data' / 'first_infection_bangladesh.csv')
    cens = fi.loc[fi.event_observed == 0, 'age_event_months'].dropna().values; cens = cens[cens > 0]
    # EXACT params per idx from the cached NROY draw (the stored par_* are rounded to 6dp,
    # insufficient to reconstruct knife-edge trajectories -- see repro_check.py).
    nroy = pd.read_csv(HERE / 'outputs' / 'nroy_draw.csv')
    return recs, w, cens, nroy


def _exact_params(nroy, idx):
    return rw._untransform(nroy.iloc[idx])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--k', type=int, default=3, help='number of top-weight trajectories to re-run')
    ap.add_argument('--verify', action='store_true', help='run top idx 3 ways: stored / ref(cohort-only) / rec(+observer)')
    args = ap.parse_args()
    recs, w, cens, nroy = _load()
    print(f'starsim {ss.__version__}, rotasim {getattr(rs, "__version__", "?")}, numpy {np.__version__}')

    if args.verify:
        i = int(np.argmax(w)); r = recs[i]; idx = r['idx']
        params = _exact_params(nroy, idx); seed = 20260605 + idx
        cols = [f'ir_symp_{b}' for b, _, _ in IR] + ['frac_ever_detected', 'repeat_detected_frac']
        print(f'\nVERIFY idx {idx}, seed {seed} (top weight {w[i]:.3f}):')
        ref = _obs_from(_build_sim(params, seed, cens, observer=False).run())       # local, cohort only
        rec = _run_inside((idx, params, seed, cens))['obs']                          # local, + observer
        print(f'  {"":>22}' + ''.join(f'{c.replace("ir_symp_",""):>12}' for c in cols))
        print(f'  {"stored (capy)":>22}' + ''.join(f'{r[c]:>12.4f}' for c in cols))
        print(f'  {"ref (local,cohort)":>22}' + ''.join(f'{ref[c]:>12.4f}' for c in cols))
        print(f'  {"rec (local,+observer)":>22}' + ''.join(f'{rec[c]:>12.4f}' for c in cols))
        ref_eq_rec = all(abs(ref[c] - rec[c]) < 1e-6 for c in cols)
        ref_eq_stored = all(abs(ref[c] - r[c]) < 1e-4 for c in cols)
        print(f'\n  ref == rec (observer harmless)?    {ref_eq_rec}')
        print(f'  ref == stored (platform-stable)?   {ref_eq_stored}')
        print('  => ' + ('observer is faithful; ' if ref_eq_rec else 'OBSERVER PERTURBS RNG; ')
              + ('local reproduces capy.' if ref_eq_stored else 'platform/env changes the trajectory -> run on capy for exact match.'))
        return

    topidx = np.argsort(w)[::-1][:args.k]
    print(f'Re-running top {args.k} by weight: idx={[recs[i]["idx"] for i in topidx]}')
    tasks = [(recs[i]['idx'], _exact_params(nroy, recs[i]['idx']), 20260605 + recs[i]['idx'], cens)
             for i in topidx]
    t0 = sc.tic()
    with get_context('spawn').Pool(processes=min(args.k, 4), maxtasksperchild=1) as pool:
        outs = list(pool.imap_unordered(_run_inside, tasks))
    print(f'  ran {len(outs)} in {sc.toc(t0, output=True):.0f}s')
    by_idx = {o['idx']: o for o in outs}
    print('\nReproduction vs stored:')
    for i in topidx:
        r = recs[i]; o = by_idx[r['idx']]
        match = all(abs(o['obs'][f'ir_symp_{b}'] - r[f'ir_symp_{b}']) < 1e-4 for b, _, _ in IR)
        print(f"  idx {r['idx']}: IR re-run={[o['obs'][f'ir_symp_{b}'] for b,_,_ in IR]} "
              f"stored={[r[f'ir_symp_{b}'] for b,_,_ in IR]} -> {'MATCH' if match else 'differ'}")
    sc.savejson(HERE / 'outputs' / 'inside_model.json', outs)
    print('\nsaved -> outputs/inside_model.json')


if __name__ == '__main__':
    main()
