"""
Exp 04 — age-structured contacts feasibility (see README.md).

Replaces homogeneous mixing (RandomNet) with ss.MixingPools over 2 age groups
(under-5 reservoir vs 5+) and asks the feasibility question: can ANY combination
of (base_beta, within-under-5 contact intensity) produce a 6-11 mo SYMPTOMATIC
peak — the thing homogeneous mixing demonstrably could not (exp 03)? Symptoms are
infection-number (p_symp_1>=p_symp_2>=p_symp_3plus), fixed for this sweep.

MixingPools wiring (de-risked): pool beta=1.0 (the DISEASE beta carries
transmission; they multiply), diseases='G1P8', set disease beta before sim.init(),
grab the immunity connector after init. Runs locally (no VM needed).

  python run.py            # uses config.yaml
"""
import sys
import json
import itertools
import pathlib
import argparse
from multiprocessing import get_context

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR))
import process_incidence_maled as P   # noqa: E402

BINS = P.MALED_AGE_BINS
WINDOW = (5.0, 10.0)


def _run_mp(task):
    """Build the rotasim sim with MixingPools, run, return symptomatic + all-infection IR by age."""
    import starsim as ss
    import rotasim as rs
    cfg, pars, matrix, seed = task
    an = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=1.0)
    ic0 = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    ppl = ss.People(n_agents=cfg['n_agents'], age_data=cfg['age_data_path'])
    ag = {'inf': ss.AgeGroup(0, 1), 'young': ss.AgeGroup(1, 5), 'rest': ss.AgeGroup(5, None)}
    net = ss.MixingPools(diseases='G1P8', beta=1.0, src=ag, dst=ag, n_contacts=matrix)
    sim = rs.Sim(n_agents=cfg['n_agents'], start='2003-01-01', stop='2013-01-01', verbose=False,
                 scenario='single', people=ppl, analyzers=[an], networks=net,
                 demographics=[ss.Births(birth_rate=ss.peryear(cfg['birth_rate'])),
                               ss.Deaths(death_rate=ss.peryear(cfg['death_rate']))],
                 interventions=[], connectors=[ic0], rand_seed=seed)
    sim.pars.base_beta = pars['base_beta']
    for d in sim.pars.diseases:                       # disease beta BEFORE init
        if isinstance(d, rs.Rotavirus):
            d.pars.beta = ss.perday(pars['base_beta'] * d.pars.fitness)
    sim.init()
    ic = sim.connectors.rotaimmunityconnector
    ic.pars['use_fixed_susceptibility'] = True
    ic.pars['sus_after_1'] = pars['sus_after_1']
    ic.pars['sus_after_2'] = pars['sus_after_2']
    ic.pars['sus_after_3plus'] = pars['sus_after_3plus']
    ic.pars['maternal_immunity_efficacy'] = pars['maternal_immunity_efficacy']
    ic.pars['maternal_immunity_n_stages'] = pars['maternal_n_stages']            # sharp Erlang waning
    ic.pars['maternal_immunity_mean_duration'] = ss.days(pars['maternal_mean_duration_days'])
    ic.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
    sim.run()
    df = sim.analyzers['infectedstrainstats'].to_df()
    pt = P.compute_person_months_steady_state(sim.people.age.values, (WINDOW[1] - WINDOW[0]) * 12.0)
    kw = dict(person_months_by_bin=pt, symptom_model='infection_number',
              reporting_rate=1.0, calibration_window=WINDOW, censor_at_months=36.0, p_asymp_detect=0.4)
    mo_symp = P.process_model(df, p_symp_1=pars['p_symp_1'], p_symp_2=pars['p_symp_2'],
                              p_symp_3plus=pars['p_symp_3plus'], **kw)
    mo_all = P.process_model(df, p_symp_1=1.0, p_symp_2=1.0, p_symp_3plus=1.0, **kw)
    win = df[(df['CollectionTime'] >= WINDOW[0]) & (df['CollectionTime'] < WINDOW[1])]
    return dict(symp_ir=[float(mo_symp['ir_by_age'].loc[b, 'IR']) for b in BINS],
                all_ir=[float(mo_all['ir_by_age'].loc[b, 'IR']) for b in BINS],
                first_med=float(mo_symp['first_infection']['median']),
                windowed_infections=int(len(win)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--out-root', default=str(HERE))
    ap.add_argument('--n-agents', type=int, default=None)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    if args.n_agents:
        cfg['n_agents'] = args.n_agents
    out_root = pathlib.Path(args.out_root)
    (out_root / 'outputs').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    from calibrate_maled import SITE_DEMOGRAPHICS
    demo = SITE_DEMOGRAPHICS[cfg['site']]
    base = dict(n_agents=cfg['n_agents'],
                age_data_path=str(P.thisdir / 'uk_age_data.csv'),
                birth_rate=demo['birth_rate'], death_rate=demo['death_rate'])

    combos = list(itertools.product(cfg['infant_exposure_sweep'], cfg['maternal_mean_duration_sweep']))
    rng = np.random.default_rng(cfg['seed'])
    seeds = rng.integers(0, 1_000_000, len(combos)).tolist()
    c, w = cfg['cross_contacts'], cfg['young_reservoir_contacts']
    tasks = []
    for (iexp, mdur), s in zip(combos, seeds):
        pars = dict(cfg['fixed'])
        pars['base_beta'] = cfg['base_beta']
        pars['maternal_n_stages'] = cfg['maternal_n_stages']
        pars['maternal_mean_duration_days'] = mdur
        # rows=src, cols=dst, groups [inf, young, rest]; young->inf = iexp (low), young->young = w (high)
        matrix = [[c, c, c], [float(iexp), w, c], [c, c, c]]
        tasks.append((base, pars, matrix, int(s)))

    tgt_ir = np.array([P.load_targets(cfg['site'])['ir_by_age'].loc[b, 'IR'] for b in BINS])
    print(f"Sweep: {len(combos)} combos (base_beta x within_u5), {cfg['n_agents']} agents each.")
    with get_context('spawn').Pool(processes=min(len(tasks), 10)) as pool:
        results = pool.map(_run_mp, tasks)

    rows, jsonl = [], (out_root / 'outputs' / 'sweep.jsonl')
    with jsonl.open('w') as f:
        for (iexp, mdur), r in zip(combos, results):
            r2 = dict(infant_exposure=iexp, maternal_mean_duration=mdur, **r)
            rows.append(r2); f.write(json.dumps(r2) + '\n')

    print("\n=== symptomatic IR by age (infection-number) — peak at 6-11mo? ===")
    print(f"  {'DATA':<26} {[round(v,2) for v in tgt_ir]}  (peak 6-11mo)")
    any_peak = False
    for r in rows:
        ir = np.array(r['symp_ir'])
        peaked = (ir[1] > ir[0]) and (ir[1] >= ir[2]) and r['windowed_infections'] > 1000
        any_peak |= peaked
        tag = "<-- 6-11mo PEAK" if peaked else ("(faded)" if r['windowed_infections'] <= 1000 else "")
        print(f"  infExp={r['infant_exposure']:<5} matDur={r['maternal_mean_duration']:<5}: {[round(v,2) for v in r['symp_ir']]}  "
              f"first-med={r['first_med']:.1f} n={r['windowed_infections']} {tag}")
    print(f"\n  FEASIBILITY: {'PEAK ACHIEVED — age-structured contacts + sharp maternal can produce a 6-11mo peak' if any_peak else 'NO 6-11mo peak found in this sweep'}")

    # figure: symptomatic IR by age for every combo, vs data
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(BINS))
    ax.plot(x, tgt_ir, 'r*-', ms=16, lw=2.5, label='MAL-ED data', zorder=5)
    for r in rows:
        if r['windowed_infections'] <= 1000:
            continue
        ax.plot(x, r['symp_ir'], 'o-', alpha=0.7, label=f"infExp={r['infant_exposure']:.0f}, matDur={r['maternal_mean_duration']:.0f}d")
    ax.set_xticks(x); ax.set_xticklabels(BINS)
    ax.set_ylabel('Symptomatic IR /100 PM (infection-number)')
    ax.set_title('Exp 04 — age-structured contacts: symptomatic incidence by age\n(can MixingPools produce the 6-11mo peak? data in red)')
    ax.legend(frameon=False, fontsize=8); ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out_root / 'figures' / 'feasibility_peak.png', dpi=150)
    print(f"\nwrote {out_root / 'figures' / 'feasibility_peak.png'}")


if __name__ == '__main__':
    main()
