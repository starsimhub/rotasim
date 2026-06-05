"""
Exp 03 (validation) — confirm MALEDTargets reproduces the old
InfectedStrainStats -> process_incidence_maled.process_model path.

Runs ONE sim with BOTH analyzers attached, then compares IR-by-age and
first-infection quartiles.

Test 1 (exact): deterministic detection (beta0 huge -> symp prob ~1,
  p_asymp_detect=1). Both paths reduce to pure binning/ordering and must
  match EXACTLY (modulo float).
Test 2 (stochastic): realistic betas; the two use independent RNG streams, so
  they should agree within Poisson noise, not exactly.

Usage:
  uv run python experiments/03_coverage_check/validate_analyzer.py
"""
import sys
from pathlib import Path
import numpy as np
import starsim as ss
import sciris as sc
import rotasim as rs

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
CALIB_DIR = REPO / 'calibration'
AGE_DATA = CALIB_DIR / 'uk_age_data.csv'
if str(CALIB_DIR) not in sys.path:
    sys.path.insert(0, str(CALIB_DIR))
process_incidence_maled = sc.importbypath(CALIB_DIR / 'process_incidence_maled.py')
MALEDTargets = rs.MALEDTargets
CAL_WINDOW = (5.0, 10.0)
LABELS = rs.MALEDTargets.LABELS

N_AGENTS = 10_000
DEMO = dict(birth_rate=19, death_rate=6)


def build_and_run(params, p_asymp, seed=1):
    """Run one sim with BOTH analyzers; return (maled_out, df, person_months)."""
    targets = MALEDTargets(calibration_window=CAL_WINDOW,
                           beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'],
                           reporting_rate=1.0, constant_severity=1.0,
                           p_asymp_detect=p_asymp, seed=seed)
    iss = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=1.0)
    pt = rs.PersonTimeByAge(calibration_window=CAL_WINDOW)
    ic = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    people = ss.People(n_agents=N_AGENTS, age_data=str(AGE_DATA))
    sim = rs.Sim(n_agents=N_AGENTS, start='2003-01-01', stop='2013-01-01', dt=ss.days(1),
                 verbose=False, scenario='single', people=people,
                 analyzers=[targets, iss, pt],
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

    new_out = sim.analyzers['maledtargets'].results_dict()
    df = sim.analyzers['infectedstrainstats'].to_df()
    pm = sim.analyzers['persontimebyage'].person_months
    return new_out, df, pm


def old_path(df, pm, params, p_asymp, seed):
    return process_incidence_maled.process_model(
        df, person_months_by_bin=pm, symptom_model='age_and_infection_simple',
        beta0=params['beta0'], beta1=params['beta1'], beta2=params['beta2'],
        reporting_rate=1.0, calibration_window=CAL_WINDOW,
        censor_at_months=36.0, p_asymp_detect=p_asymp, rng_seed=seed)


def compare(tag, new_out, old, tol):
    print(f'\n--- {tag} ---')
    print(f'  {"bin":>8}  {"new IR":>9}  {"old IR":>9}  {"|diff|":>8}   '
          f'{"newCase":>7} {"oldCase":>7}   {"newPM":>9} {"oldPM":>9}')
    ok = True
    for b in LABELS:
        n = new_out['ir'][b]
        o = float(old['ir_by_age'].loc[b, 'IR'])
        d = abs(n - o)
        ok &= (d <= tol * max(1.0, o))
        nc = new_out['cases'][b]; oc = int(old['ir_by_age'].loc[b, 'cases'])
        npm = new_out['person_months'][b]; opm = float(old['ir_by_age'].loc[b, 'PT'])
        print(f'  {b:>8}  {n:>9.4f}  {o:>9.4f}  {d:>8.4f}   '
              f'{nc:>7} {oc:>7}   {npm:>9.1f} {opm:>9.1f}')
    of = old['first_infection']
    print(f'  first-inf  new med={new_out["fi_median"]:.3f} q25={new_out["fi_q25"]:.3f} q75={new_out["fi_q75"]:.3f}')
    print(f'             old med={of["median"]:.3f} q25={of["q25"]:.3f} q75={of["q75"]:.3f}')
    return ok


def main():
    base = dict(base_beta=0.2, beta1=0.0, beta2=0.0,
                sus_after_3plus=0.5, sus_after_2=0.7, sus_after_1=0.95,
                maternal_immunity_efficacy=0.8, maternal_immunity_half_life_days=150.0)

    # Test 1: deterministic detection (beta0 huge -> symp~1, p_asymp=1).
    p1 = dict(base, beta0=20.0)
    new1, df1, pm1 = build_and_run(p1, p_asymp=1.0, seed=1)
    old1 = old_path(df1, pm1, p1, p_asymp=1.0, seed=1)
    ok1 = compare('TEST 1 (deterministic, must match exactly)', new1, old1, tol=1e-6)
    print(f'  => {"EXACT MATCH" if ok1 else "MISMATCH"} on IR')

    # Test 2: realistic stochastic detection — expect agreement within noise.
    p2 = dict(base, beta0=-1.0, beta1=-0.1, beta2=-0.05)
    new2, df2, pm2 = build_and_run(p2, p_asymp=0.4, seed=2)
    old2 = old_path(df2, pm2, p2, p_asymp=0.4, seed=2)
    ok2 = compare('TEST 2 (stochastic, within ~15% noise)', new2, old2, tol=0.15)
    print(f'  => {"agrees within tol" if ok2 else "OUTSIDE tol (investigate)"}')

    print(f'\nRESULT: deterministic-exact={ok1}, stochastic-within-noise={ok2}')


if __name__ == '__main__':
    main()
