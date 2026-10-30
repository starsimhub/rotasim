"""
Verify reproducibility of Trial 46 from 50 trials calibration

This script tests whether we can reproduce the exact GOF value from the VM
by using the same parameters and different random seeds.
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import json

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("REPRODUCIBILITY VERIFICATION - Trial 46")
print("=" * 80)
print()

# Load Trial 46 parameters
with open('trial_46_params.json', 'r') as f:
    trial_data = json.load(f)

print("VM Results:")
print(f"  Trial: #{trial_data['trial_number']}")
print(f"  GOF: {trial_data['gof']:.10f}")
print()

params = trial_data['params']
print("Parameters:")
for key, value in params.items():
    print(f"  {key:<20}: {value}")
print()

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("Target Data:")
print(f"  Incidence: {target_incidence:.2f} per 100,000")
print(f"  Age distribution: {target_age_distribution.proportion.tolist()}")
print()


def compute_gof(sim, target_incidence, target_age_distribution, params):
    """Compute GOF exactly as in calibration script"""
    df = sim.analyzers['infectedstrainstats'].to_df()
    df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }

    overall_incidence, age_distribution = process_incidence_uk_age.process_model(
        df,
        age_counts=age_counts,
        symptom_model='age_and_infection_simple',
        beta0=params['beta0'],
        beta1=params['beta1'],
        beta2=params['beta2'],
        beta3=0,
        reporting_rate=params['reporting_rate']
    )

    # Compute GOF (same formula as calibration)
    eps = 1e-6
    log_target = np.log(target_incidence + eps)
    log_fitted = np.log(overall_incidence + eps)
    incidence_gof = (log_target - log_fitted) ** 2

    age_gof = 0.0
    for i in range(len(target_age_distribution)):
        target_prop = target_age_distribution.proportion.iloc[i]
        fitted_prop = age_distribution.proportion.iloc[i]
        age_gof += (target_prop - fitted_prop) ** 2

    total_gof = 10 * age_gof + incidence_gof

    return total_gof, overall_incidence, age_distribution


def run_simulation_with_seed(params, seed=None):
    """Run simulation with exact parameters and specified seed"""
    # Create base simulation
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)

    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')

    sim_kwargs = {
        'n_agents': 100000,
        'start': '2003-01-01',
        'stop': '2013-01-01',
        'verbose': False,
        'scenario': 'single',
        'people': people,
        'analyzers': [analyzer],
        'networks': ss.RandomNet(n_contacts=7),
        'demographics': [
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
        'interventions': [],
        'connectors': [immunity_connector],
    }

    if seed is not None:
        sim_kwargs['rand_seed'] = seed

    sim = rs.Sim(**sim_kwargs)

    # Update base_beta before initialization
    sim.pars.base_beta = params['base_beta']
    for disease in sim.pars.diseases:
        if isinstance(disease, rs.Rotavirus):
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

    # Initialize
    sim.init()

    # Configure immunity connector
    immunity_connector = sim.connectors.rotaimmunityconnector
    immunity_connector.pars['use_fixed_susceptibility'] = True
    immunity_connector.pars['sus_after_1'] = params['sus_after_1']
    immunity_connector.pars['sus_after_2'] = params['sus_after_2']
    immunity_connector.pars['sus_after_3plus'] = params['sus_after_3plus']

    # Initialize adult immunity
    immunity_connector.initialize_immunity(
        min_age=18, max_age=125, min_exposures=5, max_exposures=15
    )

    # Run
    sim.run()

    return sim


print("=" * 80)
print("TESTING REPRODUCIBILITY WITH DIFFERENT SEEDS")
print("=" * 80)
print()

# Test with different seeds
test_seeds = [None, 0, 1, 12345, 42, 123456]
results = []

for seed in test_seeds:
    print(f"Testing with seed={seed}...")
    sim = run_simulation_with_seed(params, seed=seed)
    gof, incidence, age_dist = compute_gof(sim, target_incidence, target_age_distribution, params)

    results.append({
        'seed': seed,
        'gof': gof,
        'incidence': incidence,
        'age_dist': age_dist.proportion.tolist()
    })

    diff = abs(gof - trial_data['gof'])
    match_str = "✓ MATCH!" if diff < 0.0001 else f"Δ = {diff:.6f}"

    print(f"  Seed {str(seed):<8} → GOF = {gof:.10f}  {match_str}")
    print(f"    Incidence: {incidence:.2f}")
    print(f"    Age dist: [{', '.join([f'{x:.4f}' for x in age_dist.proportion.tolist()])}]")
    print()

print("=" * 80)
print("SUMMARY")
print("=" * 80)
print()
print(f"VM GOF: {trial_data['gof']:.10f}")
print()
print("GOF Results:")
for r in results:
    diff = abs(r['gof'] - trial_data['gof'])
    print(f"  Seed {str(r['seed']):<8}: {r['gof']:.10f}  (diff = {diff:.10f})")

print()
gof_values = [r['gof'] for r in results]
print(f"Mean GOF: {np.mean(gof_values):.10f}")
print(f"Std GOF:  {np.std(gof_values):.10f}")
print(f"Min GOF:  {np.min(gof_values):.10f}")
print(f"Max GOF:  {np.max(gof_values):.10f}")
print()

# Check if any match
matches = [r for r in results if abs(r['gof'] - trial_data['gof']) < 0.0001]
if matches:
    print(f"✓ EXACT MATCH FOUND with seed={matches[0]['seed']}")
else:
    closest = min(results, key=lambda r: abs(r['gof'] - trial_data['gof']))
    diff = abs(closest['gof'] - trial_data['gof'])
    print(f"✗ No exact match found")
    print(f"  Closest: seed={closest['seed']}, diff={diff:.10f}")
    print()
    print("CONCLUSION:")
    print("  Stochastic variation in simulation affects GOF")
    print("  To reproduce exact GOF, need the simulation seed used on VM")
    print(f"  Current variation: ±{np.std(gof_values):.6f}")

print()
print("=" * 80)
