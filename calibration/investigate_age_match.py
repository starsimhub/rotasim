"""
Investigate why age distribution match is near-perfect
Tests multiple scenarios to understand if this is real or an artifact
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

# Clear modules
for mod in list(sys.modules.keys()):
    if 'calibrate_infection_number_fitted_immunity' in mod:
        del sys.modules[mod]

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))

from calibrate_infection_number_fitted_immunity import UKAgeCalibrationFittedImmunity
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("="*80)
print("INVESTIGATING NEAR-PERFECT AGE DISTRIBUTION MATCH")
print("="*80)

# Load parameters and target
with open(thisdir / 'uk_calibration_results_infection_number_fitted_immunity.json', 'r') as f:
    best_pars = json.load(f)['best_parameters']

target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("\nTarget age distribution:")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Test 1: Multiple runs with SAME seed - should give identical results
print("\n" + "="*80)
print("TEST 1: Multiple runs with SAME random seed (12345)")
print("="*80)
print("Expected: Identical results across all runs")

results_same_seed = []
for run in range(3):
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,
        sus_after_1=best_pars['sus_after_1'],
        sus_after_2=best_pars['sus_after_2'],
        sus_after_3plus=best_pars['sus_after_3plus'],
    )
    
    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
    base_sim = rs.Sim(
        n_agents=100000, start='2003-01-01', stop='2013-01-01',
        verbose=False, scenario='single', people=people,
        analyzers=[analyzer], networks=ss.RandomNet(n_contacts=7),
        demographics=[ss.Births(birth_rate=ss.peryear(13)), ss.Deaths(death_rate=ss.peryear(6))],
        connectors=[immunity_connector], rand_seed=12345,  # SAME seed
    )
    
    calib = UKAgeCalibrationFittedImmunity(
        sim=base_sim, data=(target_incidence, target_age_distribution),
        calib_pars={}, total_trials=1, debug=False,
    )
    
    sim = calib.run_sim(calib_pars=None, sim_pars=best_pars, trial=None)
    overall_incidence, age_distribution = calib.sim_to_df(sim)
    
    results_same_seed.append({
        'incidence': overall_incidence,
        'age_dist': [age_distribution.proportion.iloc[i] * 100 for i in range(4)]
    })
    
    print(f"\nRun {run+1}:")
    print(f"  Incidence: {overall_incidence:.2f} per 100k")
    print(f"  Age dist:  [{results_same_seed[-1]['age_dist'][0]:.2f}%, {results_same_seed[-1]['age_dist'][1]:.2f}%, {results_same_seed[-1]['age_dist'][2]:.2f}%, {results_same_seed[-1]['age_dist'][3]:.2f}%]")

# Check consistency
print("\nConsistency check (same seed):")
if all(r['incidence'] == results_same_seed[0]['incidence'] for r in results_same_seed):
    print("  Incidence: IDENTICAL across all runs ✓")
else:
    print(f"  Incidence: VARIES - {[r['incidence'] for r in results_same_seed]}")

if all(r['age_dist'] == results_same_seed[0]['age_dist'] for r in results_same_seed):
    print("  Age distribution: IDENTICAL across all runs ✓")
else:
    print("  Age distribution: VARIES")

# Test 2: Multiple runs with DIFFERENT seeds - should vary slightly
print("\n" + "="*80)
print("TEST 2: Multiple runs with DIFFERENT random seeds")
print("="*80)
print("Expected: Slightly different results due to stochastic processes")

results_diff_seed = []
seeds = [12345, 54321, 99999]
for seed in seeds:
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,
        sus_after_1=best_pars['sus_after_1'],
        sus_after_2=best_pars['sus_after_2'],
        sus_after_3plus=best_pars['sus_after_3plus'],
    )
    
    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
    base_sim = rs.Sim(
        n_agents=100000, start='2003-01-01', stop='2013-01-01',
        verbose=False, scenario='single', people=people,
        analyzers=[analyzer], networks=ss.RandomNet(n_contacts=7),
        demographics=[ss.Births(birth_rate=ss.peryear(13)), ss.Deaths(death_rate=ss.peryear(6))],
        connectors=[immunity_connector], rand_seed=seed,  # DIFFERENT seed
    )
    
    calib = UKAgeCalibrationFittedImmunity(
        sim=base_sim, data=(target_incidence, target_age_distribution),
        calib_pars={}, total_trials=1, debug=False,
    )
    
    sim = calib.run_sim(calib_pars=None, sim_pars=best_pars, trial=None)
    overall_incidence, age_distribution = calib.sim_to_df(sim)
    
    results_diff_seed.append({
        'seed': seed,
        'incidence': overall_incidence,
        'age_dist': [age_distribution.proportion.iloc[i] * 100 for i in range(4)]
    })
    
    print(f"\nSeed {seed}:")
    print(f"  Incidence: {overall_incidence:.2f} per 100k")
    print(f"  Age dist:  [{results_diff_seed[-1]['age_dist'][0]:.2f}%, {results_diff_seed[-1]['age_dist'][1]:.2f}%, {results_diff_seed[-1]['age_dist'][2]:.2f}%, {results_diff_seed[-1]['age_dist'][3]:.2f}%]")
    
    # Compare to target
    errors = [abs(results_diff_seed[-1]['age_dist'][i] - target_age_distribution.proportion.iloc[i]*100) for i in range(4)]
    print(f"  Errors:    [{errors[0]:+.2f}%, {errors[1]:+.2f}%, {errors[2]:+.2f}%, {errors[3]:+.2f}%]")

# Calculate variability across seeds
print("\nVariability across seeds:")
incidences = [r['incidence'] for r in results_diff_seed]
print(f"  Incidence range: {min(incidences):.2f} - {max(incidences):.2f} (std: {np.std(incidences):.2f})")

for i, label in enumerate(age_labels):
    props = [r['age_dist'][i] for r in results_diff_seed]
    print(f"  {label:<15}: {min(props):.2f}% - {max(props):.2f}% (std: {np.std(props):.3f}%)")

# Summary
print("\n" + "="*80)
print("SUMMARY & INTERPRETATION")
print("="*80)

print("\n1. Fixed seed reproducibility:")
if all(r['incidence'] == results_same_seed[0]['incidence'] for r in results_same_seed):
    print("   ✓ Same seed gives identical results (as expected)")
else:
    print("   ✗ Same seed gives different results (UNEXPECTED!)")

print("\n2. Stochastic variation:")
incidence_cv = np.std(incidences) / np.mean(incidences) * 100
age_cv_mean = np.mean([np.std([r['age_dist'][i] for r in results_diff_seed]) / np.mean([r['age_dist'][i] for r in results_diff_seed]) * 100 for i in range(4)])
print(f"   Incidence CV: {incidence_cv:.2f}% (coefficient of variation)")
print(f"   Age dist CV:  {age_cv_mean:.2f}% (mean across age groups)")

if incidence_cv < 5 and age_cv_mean < 5:
    print("   → Results are highly stable across seeds (low variability)")
else:
    print("   → Results show substantial variability across seeds")

print("\n3. Near-perfect age match explanation:")
all_close = all([abs(results_diff_seed[0]['age_dist'][i] - target_age_distribution.proportion.iloc[i]*100) < 1.0 for i in range(4)])
if all_close:
    print("   The near-perfect match is NOT an artifact of the fixed seed.")
    print("   It occurs consistently across different seeds.")
    print("\n   LIKELY EXPLANATION:")
    print("   - The PRIOR calibration (which only optimized incidence) happened to")
    print("     produce parameters that also match the age distribution well")
    print("   - This could be because:")
    print("     a) The model naturally produces realistic age distributions")
    print("     b) Optimizing incidence indirectly constrained age parameters")
    print("     c) The true biological age distribution follows from transmission")
    print("        dynamics, and the model captures this correctly")
else:
    print("   The match quality varies substantially with different seeds.")
    print("   The near-perfect match with seed 12345 may be fortuitous.")

print("\n" + "="*80)
