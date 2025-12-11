"""
Fresh 20-trial calibration of infection_number model with fitted immunity
Uses FIXED age distribution extraction code
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna
import json
from pathlib import Path

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibrate_infection_number_fitted_immunity import UKAgeCalibrationFittedImmunity
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("FRESH 20-TRIAL CALIBRATION: Fitted Immunity Model (FIXED Age Extraction)")
print("=" * 80)
print("\nThis calibration uses the FIXED age distribution extraction code.")
print("Both incidence AND age distribution will be properly optimized.\n")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print(f"Target incidence: {target_incidence:.2f} per 100,000")
print("Target age distribution:")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Define calibration parameters
calib_pars = dict(
    reporting_rate=[0.01, 0.001, 0.05],  # [best_guess, lower_bound, upper_bound]
    base_beta=[2.5, 1.0, 8.0],
    sus_after_1=[0.85, 0.5, 1.0],       # After 1st infection
    sus_after_2=[0.7, 0.3, 0.95],        # After 2nd infection  
    sus_after_3plus=[0.5, 0.1, 0.9],     # After 3+ infections
)

print("\n" + "-" * 80)
print("Calibration Parameter Ranges:")
print("-" * 80)
for param, (guess, lower, upper) in calib_pars.items():
    print(f"  {param:<20}: [{lower:.3f}, {upper:.3f}] (guess: {guess:.3f})")

# Create base simulation (will be configured with immunity connector in run_sim)
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
base_sim = rs.Sim(
    n_agents=100000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    people=people,
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[],
)

# Create calibration object
n_trials = 20
print(f"\n{'=' * 80}")
print(f"Starting calibration with {n_trials} trials...")
print(f"{'=' * 80}\n")

calib = UKAgeCalibrationFittedImmunity(
    sim=base_sim,
    data=(target_incidence, target_age_distribution),
    calib_pars=calib_pars,
    total_trials=n_trials,
    debug=False,
)

# Run calibration
study = calib.calibrate()

# Extract results
best_pars = study.best_params
best_gof = study.best_value

print("\n" + "=" * 80)
print("CALIBRATION COMPLETE")
print("=" * 80)
print(f"\nBest GOF: {best_gof:.6f}")
print("\nBest parameters:")
for k, v in best_pars.items():
    if k.startswith('sus_'):
        protection = (1 - v) * 100
        print(f"  {k:<20}: {v:.6f} ({protection:.2f}% protection)")
    else:
        print(f"  {k:<20}: {v:.6f}")

# Save results
output_file = thisdir / 'uk_calibration_results_infection_number_fitted_immunity_20trials.json'
results = {
    'model': 'infection_number_fitted_immunity',
    'description': 'Fresh 20-trial calibration with FIXED age distribution extraction',
    'n_trials': n_trials,
    'best_parameters': best_pars,
    'best_gof': best_gof,
    'target_incidence': float(target_incidence),
    'calibration_pars': {k: list(v) for k, v in calib_pars.items()},
}

sc.savejson(output_file, results, indent=2)
print(f"\n✓ Results saved to: {output_file}")

print("\n" + "=" * 80)
print("Next step: Run multi-seed evaluation with best parameters")
print("=" * 80)
