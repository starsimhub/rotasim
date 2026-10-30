"""
Multi-seed evaluation of fitted immunity calibration results
Assesses stability of incidence and age distribution fits across random seeds
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
from calibrate_infection_number_fitted_immunity import UKAgeCalibrationFittedImmunity
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Load best parameters
results_file = thisdir / 'uk_calibration_results_infection_number_fitted_immunity.json'
with open(results_file, 'r') as f:
    results = json.load(f)
    best_pars = results['best_parameters']

print("=" * 80)
print("MULTI-SEED EVALUATION: Fitted Immunity Model")
print("=" * 80)
print("\nBest parameters from calibration:")
for k, v in best_pars.items():
    if k.startswith('sus_'):
        protection = (1 - v) * 100
        print(f"  {k}: {v:.6f} ({protection:.2f}% protection)")
    else:
        print(f"  {k}: {v:.6f}")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print(f"\nTarget incidence: {target_incidence:.2f} per 100,000")
print("Target age distribution:")
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Number of runs
n_runs = 5
print(f"\n{'-' * 80}")
print(f"Running {n_runs} simulations with different random seeds...")
print(f"{'-' * 80}")

# Storage for results
incidence_results = []
age_dist_results = []

for run_idx in range(n_runs):
    seed = 12345 + run_idx

    # Create analyzer
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

    # Create immunity connector (will be configured during run_sim)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,
        sus_after_1=best_pars['sus_after_1'],
        sus_after_2=best_pars['sus_after_2'],
        sus_after_3plus=best_pars['sus_after_3plus'],
    )

    # Create base simulation
    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
    base_sim = rs.Sim(
        n_agents=100000,
        start='2003-01-01',
        stop='2013-01-01',
        verbose=False,
        scenario='single',
        people=people,
        analyzers=[analyzer],
        networks=ss.RandomNet(n_contacts=7),
        demographics=[
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
        interventions=[],
        connectors=[immunity_connector],
        rand_seed=seed,
    )

    # Create calibration object
    calib = UKAgeCalibrationFittedImmunity(
        sim=base_sim,
        data=(target_incidence, target_age_distribution),
        calib_pars={},
        total_trials=1,
        debug=False,
    )

    # Run simulation with best parameters
    print(f"Run {run_idx+1}/{n_runs} (seed={seed})...", end=' ', flush=True)
    sim = calib.run_sim(calib_pars=None, sim_pars=best_pars, trial=None)

    # Extract results
    overall_incidence, age_distribution = calib.sim_to_df(sim)

    # Store results
    incidence_results.append(overall_incidence)
    age_dist_results.append(age_distribution.proportion.values)

    print(f"Incidence: {overall_incidence:.2f}")

# Convert to arrays
incidence_results = np.array(incidence_results)
age_dist_results = np.array(age_dist_results)  # Shape: (n_runs, 4)

# Calculate statistics
incidence_mean = incidence_results.mean()
incidence_std = incidence_results.std()
incidence_ci_lower = np.percentile(incidence_results, 2.5)
incidence_ci_upper = np.percentile(incidence_results, 97.5)

age_dist_mean = age_dist_results.mean(axis=0)
age_dist_std = age_dist_results.std(axis=0)
age_dist_ci_lower = np.percentile(age_dist_results, 2.5, axis=0)
age_dist_ci_upper = np.percentile(age_dist_results, 97.5, axis=0)

# Display results
print("\n" + "=" * 80)
print("RESULTS SUMMARY")
print("=" * 80)

print("\n1. OVERALL INCIDENCE (per 100,000)")
print("-" * 80)
print(f"{'Metric':<25} {'Value':>15} {'95% CI':>25}")
print("-" * 80)
print(f"{'Target':<25} {target_incidence:>15.2f}")
print(f"{'Mean (n={n_runs})':<25} {incidence_mean:>15.2f} [{incidence_ci_lower:>8.2f}, {incidence_ci_upper:>8.2f}]")
print(f"{'Std Dev':<25} {incidence_std:>15.2f}")
error = incidence_mean - target_incidence
error_pct = (error / target_incidence) * 100
print(f"{'Error':<25} {error:>+15.2f} ({error_pct:>+6.2f}%)")

print("\n2. AGE DISTRIBUTION (Proportion of Cases)")
print("-" * 80)
print(f"{'Age Group':<20} {'Target':>10} {'Mean':>10} {'Std Dev':>10} {'95% CI':>20}")
print("-" * 80)
for i, label in enumerate(age_labels):
    target_val = target_age_distribution.proportion.iloc[i] * 100
    mean_val = age_dist_mean[i] * 100
    std_val = age_dist_std[i] * 100
    ci_low = age_dist_ci_lower[i] * 100
    ci_high = age_dist_ci_upper[i] * 100
    print(f"{label:<20} {target_val:>9.2f}% {mean_val:>9.2f}% {std_val:>9.2f}% [{ci_low:>5.2f}%, {ci_high:>5.2f}%]")

# Calculate GOF statistics
incidence_gof = abs(incidence_mean - target_incidence) / target_incidence
age_gof = np.abs((age_dist_mean - target_age_distribution.proportion.values) /
                 target_age_distribution.proportion.values).sum()
total_gof = incidence_gof + age_gof

print("\n3. GOODNESS OF FIT")
print("-" * 80)
print(f"Incidence GOF:        {incidence_gof:.4f}")
print(f"Age distribution GOF: {age_gof:.4f}")
print(f"Total GOF:            {total_gof:.4f}")

# Calculate within-target statistics
within_10pct = (np.abs(incidence_results - target_incidence) / target_incidence < 0.10).sum()
within_20pct = (np.abs(incidence_results - target_incidence) / target_incidence < 0.20).sum()

print("\n4. MODEL PERFORMANCE")
print("-" * 80)
print(f"Runs within 10% of target incidence: {within_10pct}/{n_runs} ({within_10pct/n_runs*100:.1f}%)")
print(f"Runs within 20% of target incidence: {within_20pct}/{n_runs} ({within_20pct/n_runs*100:.1f}%)")

# Age distribution fit quality
age_errors = np.abs(age_dist_mean - target_age_distribution.proportion.values) / target_age_distribution.proportion.values
max_age_error = age_errors.max() * 100
mean_age_error = age_errors.mean() * 100

print(f"Maximum age group error: {max_age_error:.2f}%")
print(f"Mean age group error:    {mean_age_error:.2f}%")

# Save results
output_data = {
    'model': 'infection_number_fitted_immunity',
    'n_runs': n_runs,
    'best_parameters': best_pars,
    'incidence': {
        'target': float(target_incidence),
        'mean': float(incidence_mean),
        'std': float(incidence_std),
        'ci_lower': float(incidence_ci_lower),
        'ci_upper': float(incidence_ci_upper),
        'error_percent': float(error_pct),
        'all_runs': incidence_results.tolist(),
    },
    'age_distribution': {
        'target': target_age_distribution.proportion.values.tolist(),
        'mean': age_dist_mean.tolist(),
        'std': age_dist_std.tolist(),
        'ci_lower': age_dist_ci_lower.tolist(),
        'ci_upper': age_dist_ci_upper.tolist(),
        'all_runs': age_dist_results.tolist(),
        'labels': age_labels,
    },
    'gof': {
        'incidence': float(incidence_gof),
        'age_distribution': float(age_gof),
        'total': float(total_gof),
    },
    'performance': {
        'within_10pct': int(within_10pct),
        'within_20pct': int(within_20pct),
        'max_age_error_pct': float(max_age_error),
        'mean_age_error_pct': float(mean_age_error),
    }
}

output_file = thisdir / 'fitted_immunity_multiseed_evaluation.json'
sc.savejson(output_file, output_data, indent=2)
print(f"\n✓ Results saved to: {output_file}")

print("\n" + "=" * 80)
print("EVALUATION COMPLETE")
print("=" * 80)
