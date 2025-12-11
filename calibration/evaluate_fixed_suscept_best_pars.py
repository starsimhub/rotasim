"""
Evaluate best parameters from fixed susceptibility calibration with multiple seeds
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibrate_infection_number_fixed_suscept import UKAgeCalibrationFixedSus
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Best parameters from calibration (Trial 19 from recalibration)
BEST_PARAMS = {
    'reporting_rate': 0.0021258099445399365,
    'base_beta': 6.973398294883567
}

N_RUNS = 10  # Multiple runs to handle stochasticity

print("=" * 60)
print("Evaluating Fixed Susceptibility Best Parameters")
print("=" * 60)
print("\nFixed Susceptibility Model:")
print("  After 1 infection:  rel_sus = 0.67 (33% protection)")
print("  After 2 infections: rel_sus = 0.50 (50% protection)")
print("  After 3+ infections: rel_sus = 0.36 (64% protection)")
print("\nBest parameters:")
for k, v in BEST_PARAMS.items():
    print(f"  {k}: {v:.6f}")
print(f"\nRunning {N_RUNS} stochastic simulations...")
print("=" * 60)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Storage for results
incidence_results = []
age_dist_results = {0: [], 1: [], 2: [], 5: []}  # age groups

for run in range(N_RUNS):
    print(f"\rRun {run+1}/{N_RUNS}...", end="", flush=True)

    # Create analyzer
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

    # Create immunity connector with fixed susceptibility
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,
        sus_after_1=0.67,
        sus_after_2=0.50,
        sus_after_3plus=0.36,
    )

    # Create sim with unique seed
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
        rand_seed=12345 + run,
    )

    # Create calibration object
    calib = UKAgeCalibrationFixedSus(
        sim=base_sim,
        data=(target_incidence, target_age_distribution),
        calib_pars={},
        total_trials=1,
        debug=False,
    )

    # Run simulation
    sim = calib.run_sim(calib_pars=None, sim_pars=BEST_PARAMS, trial=None)

    # Extract results
    overall_incidence, age_distribution = calib.sim_to_df(sim)

    incidence_results.append(overall_incidence)

    # Extract age distribution proportions
    for age_group in [0, 1, 2, 5]:
        prop = age_distribution.proportion.loc[age_group] if age_group in age_distribution.index else 0.0
        age_dist_results[age_group].append(prop)

print("\n\nProcessing results...")

# Calculate statistics
incidence_mean = np.mean(incidence_results)
incidence_std = np.std(incidence_results)
incidence_ci_lower = np.percentile(incidence_results, 2.5)
incidence_ci_upper = np.percentile(incidence_results, 97.5)

# Age distribution statistics
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
age_groups = [0, 1, 2, 5]
age_stats = {}

for i, (label, age_group) in enumerate(zip(age_labels, age_groups)):
    props = age_dist_results[age_group]
    age_stats[label] = {
        'mean': np.mean(props),
        'std': np.std(props),
        'ci_lower': np.percentile(props, 2.5),
        'ci_upper': np.percentile(props, 97.5),
        'target': target_age_distribution.proportion.iloc[i]
    }

# Display results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nOverall Incidence (per 100,000):")
print(f"  Target:      {target_incidence:.2f}")
print(f"  Mean:        {incidence_mean:.2f}")
print(f"  Std Dev:     {incidence_std:.2f}")
print(f"  95% CI:      [{incidence_ci_lower:.2f}, {incidence_ci_upper:.2f}]")
error = incidence_mean - target_incidence
error_pct = (error / target_incidence) * 100
print(f"  Error:       {error:+.2f} ({error_pct:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age Group':<15} {'Target':>10} {'Mean':>10} {'Std Dev':>10} {'95% CI':>25}")
print("-" * 80)
for label in age_labels:
    stats = age_stats[label]
    target = stats['target'] * 100
    mean = stats['mean'] * 100
    std = stats['std'] * 100
    ci_lower = stats['ci_lower'] * 100
    ci_upper = stats['ci_upper'] * 100
    print(f"{label:<15} {target:>9.2f}% {mean:>9.2f}% {std:>9.2f}% [{ci_lower:>6.2f}%, {ci_upper:>6.2f}%]")

# Calculate GOF
incidence_gof = abs(incidence_mean - target_incidence) / target_incidence

age_gof = 0.0
for i, age_group in enumerate(age_groups):
    target_prop = target_age_distribution.proportion.iloc[i]
    mean_prop = np.mean(age_dist_results[age_group])
    age_gof += abs(mean_prop - target_prop) / target_prop

total_gof = incidence_gof + age_gof

print(f"\nGoodness of Fit:")
print(f"  Incidence GOF:        {incidence_gof:.4f}")
print(f"  Age distribution GOF: {age_gof:.4f}")
print(f"  Total GOF:            {total_gof:.4f}")

# Save results
results_dict = {
    'model': 'infection_number_fixed_susceptibility',
    'susceptibility_model': {
        'type': 'fixed',
        'sus_after_1': 0.67,
        'sus_after_2': 0.50,
        'sus_after_3plus': 0.36
    },
    'best_parameters': BEST_PARAMS,
    'n_agents': 100000,
    'n_runs': N_RUNS,
    'target_incidence': float(target_incidence),
    'incidence': {
        'mean': float(incidence_mean),
        'std': float(incidence_std),
        'ci_lower': float(incidence_ci_lower),
        'ci_upper': float(incidence_ci_upper),
        'all_runs': [float(x) for x in incidence_results]
    },
    'error_percent': float(error_pct),
    'gof': {
        'incidence': float(incidence_gof),
        'age_distribution': float(age_gof),
        'total': float(total_gof)
    },
    'age_distribution': {
        'labels': age_labels,
        'target': target_age_distribution.proportion.tolist(),
        'statistics': {
            label: {
                'mean': float(stats['mean']),
                'std': float(stats['std']),
                'ci_lower': float(stats['ci_lower']),
                'ci_upper': float(stats['ci_upper'])
            }
            for label, stats in age_stats.items()
        },
        'all_runs': {
            str(ag): [float(x) for x in age_dist_results[ag]]
            for ag in age_groups
        }
    }
}

results_file = thisdir / 'uk_calibration_results_infection_number_fixed_sus_FINAL.json'
sc.savejson(results_file, results_dict, indent=2)
print(f"\n✓ Results saved to: {results_file}")

print("\n" + "=" * 60)
print("Evaluation complete!")
print("=" * 60)
