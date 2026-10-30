"""
Evaluate all three age models at MLE parameters with uncertainty quantification

Runs multiple replicates with different random seeds for each model:
1. infection_number
2. age_and_infection
3. age_and_infection_simple
"""

# Fix for PyCharm: Remove parent directory from sys.path
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import pandas as pd
import json

thisdir = sc.thispath(__file__)
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Import the calibration class
sys.path.insert(0, str(thisdir))
from calibrate_uk_age_model import UKAgeCalibration

# ============================================================
# CONFIGURATION
# ============================================================
N_REPLICATES = 10  # Number of replicates per model
N_AGENTS = 100000  # Match calibration population size

# Load MLE parameters from calibration results
with open(thisdir / 'uk_calibration_results_infection_number.json', 'r') as f:
    infection_number_results = json.load(f)

with open(thisdir / 'uk_calibration_results_age_and_infection.json', 'r') as f:
    age_and_infection_results = json.load(f)

with open(thisdir / 'uk_calibration_results_age_and_infection_simple.json', 'r') as f:
    age_and_infection_simple_results = json.load(f)

MLE_PARAMS = {
    'infection_number': infection_number_results['best_parameters'],
    'age_and_infection': age_and_infection_results['best_parameters'],
    'age_and_infection_simple': age_and_infection_simple_results['best_parameters'],
}

print("="*80)
print("MULTIPLE-SEED VALIDATION FOR ALL THREE MODELS")
print("="*80)
print(f"\nPopulation size: {N_AGENTS:,} agents (matching calibration)")
print(f"Running {N_REPLICATES} replicates per model with different random seeds\n")


def extract_age_specific_population_counts(sim):
    """Extract actual age-specific population counts from simulation"""
    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }
    return age_counts


def run_single_simulation(calib_obj, params, replicate_num, symptom_model, base_seed=12345):
    """
    Run a single simulation using the calibration framework

    Args:
        calib_obj: UKAgeCalibration instance
        params: Dictionary of parameters
        replicate_num: Replicate number (0-indexed)
        symptom_model: Model type string
        base_seed: Base random seed
    """
    # Create unique seed for this replicate
    sim_seed = base_seed + replicate_num

    # Set seed in parameters
    params_with_seed = params.copy()
    params_with_seed['rand_seed'] = sim_seed

    # Use calibration's run_sim method with parameters
    sim = calib_obj.run_sim(calib_pars=params_with_seed, sim_pars=params_with_seed, trial=None)

    # Extract results using calibration's sim_to_df method
    overall_incidence, age_distribution = calib_obj.sim_to_df(sim)

    return overall_incidence, age_distribution


def evaluate_model(symptom_model, params, n_replicates=10):
    """Evaluate a single model with multiple replicates"""
    print(f"\n{'='*80}")
    print(f"Evaluating: {symptom_model.upper()}")
    print(f"{'='*80}")
    print(f"MLE Parameters:")
    for key, val in params.items():
        print(f"  {key}: {val:.6f}")
    print(f"\nRunning {n_replicates} replicates...")

    # Create analyzer with appropriate severity settings
    if symptom_model in ['infection_number', 'age_and_infection']:
        analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
    elif symptom_model == 'age_and_infection_simple':
        analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.05)
    else:
        raise ValueError(f"Unknown symptom model: {symptom_model}")

    # Create base sim
    people = ss.People(n_agents=N_AGENTS, age_data=thisdir / 'uk_age_data.csv')
    base_sim = rs.Sim(
        n_agents=N_AGENTS,
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
    )

    # Get target data
    target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

    # Create calibration object
    calib = UKAgeCalibration(
        sim=base_sim,
        data=(target_incidence, target_age_distribution),
        calib_pars={},
        total_trials=1,
        debug=False,
        symptom_model=symptom_model,
    )

    # Run replicates
    results = []
    for i in range(n_replicates):
        print(f"  Replicate {i + 1}/{n_replicates}...", end='', flush=True)
        overall_inc, age_dist = run_single_simulation(
            calib, params, replicate_num=i, symptom_model=symptom_model, base_seed=12345
        )
        results.append({
            'overall_incidence': overall_inc,
            'age_distribution': age_dist,
            'replicate': i,
        })
        print(" done")

    # Calculate summary statistics
    incidence_values = [r['overall_incidence'] for r in results]
    age_distributions = [r['age_distribution']['proportion'].values for r in results]

    mean_incidence = np.mean(incidence_values)
    std_incidence = np.std(incidence_values)
    mean_age_dist = np.mean(age_distributions, axis=0)
    std_age_dist = np.std(age_distributions, axis=0)

    # Print summary
    print(f"\n{'='*80}")
    print(f"Summary: {symptom_model.upper()}")
    print(f"{'='*80}")
    print(f"\nOverall Incidence (per 100k):")
    print(f"  Target:     {target_incidence:.2f}")
    print(f"  Mean:       {mean_incidence:.2f} ± {std_incidence:.2f}")
    print(f"  Range:      [{np.min(incidence_values):.2f}, {np.max(incidence_values):.2f}]")
    print(f"  Error:      {mean_incidence - target_incidence:+.2f} ({(mean_incidence - target_incidence)/target_incidence*100:+.1f}%)")
    if mean_incidence > 0:
        print(f"  CV:         {std_incidence/mean_incidence*100:.2f}%")

    print(f"\nAge Distribution (proportions):")
    print(f"{'Age':<12} {'Target':>10} {'Mean ± SD':>25} {'Range':>20}")
    print("-"*70)
    age_labels = ['0-11mo', '12-23mo', '24-59mo', '5+yr']
    target_props = target_age_distribution['proportion'].values
    for i, label in enumerate(age_labels):
        target = target_props[i] * 100
        mean_val = mean_age_dist[i] * 100
        std_val = std_age_dist[i] * 100
        min_val = min([ad[i] for ad in age_distributions]) * 100
        max_val = max([ad[i] for ad in age_distributions]) * 100
        cv = (std_val / mean_val * 100) if mean_val > 0 else 0
        print(f"{label:<12} {target:>7.1f}% {mean_val:>7.1f} ± {std_val:>5.1f}% (CV: {cv:>5.1f}%)   [{min_val:>5.1f}, {max_val:>5.1f}]%")

    return {
        'symptom_model': symptom_model,
        'mean_incidence': mean_incidence,
        'std_incidence': std_incidence,
        'mean_age_dist': mean_age_dist,
        'std_age_dist': std_age_dist,
        'raw_results': results,
    }


# ============================================================
# Run evaluations for all three models
# ============================================================
all_results = {}

for model_name in ['infection_number', 'age_and_infection', 'age_and_infection_simple']:
    all_results[model_name] = evaluate_model(
        model_name,
        MLE_PARAMS[model_name],
        n_replicates=N_REPLICATES
    )

# ============================================================
# Final comparison
# ============================================================
print(f"\n{'='*80}")
print("FINAL COMPARISON ACROSS ALL MODELS")
print(f"{'='*80}")
print(f"\n{'Model':<30} {'Mean Incidence':>20} {'Error':>15} {'CV':>10}")
print("-"*80)

target_incidence, _ = process_incidence_uk_age.process_data()
for model_name in ['infection_number', 'age_and_infection', 'age_and_infection_simple']:
    res = all_results[model_name]
    mean_inc = res['mean_incidence']
    std_inc = res['std_incidence']
    error_pct = (mean_inc - target_incidence) / target_incidence * 100
    cv = (std_inc / mean_inc * 100) if mean_inc > 0 else 0

    print(f"{model_name:<30} {mean_inc:>10.2f} ± {std_inc:>6.2f} {error_pct:>12.1f}% {cv:>9.1f}%")

print(f"\n{'='*80}")
print("Evaluation complete!")
print(f"{'='*80}")

# Save results
output_file = thisdir / 'all_models_evaluation_results.json'
results_to_save = {}
for model_name, res in all_results.items():
    results_to_save[model_name] = {
        'mean_incidence': float(res['mean_incidence']),
        'std_incidence': float(res['std_incidence']),
        'mean_age_dist': res['mean_age_dist'].tolist(),
        'std_age_dist': res['std_age_dist'].tolist(),
    }

with open(output_file, 'w') as f:
    json.dump(results_to_save, f, indent=2)

print(f"\n✓ Results saved to: {output_file}")
