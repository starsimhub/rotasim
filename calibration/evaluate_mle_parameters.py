"""Evaluate age_and_infection_simple model at MLE parameters"""
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
from calibrate_uk_age_model import UKAgeCalibration
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# MLE parameters from user
MLE_PARAMS = {
    'reporting_rate': 0.0284,
    'base_beta': 2.082,
    'beta0': -1.421,
    'beta1': 0.171,
    'beta2': -0.004516
}

N_AGENTS = 100000

print("=" * 60)
print("EVALUATING: age_and_infection_simple at MLE Parameters")
print("=" * 60)
print(f"\nMLE Parameters:")
for k, v in MLE_PARAMS.items():
    print(f"  {k}: {v:.6f}")
print(f"\nUsing {N_AGENTS} agents...")
print("=" * 60)

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Create analyzer with FIXED 5% severity
analyzer = rs.InfectedStrainStats(
    use_infection_based_severity=False,
    constant_severity=0.05
)

# Create base simulation
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
    rand_seed=12345,
)

# Create calibration object
calib = UKAgeCalibration(
    sim=base_sim,
    data=(target_incidence, target_age_distribution),
    calib_pars={},
    total_trials=1,
    debug=False,
    symptom_model='age_and_infection_simple',
)

# Run simulation using calibration framework
print("\nRunning simulation at MLE parameters...")
sim = calib.run_sim(calib_pars=MLE_PARAMS, sim_pars=MLE_PARAMS, trial=None)

# Extract results
print("Processing results...")
overall_incidence, age_distribution = calib.sim_to_df(sim)

# Display results
print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)

print(f"\nOverall Incidence (per 100,000):")
print(f"  Target:  {target_incidence:.2f}")
print(f"  Fitted:  {overall_incidence:.2f}")
error = overall_incidence - target_incidence
error_pct = (error / target_incidence) * 100
print(f"  Error:   {error:+.2f} ({error_pct:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age Group':<15} {'Target':>10} {'Fitted':>10} {'Difference':>12}")
print("-" * 50)
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution.proportion.iloc[i]
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

# Calculate GOF metrics
incidence_gof = abs(overall_incidence - target_incidence) / target_incidence
age_gof = ((age_distribution.proportion - target_age_distribution.proportion).abs() / target_age_distribution.proportion).sum()
total_gof = incidence_gof + age_gof

print(f"\nGoodness of Fit:")
print(f"  Incidence GOF:        {incidence_gof:.4f}")
print(f"  Age distribution GOF: {age_gof:.4f}")
print(f"  Total GOF:            {total_gof:.4f}")

# Save results
results_dict = {
    'model': 'age_and_infection_simple',
    'severity': 0.05,
    'mle_parameters': MLE_PARAMS,
    'n_agents': N_AGENTS,
    'target_incidence': float(target_incidence),
    'fitted_incidence': float(overall_incidence),
    'error_percent': float(error_pct),
    'gof': {
        'incidence': float(incidence_gof),
        'age_distribution': float(age_gof),
        'total': float(total_gof)
    },
    'age_distribution': {
        'target': target_age_distribution.proportion.tolist(),
        'fitted': age_distribution.proportion.tolist(),
        'labels': age_labels
    }
}

results_file = thisdir / 'age_simple_mle_evaluation.json'
sc.savejson(results_file, results_dict, indent=2)
print(f"\n✓ Results saved to: {results_file}")

print("\n" + "=" * 60)
print("Evaluation complete!")
print("=" * 60)
