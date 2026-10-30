"""
Show the age distribution comparison between model and data
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import sciris as sc
from calibrate_uk import UKCalibration

# Best-fit parameters from the calibration
best_pars = {
    'reporting_rate': 0.39689528254519607,
    'homotypic_immunity_efficacy': 0.4888011311794621,
    'partial_heterotypic_immunity_efficacy': 0.234657454216119,
    'complete_heterotypic_immunity_efficacy': 0.28412525957726026,
    'base_beta': 0.2908432464797052,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9158414206368015
}

# Target data age distribution
target_age_dist = sc.dataframe({
    'ages': [0, 1, 2, 5],
    'proportion': [0.137698, 0.276685, 0.468881, 0.116737]
})

print("="*80)
print("AGE DISTRIBUTION COMPARISON")
print("="*80)
print("\nTarget data (UK 2008-2012):")
print(target_age_dist)
print(f"\nTotal: {target_age_dist['proportion'].sum():.6f}")

# Create calibration object
calib = UKCalibration()

# Run simulation with best-fit parameters
print("\n" + "="*80)
print("Running simulation with best-fit parameters...")
print("="*80)
sim = calib.run_sim(pars_dict=best_pars, return_sim=True)

# Get the model output
overall_incidence, model_age_dist = UKCalibration.sim_to_df(sim)

print(f"\nModel overall incidence: {overall_incidence:.2f} per 100k")
print(f"Target overall incidence: 1.4 per 100k")
print(f"Difference: {overall_incidence - 1.4:.2f} per 100k")

print("\n" + "="*80)
print("Model age distribution (proportions):")
print("="*80)
print(model_age_dist)
print(f"\nTotal: {model_age_dist['proportion'].sum():.6f}")

print("\n" + "="*80)
print("SIDE-BY-SIDE COMPARISON")
print("="*80)
print(f"{'Age Group':<15} {'Target':<15} {'Model':<15} {'Difference':<15}")
print("-"*60)

# Match up the age groups
for idx, row in target_age_dist.iterrows():
    age_group = row['ages']
    target_prop = row['proportion']

    # Find corresponding model proportion
    model_row = model_age_dist[model_age_dist['ages'] == age_group]
    if len(model_row) > 0:
        model_prop = model_row['proportion'].values[0]
    else:
        model_prop = 0.0

    diff = model_prop - target_prop

    print(f"{age_group:<15} {target_prop:<15.4f} {model_prop:<15.4f} {diff:<15.4f}")

print("="*80)
