"""
Calibration script for UK data (2008-2012) with long-term immunity

UK Demographics:
- Population: [0.63%, 0.63%, 1.27%, 3.66%, 93.81%] for 6mo bins
- Aggregated to 4 bins: [1.26%, 1.27%, 3.66%, 93.81%]
- Birth rate: ~13/1000
- Death rate: ~6/1000 (accounting for net immigration of +4/1000)
- Follow-up period: 5 years (2008-2012)
- Burn-in: 5 years (2003-2007)
"""

import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*60)
print("UK Calibration (2008-2012) with Long-Term Immunity")
print("="*60)
print("\nUK Demographics:")
print("  Birth rate: 13/1000")
print("  Death rate: 6/1000")
print("  Net migration: +4/1000 adults (approximated in death rate)")
print("  Target age distribution: [1.26%, 1.27%, 3.66%, 93.81%]")
print("  Burn-in: 5 years (2003-2007)")
print("  Follow-up: 5 years (2008-2012)")
print("="*60)

# Create sim with 5-year burn-in (start in 2003)
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',  # 5 year burn-in before 2008
    stop='2013-01-01',   # End in 2012
    verbose=False,
    scenario='baseline',
    base_beta=0.40,  # Higher initial beta to account for long-term immunity
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),  # UK birth rate
        ss.Deaths(death_rate=ss.peryear(6)),   # UK death rate (adjusted for immigration)
    ],
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# Calibration parameters
calib_pars = sc.objdict(
    reporting_rate=[0.0002, 0.0001, 0.001],
    homotypic_immunity_efficacy=[0.5, 0.1, 0.9],
    partial_heterotypic_immunity_efficacy=[0.2, 0.0, 0.5],
    complete_heterotypic_immunity_efficacy=[0.1, 0.0, 0.3],
    base_beta=[0.40, 0.20, 1.50],  # Expanded range to allow higher transmission rates with long-term immunity
    maternal_immunity_efficacy=[0.0, 0.0, 0.0],
    # Long-term immunity parameters (calibrate these too)
    long_term_immunity_prob_after_1=[0.39, 0.2, 0.6],
    long_term_immunity_prob_after_2=[0.52, 0.3, 0.7],
    long_term_immunity_prob_after_3=[0.67, 0.4, 0.9],
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (30 trials)...")
print("="*60)

calib = Calibration(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=30,
    debug=False,
)

calib.calibrate()

print("\n" + "="*60)
print("Checking fit...")
print("="*60)
calib.check_fit()

print("\n" + "="*60)
print("Results:")
print("="*60)

print("\nBest parameters:")
for par, val in calib.best_pars.items():
    print(f"  {par}: {val:.6f}")

print("\n" + "="*60)
print("Comparing Overall Incidence:")
print("="*60)
print(f"Target:  {overall_incidence:.1f} per 100k")
print(f"Before:  {calib.before_overall_incidence:.1f} per 100k")
print(f"After:   {calib.after_overall_incidence:.1f} per 100k")
err_before_inci = (calib.before_overall_incidence - overall_incidence) / overall_incidence * 100
err_after_inci = (calib.after_overall_incidence - overall_incidence) / overall_incidence * 100
print(f"\nError before: {err_before_inci:+.1f}%")
print(f"Error after:  {err_after_inci:+.1f}%")

print("\n" + "="*60)
print("Comparing Age Distribution (proportions):")
print("="*60)
print(f"\n{'Age':<10} {'Target':<15} {'Before':<15} {'After':<15} {'Error Before':<20} {'Error After':<20}")
print("-"*100)

for i in range(len(age_distribution)):
    if i < len(calib.after_age_distribution):
        age = age_distribution.ages.iloc[i]
        target_prop = age_distribution.proportion.iloc[i] * 100
        before_prop = calib.before_age_distribution.proportion.iloc[i] * 100
        after_prop = calib.after_age_distribution.proportion.iloc[i] * 100

        err_before = before_prop - target_prop
        err_after = after_prop - target_prop

        print(f"{age:<10} {target_prop:<15.1f}% {before_prop:<15.1f}% {after_prop:<15.1f}% {err_before:<20.1f}pp {err_after:<20.1f}pp")

print("\n" + "="*60)
print("Summary:")
print("="*60)

print(f"\nOverall Incidence:")
print(f"  Target:  {overall_incidence:.1f} per 100k")
print(f"  Before:  {calib.before_overall_incidence:.1f} per 100k ({err_before_inci:+.1f}%)")
print(f"  After:   {calib.after_overall_incidence:.1f} per 100k ({err_after_inci:+.1f}%)")

improvement_inci = abs(err_before_inci) - abs(err_after_inci)
print(f"  Improvement: {improvement_inci:.1f} percentage points")

print(f"\nAge Distribution GOF:")
print(f"  Before: {calib.before_age_gof:.4f}")
print(f"  After:  {calib.after_age_gof:.4f}")
improvement_age = calib.before_age_gof - calib.after_age_gof
print(f"  Improvement: {improvement_age:.4f}")

if abs(err_after_inci) < 20 and calib.after_age_gof < 0.5:
    print("\n✓ Excellent fit: Both incidence and age distribution match well!")
elif abs(err_after_inci) < 50 and calib.after_age_gof < 1.0:
    print("\n✓ Good fit: Both metrics improved")
else:
    print("\n⚠ Model fit could be improved further")

# Save best parameters
print("\n" + "="*60)
print("Saving best parameters...")
print("="*60)
sc.save(thisdir / 'uk_best_pars_with_lti.obj', calib.best_pars)
print(f"Saved to: {thisdir / 'uk_best_pars_with_lti.obj'}")

print("\n✓ UK calibration complete!")
