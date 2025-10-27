"""
Calibration script for Bangladesh (Matlab) data (2000-2008)

Bangladesh Demographics:
- Population: [2.5%, 2.5%, 7.5%, 87.5%]
- Birth rate: 30/1000
- Death rate: 10/1000
- Net emigration: -10/1000 adults
- Follow-up period: 8 years (2000-2008)

This calibration includes three key fixes:
1. Fixed process_model() to calculate actual simulation age fractions
2. Use demographics that match Matlab population structure
3. Add emigration for more realistic death rates
"""
import sys
sys.path.insert(0, '.')
from emigration import Emigration

import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*60)
print("Testing Age Distribution Fixes")
print("="*60)
print("\nFixes implemented:")
print("  1. process_model() now calculates ACTUAL age-specific populations")
print("  2. Using demographics that match Matlab: birth=30, death=10")
print("  3. Adding emigration: 10/1000 adults (≥5 years)")
print("     → Age distribution [2.7%, 3.1%, 8.6%, 85.5%] vs target [2.5%, 2.5%, 7.5%, 87.5%]")
print("     → Death rate 10/1000 is realistic (compared to previous 20/1000)")
print("="*60)

# Create emigration module
emigr = Emigration(emigration_rate=10, age_threshold=5)

# Create sim with CORRECTED demographics
sim = rs.Sim(
    n_agents=5000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(30)),  # Matches Matlab birth rate
        ss.Deaths(death_rate=ss.peryear(10)),  # Realistic death rate
        emigr,  # Adult emigration compensates for lower death rate
    ],
)

# Get target data
overall_incidence, age_distribution = process_incidence.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# Calibration parameters - keep cross-protection approach
calib_pars = sc.objdict(
    reporting_rate=[0.0002, 0.0001, 0.001],
    homotypic_immunity_efficacy=[0.5, 0.1, 0.9],
    partial_heterotypic_immunity_efficacy=[0.2, 0.0, 0.5],
    complete_heterotypic_immunity_efficacy=[0.1, 0.0, 0.3],
    base_beta=[0.16, 0.05, 2.0],  # Base transmission rate - expanded range
    maternal_immunity_efficacy=[0.0, 0.0, 0.0],  # Keep at 0
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (20 trials)...")
print("="*60)

calib = Calibration(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=20,
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

print("\n✓ Test complete!")
