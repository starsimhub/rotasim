"""
Test calibration with improved parameters:
- rel_beta range: [0.001, 0.1]
- immunity_waning_delay: 180 days
"""
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*60)
print("Testing Improved Calibration")
print("="*60)
print("\nChanges:")
print("  1. rel_beta range: [0.001, 0.1] (vs old [0.5, 1.5])")
print("  2. immunity_waning_delay: 180 days (vs 0.5 days)")
print("="*60)

# Create sim with parameters from tests/simple.py that sustain endemic circulation
sim = rs.Sim(
    n_agents=5000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    base_beta=0.16,  # From simple.py - will be modified by rel_beta during calibration
    override_prevalence=0.002,  # From simple.py (0.2%)
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),  # From simple.py - explicit contact network
    demographics=[
        ss.Births(birth_rate=ss.peryear(70)),  # From simple.py - higher birth rate for endemic circulation
        ss.Deaths(death_rate=ss.peryear(20)),  # From simple.py
    ],
)

# Get both overall incidence and age distribution from data
overall_incidence, age_distribution = process_incidence.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# JOINT CALIBRATION STRATEGY:
# 1. reporting_rate: Scale overall incidence magnitude (~0.2%)
# 2. maternal_immunity_efficacy: Protect infants 0-6 months (KEY for age distribution)
#    - Allow 0% to disable maternal immunity for sites with neonatal infections
# 3. maternal_immunity_half_life: Control duration of maternal protection
#    - Allow 0 days to disable maternal immunity
# 4. rel_beta: Adjust overall transmission intensity
# 5. reassortment_rate: Strain diversity
calib_pars = sc.objdict(
    reporting_rate=[0.002, 0.0001, 0.01],  # 0.01-1% surveillance capture (expanded range)
    maternal_immunity_efficacy=[0.85, 0.0, 0.95],  # 0-95% protection at birth (0% = no maternal immunity)
    maternal_immunity_half_life=[90, 1, 120],  # 1-120 days half-life (1 day ≈ disabled)
    rel_beta=[1.0, 0.5, 3.0],              # 0.5-3x transmission (expanded range)
    reassortment_rate=[0.10, 0.05, 0.15]   # 5-15% reassortment
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (20 trials)...")
print("="*60)

calib = Calibration(
    sim=sim,
    data=(overall_incidence, age_distribution),  # Pass as tuple
    calib_pars=calib_pars,
    total_trials=20,  # Increased for better parameter estimates
    debug=False,  # Run in parallel for all 20 trials
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
        target_prop = age_distribution.proportion.iloc[i] * 100  # Convert to percentage
        before_prop = calib.before_age_distribution.proportion.iloc[i] * 100
        after_prop = calib.after_age_distribution.proportion.iloc[i] * 100

        err_before = before_prop - target_prop  # Absolute difference in percentage points
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

if abs(err_after_inci) < 20 and calib.after_age_gof < 0.1:
    print("\n✓ Excellent fit: Both incidence and age distribution match well!")
elif abs(err_after_inci) < 50 and calib.after_age_gof < 0.2:
    print("\n✓ Good fit: Both metrics improved")
else:
    print("\n⚠ Model fit could be improved further")

print("\n✓ Test complete!")
