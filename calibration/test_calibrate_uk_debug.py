"""
Debug version of UK calibration with reduced trials
"""

import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*60)
print("UK Calibration Debug Test")
print("="*60)

# Create sim
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk.process_data()
print(f"\nTarget incidence: {overall_incidence:.1f} per 100k")
print(f"Target age distribution shape: {age_distribution.shape}")
print(age_distribution)

# Calibration parameters
calib_pars = sc.objdict(
    reporting_rate=[0.0002, 0.0001, 0.001],
    homotypic_immunity_efficacy=[0.5, 0.1, 0.9],
    partial_heterotypic_immunity_efficacy=[0.2, 0.0, 0.5],
    complete_heterotypic_immunity_efficacy=[0.1, 0.0, 0.3],
    base_beta=[0.16, 0.05, 0.30],
    maternal_immunity_efficacy=[0.0, 0.0, 0.0],
)

print("\nCreating calibration object...")
calib = Calibration(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=2,  # Just 2 trials for debugging
    debug=True,
)

print("\nRunning calibration...")
try:
    calib.calibrate()
    print("\n✓ Calibration completed successfully")
except Exception as e:
    print(f"\n✗ Calibration failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\nRunning check_fit...")
try:
    calib.check_fit()
    print("✓ check_fit completed successfully")
except Exception as e:
    print(f"✗ check_fit failed with error: {e}")
    import traceback
    traceback.print_exc()
    raise

print("\n" + "="*60)
print("Checking results...")
print("="*60)

print(f"\nBefore age distribution shape: {calib.before_age_distribution.shape if hasattr(calib, 'before_age_distribution') else 'N/A'}")
print(f"After age distribution shape: {calib.after_age_distribution.shape if hasattr(calib, 'after_age_distribution') else 'N/A'}")
print(f"Target age distribution shape: {age_distribution.shape}")

if hasattr(calib, 'before_age_distribution'):
    print("\nBefore age distribution:")
    print(calib.before_age_distribution)

if hasattr(calib, 'after_age_distribution'):
    print("\nAfter age distribution:")
    print(calib.after_age_distribution)

print("\n✓ Debug test complete!")
