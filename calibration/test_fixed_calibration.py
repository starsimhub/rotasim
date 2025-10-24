"""
Test calibration with the fixed GOF function
"""
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*60)
print("Testing Calibration with Fixed GOF Function")
print("="*60)

# Create sim
sim = rs.Sim(
    n_agents=5000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
    demographics=[
        ss.Births(birth_rate=ss.peryear(25)),
        ss.Deaths(death_rate=ss.peryear(10)),
    ],
)

data = process_incidence.process_data()
print("\nCalibration target data:")
print(data)

# Calibration parameters - intentionally wide range to see if it finds the right direction
calib_pars = sc.objdict(
    rel_beta=[1.0, 0.3, 2.0],  # Wide range
    reassortment_rate=[0.10, 0.05, 0.15]
)

print("\nRunning calibration (5 trials)...")
calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=5,
    debug=True,
)

calib.calibrate()

print("\n" + "="*60)
print("Checking fit...")
print("="*60)
calib.check_fit()

print("\n" + "="*60)
print("Best parameters:")
print("="*60)
for par, val in calib.best_pars.items():
    print(f"  {par}: {val:.4f}")

print("\nExpected behavior:")
print("  If rel_beta < 1.0: Calibration found model was overestimating")
print("  If rel_beta > 1.0: Calibration found model was underestimating")

print("\n" + "="*60)
print("Comparing actual vs predicted incidence:")
print("="*60)
print(f"\n{'Age':<10} {'Actual':<15} {'Before':<15} {'After':<15}")
print("-"*60)
for i in range(len(data)):
    age = data.ages.iloc[i]
    actual = data.inci.iloc[i]
    before = calib.before_df.inci.iloc[i] if i < len(calib.before_df) else 0
    after = calib.after_df.inci.iloc[i] if i < len(calib.after_df) else 0
    print(f"{age:<10} {actual:<15.1f} {before:<15.1f} {after:<15.1f}")

# Check if calibration moved in the right direction
if len(calib.after_df) > 0:
    avg_actual = data.inci.mean()
    avg_before = calib.before_df.inci.mean()
    avg_after = calib.after_df.inci.mean()

    print(f"\nAverage incidence:")
    print(f"  Actual: {avg_actual:.1f}")
    print(f"  Before: {avg_before:.1f} ({'over' if avg_before > avg_actual else 'under'} by {abs(avg_before-avg_actual)/avg_actual*100:.1f}%)")
    print(f"  After:  {avg_after:.1f} ({'over' if avg_after > avg_actual else 'under'} by {abs(avg_after-avg_actual)/avg_actual*100:.1f}%)")

    if abs(avg_after - avg_actual) < abs(avg_before - avg_actual):
        print("\n✓ Calibration moved CLOSER to data!")
    else:
        print("\n⚠ Calibration moved AWAY from data")

print("\n✓ Test complete!")
