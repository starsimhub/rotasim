"""
Test calibration with reporting rate parameter

Strategy:
- Use full transmission (rel_beta=1.0) to maintain endemic circulation
- Calibrate reporting_rate to match surveillance data
- This represents the fact that only a small fraction of infections are reported
"""
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*80)
print("Testing Calibration with Reporting Rate")
print("="*80)
print("\nStrategy:")
print("  - Keep transmission at endemic levels (rel_beta=1.0)")
print("  - Calibrate reporting_rate (surveillance capture rate)")
print("  - True incidence ~10% in children, reported ~0.2%")
print("  - Expected reporting_rate ~0.002-0.02 (0.2-2%)")
print("="*80)

# Create sim with 10,000 agents for stability
sim = rs.Sim(
    n_agents=10000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    base_beta=0.16,  # From simple.py - maintains endemic circulation
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),  # From simple.py
    demographics=[
        ss.Births(birth_rate=ss.peryear(70)),  # From simple.py
        ss.Deaths(death_rate=ss.peryear(20)),  # From simple.py
    ],
)

data = process_incidence.process_data()
print("\nCalibration target data:")
print(data)

# Calibration parameters: reporting_rate only
# Based on model producing ~100k per 100k per year vs target 188:
# reporting_rate should be around 188/100000 = 0.002 (0.2%)
calib_pars = sc.objdict(
    reporting_rate=[0.002, 0.0005, 0.01],  # Start at 0.2%, search 0.05%-1%
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*80)
print("Running calibration (10 trials)...")
print("="*80)

calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=10,
    debug=True,
)

calib.calibrate()

print("\n" + "="*80)
print("Checking fit...")
print("="*80)
calib.check_fit()

print("\n" + "="*80)
print("Results:")
print("="*80)

print("\nBest parameters:")
for par, val in calib.best_pars.items():
    print(f"  {par}: {val:.6f} ({val*100:.4f}%)")

print("\n" + "="*80)
print("Comparing actual vs predicted incidence:")
print("="*80)
print(f"\n{'Age':<10} {'Actual':<15} {'Before':<15} {'After':<15} {'% Error Before':<20} {'% Error After':<20}")
print("-"*100)

for i in range(len(data)):
    if i < len(calib.after_df):
        age = data.ages.iloc[i]
        actual = data.inci.iloc[i]
        before = calib.before_df.inci.iloc[i] if i < len(calib.before_df) else 0
        after = calib.after_df.inci.iloc[i]

        err_before = ((before - actual) / actual * 100) if actual > 0 else 0
        err_after = ((after - actual) / actual * 100) if actual > 0 else 0

        print(f"{age:<10} {actual:<15.1f} {before:<15.1f} {after:<15.1f} {err_before:<20.1f} {err_after:<20.1f}")

# Overall assessment
if len(calib.after_df) > 0:
    avg_actual = data.inci.mean()
    avg_before = calib.before_df.inci.mean() if len(calib.before_df) > 0 else 0
    avg_after = calib.after_df.inci.mean()

    err_before_pct = ((avg_before - avg_actual) / avg_actual * 100) if avg_actual > 0 else 0
    err_after_pct = ((avg_after - avg_actual) / avg_actual * 100) if avg_actual > 0 else 0

    print("\n" + "="*80)
    print("Summary:")
    print("="*80)
    print(f"Average incidence:")
    print(f"  Target:  {avg_actual:.1f}")
    print(f"  Before:  {avg_before:.1f} ({err_before_pct:+.1f}%)")
    print(f"  After:   {avg_after:.1f} ({err_after_pct:+.1f}%)")

    improvement = abs(err_before_pct) - abs(err_after_pct)
    print(f"\nImprovement: {improvement:.1f} percentage points")

    if abs(err_after_pct) < 20:
        print("✓✓✓ EXCELLENT! Within 20% of target!")
    elif abs(err_after_pct) < 100:
        print("✓✓ GOOD! Within 2x of target")
    else:
        print("✓ Fair - needs more tuning")

print("\n✓ Test complete!")
