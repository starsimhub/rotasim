"""
Minimal calibration test - just 2 trials to verify it works
"""
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("=" * 60)
print("Minimal Calibration Test")
print("=" * 60)

# Create base sim with demographics to sustain infections
print("\n1. Creating base simulation...")
sim = rs.Sim(
    n_agents=5000,  # Smaller for faster testing
    start='2000-01-01',
    stop='2010-01-01',  # 10 years
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
    demographics=[
        ss.Births(birth_rate=ss.peryear(25)),
        ss.Deaths(death_rate=ss.peryear(10)),
    ],
)
print("✓ Base sim created")

# Load calibration data
print("\n2. Loading calibration data...")
data = process_incidence.process_data()
print(f"✓ Loaded calibration data with {len(data)} age groups")
print(data)

# Define calibration parameters
print("\n3. Defining calibration parameters...")
calib_pars = sc.objdict(
    rel_beta=[1.0, 0.8, 1.2],  # [best, low, high]
    reassortment_rate=[0.05, 0.03, 0.10]
)
print("✓ Parameters defined:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

# Create calibration object
print("\n4. Creating calibration object...")
calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=2,  # Just 2 trials for testing
    debug=True,  # Run in serial for easier debugging
)
print("✓ Calibration object created")

# Run calibration
print("\n5. Running calibration (2 trials)...")
print("-" * 60)
try:
    calib.calibrate()
    print("-" * 60)
    print("✓ Calibration completed successfully!")
    print(f"\nBest parameters found:")
    for par, val in calib.best_pars.items():
        print(f"  {par}: {val:.4f}")

    # Check fit
    print("\n6. Checking fit...")
    before_fit, after_fit = calib.check_fit()
    print(f"  Fit before: {before_fit:.4f}")
    print(f"  Fit after:  {after_fit:.4f}")
    if after_fit <= before_fit:
        print("  ✓ Calibration improved fit!")
    else:
        print("  ⚠ Calibration did not improve fit (may need more trials)")

except Exception as e:
    print("-" * 60)
    print(f"✗ Calibration failed with error:")
    print(f"  {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

print("\n" + "=" * 60)
print("Test completed successfully!")
print("=" * 60)
