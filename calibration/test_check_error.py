"""
Test to see what specific error occurs
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import traceback

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("Creating small test to identify the error...")

# Very small sim for quick testing
sim = rs.Sim(
    n_agents=500,
    start='2000-01-01',
    stop='2002-01-01',  # Only 2 years - will likely have empty data
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
    demographics=[
        ss.Births(birth_rate=ss.peryear(25)),
        ss.Deaths(death_rate=ss.peryear(10)),
    ],
)

data = process_incidence.process_data()

calib_pars = sc.objdict(
    rel_beta=[1.0, 0.8, 1.2],
    reassortment_rate=[0.05, 0.03, 0.10]
)

calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=1,  # Just 1 trial
    debug=True,
)

print("\nRunning 1 trial...")
try:
    calib.calibrate()
    print("✓ Calibration succeeded")
except Exception as e:
    print(f"✗ Calibration failed: {e}")
    traceback.print_exc()

print("\nTrying check_fit()...")
try:
    calib.check_fit()
    print("✓ check_fit() succeeded")
except Exception as e:
    print(f"✗ check_fit() failed: {e}")
    traceback.print_exc()

print("\nTrying plot_sims()...")
try:
    calib.plot_sims()
    print("✓ plot_sims() succeeded")
except Exception as e:
    print(f"✗ plot_sims() failed: {e}")
    traceback.print_exc()
