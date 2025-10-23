"""
Test full calibration with a small number of trials
"""
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*60)
print("Full Calibration Test (5 trials)")
print("="*60)

# Create base sim
print("\nCreating simulation...")
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

calib_pars = sc.objdict(
    rel_beta=[1.0, 0.5, 1.5],
    reassortment_rate=[0.10, 0.05, 0.15]
)

print("Creating calibration object...")
calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=5,
    debug=False,  # Run in parallel
)

print("\n" + "="*60)
print("Running calibration...")
print("="*60)
calib.calibrate()

print("\n" + "="*60)
print("Checking fit...")
print("="*60)
calib.check_fit()

print("\n" + "="*60)
print("Plotting results...")
print("="*60)
calib.plot_sims()

print("\n" + "="*60)
print("✓ All tasks completed successfully!")
print("="*60)
