"""
Quick test to verify plotting works
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import matplotlib.pyplot as plt

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("Testing plot generation...")

# Very small/fast calibration
sim = rs.Sim(
    n_agents=2000,
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
    rel_beta=[1.0, 0.8, 1.2],
    reassortment_rate=[0.10, 0.05, 0.15]
)

calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=3,
    debug=True,
)

print("\nRunning calibration...")
calib.calibrate()
calib.check_fit()

print("\nGenerating plots...")
fig1 = calib.plot_sims()
fig1.savefig('test_calibration_fit.png', dpi=150, bbox_inches='tight')
print('✓ Saved: test_calibration_fit.png')

fig2 = calib.plot_trend()
fig2.savefig('test_calibration_trend.png', dpi=150, bbox_inches='tight')
print('✓ Saved: test_calibration_trend.png')

print("\n✓ All plots generated successfully!")
