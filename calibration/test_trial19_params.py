"""
Quick test of Trial 19 parameters from calibration
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
from calibrate_infection_number_fixed_suscept import UKAgeCalibrationFixedSus
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Trial 19 parameters (from calibration log)
TRIAL_19_PARAMS = {
    'reporting_rate': 0.0021258099445399365,
    'base_beta': 6.973398294883567
}

print("="*60)
print("Testing Trial 19 Parameters")
print("="*60)
print("\nParameters:")
for k, v in TRIAL_19_PARAMS.items():
    print(f"  {k}: {v:.6f}")

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Create analyzer
analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

# Create immunity connector with fixed susceptibility
immunity_connector = rs.RotaImmunityConnector(
    use_fixed_susceptibility=True,
    sus_after_1=0.67,
    sus_after_2=0.50,
    sus_after_3plus=0.36,
)

# Create sim
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
base_sim = rs.Sim(
    n_agents=100000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    people=people,
    analyzers=[analyzer],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[],
    connectors=[immunity_connector],
    rand_seed=12345,
)

# Create calibration object
calib = UKAgeCalibrationFixedSus(
    sim=base_sim,
    data=(target_incidence, target_age_distribution),
    calib_pars={},
    total_trials=1,
    debug=False,
)

# Run simulation with Trial 19 parameters
print("\nRunning simulation...")
sim = calib.run_sim(calib_pars=None, sim_pars=TRIAL_19_PARAMS, trial=None)

# Extract results
print("Extracting results...")
overall_incidence, age_distribution = calib.sim_to_df(sim)

# Calculate GOF
gof = calib.compute_gof(sim)

# Display results
print("\n" + "="*60)
print("RESULTS")
print("="*60)

print(f"\nOverall Incidence (per 100,000):")
print(f"  Target: {target_incidence:.2f}")
print(f"  Fitted: {overall_incidence:.2f}")
error = overall_incidence - target_incidence
error_pct = (error / target_incidence) * 100
print(f"  Error:  {error:+.2f} ({error_pct:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age Group':<15} {'Target':>10} {'Fitted':>10} {'Difference':>12}")
print("-"*80)
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    target_prop = target_age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution.proportion.iloc[i]
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

print(f"\nGoodness of Fit: {gof:.4f}")

print("\n" + "="*60)
print("Test complete!")
print("="*60)
