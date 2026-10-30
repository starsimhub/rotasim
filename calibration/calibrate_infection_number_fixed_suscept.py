"""
Calibration script for infection_number model with FIXED susceptibility values

Fixed susceptibility by infection count:
- After 1 infection: rel_sus = 0.67 (33% protection)
- After 2 infections: rel_sus = 0.50 (50% protection)
- After 3+ infections: rel_sus = 0.36 (64% protection)
"""

# Fix for PyCharm: Remove parent directory from sys.path
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("="*60)
print("UK Calibration: infection_number with FIXED susceptibility")
print("="*60)
print("\nFixed Susceptibility Model:")
print("  After 1 infection:  rel_sus = 0.67 (33% protection)")
print("  After 2 infections: rel_sus = 0.50 (50% protection)")
print("  After 3+ infections: rel_sus = 0.36 (64% protection)")
print("\nUK Demographics:")
print("  Birth rate: 13/1000")
print("  Death rate: 6/1000")
print("  Follow-up: 5 years (2008-2012)")
print("="*60)


def calculate_reported_cases(df, reporting_rate):
    """
    Calculate reported cases using severity-based reporting
    P(reported | symptomatic) = reporting_rate * severity
    """
    df['reported'] = np.random.random(len(df)) < (reporting_rate * df['severity'])
    reported_df = df[df['reported']].copy()
    return reported_df


def extract_age_specific_population_counts(sim):
    """Extract actual age-specific population counts from simulation"""
    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }
    return age_counts


# Create analyzer with infection-based severity
analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

# Create immunity connector with fixed susceptibility
immunity_connector = rs.RotaImmunityConnector(
    use_fixed_susceptibility=True,  # Enable fixed susceptibility mode
    sus_after_1=0.67,  # 33% protection
    sus_after_2=0.50,  # 50% protection
    sus_after_3plus=0.36,  # 64% protection
)

# Create sim with UK demographics and fixed susceptibility
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
sim = rs.Sim(
    n_agents=100000,
    start='2003-01-01',  # 5-year burn-in before 2008
    stop='2013-01-01',   # 10 years total (5 burn-in + 5 calibration)
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
    connectors=[immunity_connector],  # Use custom immunity connector with fixed susceptibility
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk_age.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# Calibration parameters - only calibrate reporting_rate and base_beta
# Higher reporting rate range to allow lower beta values and faster simulations
calib_pars = sc.objdict(
    reporting_rate=[0.015, 0.002, 0.05],  # Increased to allow lower beta
    base_beta=[5.0, 1.5, 10.0],  # Increased range due to stronger immunity
)

print("\nCalibration parameters:")
for k, v in calib_pars.items():
    print(f"  {k}: best={v[0]}, range=[{v[1]}, {v[2]}]")
print()
print("="*60)
print(f"Running calibration (20 trials)...")
print("="*60)


class UKAgeCalibrationFixedSus(Calibration):
    """Calibration class for infection_number model with fixed susceptibility"""

    def __init__(self, sim, data, calib_pars, total_trials, debug=False):
        """
        Args:
            sim: Base simulation to calibrate
            data: Tuple of (target_incidence, target_age_distribution)
            calib_pars: Dictionary of calibration parameters
            total_trials: Number of calibration trials
            debug: Enable debug output
        """
        super().__init__(sim, data, calib_pars=calib_pars, total_trials=total_trials)
        self.debug = debug

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """Run simulation with given parameters"""
        # Convert calib_pars to sim_pars if needed
        if calib_pars is not None and sim_pars is None:
            sim_pars = self.trial_to_sim_pars(calib_pars=calib_pars, trial=trial)

        if sim_pars is None:
            sim_pars = {}

        # Make a copy of base sim
        sim = sc.dcp(self.sim)

        # Apply parameters using the base class translate_pars method
        # But we need to handle base_beta specially since it needs to be set before init
        base_beta = sim_pars.pop('base_beta', None)
        if base_beta is not None:
            sim.pars.base_beta = base_beta
            for disease in sim.pars.diseases:
                if isinstance(disease, rs.Rotavirus):
                    disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

        # Set reporting_rate parameter (will be used in sim_to_df)
        if 'reporting_rate' in sim_pars:
            sim._reporting_rate = sim_pars['reporting_rate']

        # Initialize the simulation
        sim.init()

        # Initialize adult immunity
        sim.connectors.rotaimmunityconnector.initialize_immunity(
            min_age=18, max_age=125, min_exposures=5, max_exposures=15
        )

        # Run simulation
        sim.run()

        return sim

    def trial_to_sim_pars(self, calib_pars, trial):
        """Take in an optuna trial and sample from pars"""
        calib_pars = sc.mergedicts(calib_pars) # To allow None
        sim_pars = sc.objdict()
        for par, (best,low,high) in calib_pars.items():
            val = trial.suggest_float(par, low, high)
            sim_pars[par] = val
        return sim_pars

    def sim_to_df(self, sim):
        """Extract results from simulation"""
        # Access analyzer instance and call its to_df() method
        infected_analyzer = sim.analyzers['infectedstrainstats']
        df = infected_analyzer.to_df()

        # Filter to follow-up period (years 5-10 of simulation = 2008-2012)
        # CollectionTime is in years, not days
        follow_up_start_years = 5
        follow_up_end_years = 10
        df = df[(df['CollectionTime'] >= follow_up_start_years) & (df['CollectionTime'] < follow_up_end_years)]

        # Get reporting rate (stored as attribute by run_sim)
        reporting_rate = getattr(sim, '_reporting_rate', 0.005)  # Use default if not set

        # Calculate reported cases
        reported_df = calculate_reported_cases(df.copy(), reporting_rate)

        # Extract age-specific population counts
        age_counts = extract_age_specific_population_counts(sim)

        # Calculate age-specific incidence
        age_incidence = {}
        for ages_key in age_counts.keys():
            count = len(reported_df[reported_df['Age'] == ages_key])
            age_incidence[ages_key] = (count / age_counts[ages_key]) * 100000 / 5  # per 100k per year

        # Calculate overall incidence
        total_reported = len(reported_df)
        total_pop = sum(age_counts.values())
        overall_incidence = (total_reported / total_pop) * 100000 / 5  # per 100k per year

        # Calculate age distribution (proportions)
        age_dist = {}
        for ages_key in age_counts.keys():
            count = len(reported_df[reported_df['Age'] == ages_key])
            age_dist[ages_key] = count / total_reported if total_reported > 0 else 0

        # Convert to DataFrame matching target format
        import pandas as pd
        age_distribution = pd.DataFrame({
            'ages': [0, 1, 2, 5],
            'proportion': [
                age_dist['<1 y'],
                age_dist['1-2 y'],
                age_dist['2-5 y'],
                age_dist['>=5 y']
            ]
        }).set_index('ages')

        return overall_incidence, age_distribution

    def compute_gof(self, sim):
        """Compute goodness of fit"""
        overall_incidence, age_distribution = self.sim_to_df(sim)

        # Incidence GOF (relative absolute error)
        incidence_gof = abs(overall_incidence - self.overall_incidence) / self.overall_incidence

        # Age distribution GOF (sum of relative absolute errors across age groups)
        age_gof = ((age_distribution.proportion - self.age_distribution.proportion).abs() /
                   self.age_distribution.proportion).sum()

        # Total GOF
        total_gof = incidence_gof + age_gof

        if self.debug:
            print(f"  Incidence: {overall_incidence:.2f} vs {self.overall_incidence:.2f} (GOF: {incidence_gof:.4f})")
            print(f"  Age GOF: {age_gof:.4f}")
            print(f"  Total GOF: {total_gof:.4f}")

        return total_gof


# Create calibration object
calib = UKAgeCalibrationFixedSus(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=20,
    debug=False
)

# Remove existing calibration file if it exists
import os
db_path = 'rota.db'
if os.path.exists(db_path):
    os.remove(db_path)
    print(f"Removed existing calibration file {db_path}\n")

# Run calibration
calib.calibrate()

# Get best parameters
best_pars = calib.best_pars

print("\n" + "="*60)
print("CALIBRATION COMPLETE!")
print("="*60)
print("\nBest parameters:")
for k, v in best_pars.items():
    print(f"  {k}: {v:.6f}")

# Evaluate with best parameters
print("\n" + "="*60)
print("Evaluating best parameters...")
print("="*60)

best_sim = calib.run_sim(sim_pars=best_pars)
overall_incidence, age_distribution_result = calib.sim_to_df(best_sim)

print(f"\nOverall Incidence (per 100,000):")
print(f"  Target: {calib.overall_incidence:.2f}")
print(f"  Fitted: {overall_incidence:.2f}")
error = overall_incidence - calib.overall_incidence
error_pct = (error / calib.overall_incidence) * 100 if calib.overall_incidence > 0 else 0
print(f"  Error:  {error:+.2f} ({error_pct:+.1f}%)")

print(f"\nAge Distribution (proportions):")
print(f"{'Age Group':<15} {'Target':>10} {'Fitted':>10} {'Difference':>12}")
print("-"*80)
age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
for i, label in enumerate(age_labels):
    target_prop = age_distribution.proportion.iloc[i]
    fitted_prop = age_distribution_result.proportion.iloc[i]
    diff = fitted_prop - target_prop
    print(f"{label:<15} {target_prop*100:>9.2f}% {fitted_prop*100:>9.2f}% {diff*100:>+10.2f}%")

# Calculate GOF with best parameters
best_gof = calib.compute_gof(best_sim)

# Save results
results = {
    'model': 'infection_number_fixed_susceptibility',
    'susceptibility_model': {
        'type': 'fixed',
        'sus_after_1': 0.67,
        'sus_after_2': 0.50,
        'sus_after_3plus': 0.36
    },
    'best_parameters': best_pars,
    'target_incidence': float(calib.overall_incidence),
    'fitted_incidence': float(overall_incidence),
    'error_percent': float(error_pct),
    'gof': float(best_gof),
}

results_file = thisdir / 'uk_calibration_results_infection_number_fixed_sus.json'
sc.savejson(results_file, results, indent=2)
print(f"\n✓ Results saved to: {results_file}")

print("\n" + "="*60)
print("Calibration complete!")
print("="*60)
