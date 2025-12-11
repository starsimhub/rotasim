"""
Calibration script for age_and_infection_simple model WITH fitted severity parameter

This variant fits:
- beta0, beta1, beta2: Age-based symptom parameters
- reporting_rate: Reporting rate
- base_beta: Transmission rate
- constant_severity: Severity probability (instead of fixed 0.05)
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
import matplotlib.pyplot as plt

thisdir = sc.thispath(__file__)
from calibration import Calibration
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# ============================================================
# CONFIGURATION
# ============================================================
SYMPTOM_MODEL = 'age_and_infection_simple'

print("="*60)
print(f"UK Calibration with {SYMPTOM_MODEL.upper()} + FITTED SEVERITY")
print("="*60)
print("\nUK Demographics:")
print("  Birth rate: 13/1000")
print("  Death rate: 6/1000")
print("  Target age distribution: [1.26%, 1.27%, 3.66%, 93.81%]")
print("  Follow-up: 5 years (2008-2012)")
print(f"\nSymptom model: {SYMPTOM_MODEL}")
print("  Age in months, centered at 12 months (following Lewnard et al 2019)")
print("  Age capped at 60 months (5 years) for symptom calculation")
print("  **SEVERITY PARAMETER IS NOW FITTED**")
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


# ============================================================
# Custom calibration class
# ============================================================
class UKAgeCalibrationWithSeverity(Calibration):
    """Custom calibration that fits severity parameter"""

    def __init__(self, *args, symptom_model='age_and_infection_simple', **kwargs):
        super().__init__(*args, **kwargs)
        self.symptom_model = symptom_model

        # Add age model parameters and severity to known_pars
        self.known_pars.extend(['beta0', 'beta1', 'beta2', 'beta3', 'constant_severity'])

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """Override to initialize UK ages and adult immunity"""
        # First, translate parameters
        if calib_pars is not None:
            sim_pars = self.trial_to_sim_pars(calib_pars=calib_pars, trial=trial)
        print(f"Running trial with pars: {sim_pars}")

        # Update sim with new parameters (this already calls sim.init())
        sim = self.translate_pars(sim_pars=sim_pars)

        # Initialize exposure history for adults
        sim.connectors.rotaimmunityconnector.initialize_immunity(
            min_age=18, max_age=125, min_exposures=5, max_exposures=15
        )

        # Now run the full simulation
        sim.run()

        return sim

    @staticmethod
    def sim_to_df(sim):
        """
        Convert sim output using age-based symptom model
        """
        # Extract infection data from InfectedStrainStats analyzer
        infected_analyzer = None
        for analyzer in sim.analyzers.values():
            if type(analyzer).__name__ == 'InfectedStrainStats':
                infected_analyzer = analyzer
                break

        if infected_analyzer is None:
            raise ValueError("InfectedStrainStats analyzer not found")

        # Get the infection events dataframe
        df = infected_analyzer.to_df()

        # Check if severity column exists
        if 'severity' not in df.columns:
            raise ValueError("'severity' column not found in infection data")

        # Get symptom model and beta parameters from sim
        symptom_model = getattr(sim, '_symptom_model', 'age_and_infection_simple')
        beta0 = getattr(sim, '_beta0', 0)
        beta1 = getattr(sim, '_beta1', 0)
        beta2 = getattr(sim, '_beta2', 0)
        beta3 = getattr(sim, '_beta3', 0)

        # Get reporting rate
        reporting_rate = getattr(sim, '_reporting_rate', None)

        # Pass reporting_rate to process_model to apply AFTER age filtering
        reporting_rate_to_pass = reporting_rate

        # Extract actual age-specific population counts from sim
        age_counts = extract_age_specific_population_counts(sim)

        # Process using age-based symptom model
        overall_incidence, age_distribution = process_incidence_uk_age.process_model(
            df,
            age_counts=age_counts,
            symptom_model=symptom_model,
            beta0=beta0,
            beta1=beta1,
            beta2=beta2,
            beta3=beta3,
            reporting_rate=reporting_rate_to_pass
        )

        return overall_incidence, age_distribution

    def translate_pars(self, sim_pars):
        """Override to handle age model parameters and severity"""
        sim_pars = sc.mergedicts(sim_pars)

        # Extract symptom model type
        symptom_model = self.symptom_model

        # Extract age model parameters before parent translate_pars
        beta0 = sim_pars.pop('beta0', 0)
        beta1 = sim_pars.pop('beta1', 0)
        beta2 = sim_pars.pop('beta2', 0)
        beta3 = sim_pars.pop('beta3', 0)

        # Extract severity parameter
        constant_severity = sim_pars.pop('constant_severity', 0.05)

        # Call parent translate_pars for standard parameters
        sim = super().translate_pars(sim_pars)

        # Update the existing analyzer's severity parameter directly
        # Find the InfectedStrainStats analyzer
        for analyzer in sim.analyzers.values():
            if type(analyzer).__name__ == 'InfectedStrainStats':
                # Update the constant_severity attribute
                analyzer.constant_severity = constant_severity
                break

        # Store symptom model parameters on sim for use in sim_to_df
        sim._symptom_model = symptom_model
        sim._beta0 = beta0
        sim._beta1 = beta1
        sim._beta2 = beta2
        sim._beta3 = beta3
        sim._constant_severity = constant_severity

        return sim


# ============================================================
# Create base simulation
# ============================================================
# Create initial analyzer (will be replaced in translate_pars)
analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.05)

# Create sim with UK demographics
people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
sim = rs.Sim(
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
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk_age.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# ============================================================
# Calibration parameters
# ============================================================
# Start with parameters from previous age_and_infection_simple calibration
# but add severity as a fitted parameter
calib_pars = sc.objdict(
    beta0=[0.5, -2, 3],                     # Age effect (intercept)
    beta1=[-0.1, -0.5, 0.3],                # Age effect (linear)
    beta2=[-0.01, -0.05, 0.02],             # Age effect (quadratic)
    constant_severity=[0.05, 0.01, 0.15],   # NEW: Fit severity instead of fixing at 0.05
    reporting_rate=[0.01, 0.001, 0.03],     # Reporting rate
    base_beta=[3.0, 2.0, 6.0],              # Transmission rate
)

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (20 trials)...")
print("="*60)

# Run calibration
calib = UKAgeCalibrationWithSeverity(
    sim=sim,
    data=(overall_incidence, age_distribution),
    calib_pars=calib_pars,
    total_trials=20,
    debug=False,
    symptom_model=SYMPTOM_MODEL,
)

calib.calibrate()

print("\n" + "="*60)
print("Checking fit...")
print("="*60)
calib.check_fit()

print("\n" + "="*60)
print("Results:")
print("="*60)
print(f"Best parameters: {calib.best_pars}")
print(f"✓ UK calibration complete using {SYMPTOM_MODEL} model with fitted severity!")
print("="*60)

# Save results
results = {
    'symptom_model': f'{SYMPTOM_MODEL}_with_severity',
    'best_parameters': {k: float(v) for k, v in calib.best_pars.items()},
    'incidence': {
        'target': overall_incidence,
        'before': calib.before_overall_incidence,
        'after': calib.after_overall_incidence,
    },
    'gof': {
        'before_total': calib.before_total_gof,
        'after_total': calib.after_total_gof,
        'before_age': calib.before_age_gof,
        'after_age': calib.after_age_gof,
    }
}

import json
results_file = thisdir / 'uk_calibration_results_age_and_infection_simple_with_severity.json'
with open(results_file, 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n✓ Results saved to: {results_file}")
