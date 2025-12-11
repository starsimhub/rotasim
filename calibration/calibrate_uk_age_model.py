"""
Calibration script for UK data with age-based symptom model

Supports three symptom models:
1. 'infection_number': Original (first 3 infections symptomatic)
2. 'age_only': P(symptomatic) = logistic(beta0 + beta1*age + beta2*age^2)
3. 'age_and_infection': Combined age and infection number effects

Age is capped at 5 years for symptom calculation.
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
# CONFIGURATION: Choose symptom model to calibrate
# ============================================================
# Options:
#   'infection_number' - Infection history affects immunity AND severity
#   'age_and_infection' - Infection history affects immunity AND severity, age affects symptoms
#   'age_and_infection_simple' - Infection history affects immunity only, age affects symptoms
SYMPTOM_MODEL = 'age_and_infection'

print("="*60)
print(f"UK Calibration with {SYMPTOM_MODEL.upper()} symptom model")
print("="*60)
print("\nUK Demographics:")
print("  Birth rate: 13/1000")
print("  Death rate: 6/1000")
print("  Target age distribution: [1.26%, 1.27%, 3.66%, 93.81%]")
print("  Follow-up: 5 years (2008-2012)")
print(f"\nSymptom model: {SYMPTOM_MODEL}")
if SYMPTOM_MODEL != 'infection_number':
    print("  Age in months, centered at 12 months (following Lewnard et al 2019)")
    print("  Age capped at 60 months (5 years) for symptom calculation")
print("="*60)


def calculate_reported_cases(df, reporting_rate):
    """
    Calculate reported cases using severity-based reporting

    P(reported | symptomatic) = reporting_rate * severity
    Note: 'symptomatic' determination now comes from age-based model
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


# Create analyzer with appropriate severity settings based on model
if SYMPTOM_MODEL in ['infection_number', 'age_and_infection']:
    # Use infection-based severity
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)
elif SYMPTOM_MODEL == 'age_and_infection_simple':
    # Use constant severity (age affects symptoms, not severity)
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.05)
else:
    raise ValueError(f"Unknown symptom model: {SYMPTOM_MODEL}")

# Create sim with UK demographics
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
)

# Get target data
overall_incidence, age_distribution = process_incidence_uk_age.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)

# ============================================================
# Calibration parameters based on symptom model
# ============================================================
if SYMPTOM_MODEL == 'infection_number':
    # Original model: only calibrate reporting_rate and base_beta
    # Infection history affects immunity AND severity
    # Previous best: base_beta=0.79, reporting_rate=0.0016 → 5.03 per 100k
    # Target: 27.56 per 100k (need 5.5x increase)
    calib_pars = sc.objdict(
        reporting_rate=[0.005, 0.0001, 0.02],  # Slightly higher to compensate
        base_beta=[2.0, 1.0, 4.0],             # 4x increase in range
    )

elif SYMPTOM_MODEL == 'age_and_infection':
    # Full model: age affects symptoms, infection affects immunity AND severity
    # Previous best: base_beta=0.84, betas=(-0.25, -0.009, -0.49), reporting=0.0027 → 3.41 per 100k
    # Target: 27.56 per 100k (need 8x increase)
    calib_pars = sc.objdict(
        beta0=[0, -3, 2],                # Allow higher symptom probabilities (was -2 to 1)
        beta1=[-0.1, -0.5, 0.2],         # Wider age effect range (was -0.5 to 0.1)
        beta2=[-0.01, -0.05, 0.01],      # Keep quadratic term range
        reporting_rate=[0.005, 0.0001, 0.02],
        base_beta=[2.5, 1.5, 5.0],       # 5x increase in range
    )

elif SYMPTOM_MODEL == 'age_and_infection_simple':
    # Simple model: age affects symptoms, infection affects immunity only (constant severity)
    # Previous best: base_beta=0.6, betas=(-2.0, -0.3, -0.01), reporting=0.01 → 2.79 per 100k
    # Target: 27.56 per 100k (need 10x increase)
    calib_pars = sc.objdict(
        beta0=[0.5, -2, 3],              # Much higher to allow more symptoms (was -5 to 1)
        beta1=[-0.1, -0.5, 0.3],         # Wider range, allow positive (was -0.5 to 0.1)
        beta2=[-0.01, -0.05, 0.02],      # Allow small positive quadratic (was -0.05 to 0.01)
        reporting_rate=[0.01, 0.001, 0.03],
        base_beta=[3.0, 2.0, 6.0],       # 6x increase in range
    )

else:
    raise ValueError(f"Unknown symptom model: {SYMPTOM_MODEL}")

print("\nCalibration parameters:")
for par, vals in calib_pars.items():
    print(f"  {par}: best={vals[0]}, range=[{vals[1]}, {vals[2]}]")

print("\n" + "="*60)
print("Running calibration (20 trials)...")
print("="*60)


# Create custom calibration class for age-based symptom model
class UKAgeCalibration(Calibration):
    """Custom calibration that uses age-based symptom model"""

    def __init__(self, *args, symptom_model='infection_number', **kwargs):
        super().__init__(*args, **kwargs)
        self.symptom_model = symptom_model

        # Add age model parameters to known_pars
        self.known_pars.extend(['beta0', 'beta1', 'beta2', 'beta3'])

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

        Extracts beta parameters from sim and passes to processing function
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

        # Check if severity column exists (required for severity-based reporting)
        if 'severity' not in df.columns:
            raise ValueError("'severity' column not found in infection data")

        # Get symptom model and beta parameters from sim
        symptom_model = getattr(sim, '_symptom_model', 'infection_number')
        beta0 = getattr(sim, '_beta0', 0)
        beta1 = getattr(sim, '_beta1', 0)
        beta2 = getattr(sim, '_beta2', 0)
        beta3 = getattr(sim, '_beta3', 0)

        # Get reporting rate if specified
        reporting_rate = None
        if hasattr(sim, '_reporting_rate'):
            reporting_rate = sim._reporting_rate

        # For age-based models, pass reporting_rate to process_model to apply AFTER age filtering
        # For infection_number model, apply reporting filter here BEFORE process_model
        if symptom_model == 'infection_number':
            # Apply severity-based reporting filter for infection_number model
            if reporting_rate is not None:
                df = calculate_reported_cases(df, reporting_rate)
            reporting_rate_to_pass = None  # Already filtered, don't apply again
        else:
            # For age-based models, pass reporting_rate to process_model
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
        """Override to handle age model parameters"""
        sim_pars = sc.mergedicts(sim_pars)

        # Extract and store symptom model type
        symptom_model = self.symptom_model

        # Extract age model parameters before parent translate_pars
        beta0 = sim_pars.pop('beta0', 0)
        beta1 = sim_pars.pop('beta1', 0)
        beta2 = sim_pars.pop('beta2', 0)
        beta3 = sim_pars.pop('beta3', 0)

        # Call parent translate_pars for standard parameters
        sim = super().translate_pars(sim_pars)

        # Store symptom model parameters on sim for use in sim_to_df
        sim._symptom_model = symptom_model
        sim._beta0 = beta0
        sim._beta1 = beta1
        sim._beta2 = beta2
        sim._beta3 = beta3

        return sim


if __name__ == '__main__':
    # Run calibration
    calib = UKAgeCalibration(
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

    print("\nBest parameters:")
    for par, val in calib.best_pars.items():
        print(f"  {par}: {val:.6f}")

    print("\n" + "="*60)
    print("Comparing Overall Incidence:")
    print("="*60)
    print(f"Target:  {overall_incidence:.1f} per 100k")
    print(f"Before:  {calib.before_overall_incidence:.1f} per 100k")
    print(f"After:   {calib.after_overall_incidence:.1f} per 100k")
    err_before_inci = (calib.before_overall_incidence - overall_incidence) / overall_incidence * 100
    err_after_inci = (calib.after_overall_incidence - overall_incidence) / overall_incidence * 100
    print(f"\nError before: {err_before_inci:+.1f}%")
    print(f"Error after:  {err_after_inci:+.1f}%")

    print("\n" + "="*60)
    print("Comparing Age Distribution (proportions):")
    print("="*60)
    print(f"\n{'Age':<10} {'Target':<15} {'Before':<15} {'After':<15} {'Error Before':<20} {'Error After':<20}")
    print("-"*100)

    age_labels_map = {0: '0-11mo', 1: '12-23mo', 2: '24-59mo', 5: '5+yr'}
    for i in range(len(age_distribution)):
        if i < len(calib.after_age_distribution):
            age = age_distribution.ages.iloc[i]
            target_prop = age_distribution.proportion.iloc[i] * 100
            before_prop = calib.before_age_distribution.proportion.iloc[i] * 100
            after_prop = calib.after_age_distribution.proportion.iloc[i] * 100

            err_before = before_prop - target_prop
            err_after = after_prop - target_prop

            age_label = age_labels_map.get(age, f'{age}yr')
            print(f"{age_label:<10} {target_prop:<15.1f}% {before_prop:<15.1f}% {after_prop:<15.1f}% {err_before:<20.1f}pp {err_after:<20.1f}pp")

    print("\n" + "="*60)
    print("Summary:")
    print("="*60)

    print(f"\nOverall Incidence:")
    print(f"  Target:  {overall_incidence:.1f} per 100k")
    print(f"  Before:  {calib.before_overall_incidence:.1f} per 100k ({err_before_inci:+.1f}%)")
    print(f"  After:   {calib.after_overall_incidence:.1f} per 100k ({err_after_inci:+.1f}%)")

    improvement_inci = abs(err_before_inci) - abs(err_after_inci)
    print(f"  Improvement: {improvement_inci:.1f} percentage points")

    print(f"\nAge Distribution GOF:")
    print(f"  Before: {calib.before_age_gof:.4f}")
    print(f"  After:  {calib.after_age_gof:.4f}")
    improvement_age = calib.before_age_gof - calib.after_age_gof
    print(f"  Improvement: {improvement_age:.4f}")

    if abs(err_after_inci) < 20 and calib.after_age_gof < 0.5:
        print("\n✓ Excellent fit: Both incidence and age distribution match well!")
    elif abs(err_after_inci) < 50 and calib.after_age_gof < 1.0:
        print("\n✓ Good fit: Both metrics improved")
    else:
        print("\n⚠ Model fit could be improved further")

    print(f"\n✓ UK calibration complete using {SYMPTOM_MODEL} model!")

    # Create figure summarizing goodness of fit
    print("\n" + "="*60)
    print("Creating goodness of fit figure...")
    print("="*60)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left panel: Age distribution comparison
    ax1 = axes[0]
    x_positions = np.arange(len(age_distribution))

    target_props = age_distribution.proportion.values * 100
    before_props = calib.before_age_distribution.proportion.values * 100
    after_props = calib.after_age_distribution.proportion.values * 100

    width = 0.25
    ax1.bar(x_positions - width, target_props, width, label='Target', color='black', alpha=0.7)
    ax1.bar(x_positions, before_props, width, label='Before', color='lightcoral', alpha=0.7)
    ax1.bar(x_positions + width, after_props, width, label='After', color='steelblue', alpha=0.7)

    ax1.set_xlabel('Age Group', fontsize=12)
    ax1.set_ylabel('Proportion (%)', fontsize=12)
    ax1.set_title('Age Distribution of Cases', fontsize=14, fontweight='bold')
    ax1.set_xticks(x_positions)
    ax1.set_xticklabels([age_labels_map[age] for age in age_distribution.ages.values])
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)

    # Add GOF text
    ax1.text(0.02, 0.98, f'GOF Before: {calib.before_age_gof:.3f}\nGOF After: {calib.after_age_gof:.3f}',
             transform=ax1.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # Right panel: Overall incidence comparison
    ax2 = axes[1]
    categories = ['Target', 'Before\nCalibration', 'After\nCalibration']
    incidences = [overall_incidence, calib.before_overall_incidence, calib.after_overall_incidence]
    colors = ['black', 'lightcoral', 'steelblue']

    bars = ax2.bar(categories, incidences, color=colors, alpha=0.7)
    ax2.set_ylabel('Incidence (per 100k)', fontsize=12)
    ax2.set_title('Overall Incidence', fontsize=14, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels on bars
    for bar, val in zip(bars, incidences):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.1f}',
                ha='center', va='bottom', fontsize=10)

    # Add error percentage text
    ax2.text(0.02, 0.98, f'Error Before: {err_before_inci:+.1f}%\nError After: {err_after_inci:+.1f}%',
             transform=ax2.transAxes, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.suptitle(f'UK Calibration: {SYMPTOM_MODEL.replace("_", " ").title()} Model',
                 fontsize=16, fontweight='bold', y=1.00)
    plt.tight_layout()

    # Save figure
    fig_path = thisdir / f'uk_calibration_fit_{SYMPTOM_MODEL}.png'
    plt.savefig(fig_path, dpi=150, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {fig_path}")
    plt.close()

    # Save results to JSON
    results_dict = {
        'symptom_model': SYMPTOM_MODEL,
        'best_parameters': {k: float(v) for k, v in calib.best_pars.items()},
        'incidence': {
            'target': float(overall_incidence),
            'before': float(calib.before_overall_incidence),
            'after': float(calib.after_overall_incidence),
        },
        'gof': {
            'before_total': float(calib.before_fit),
            'after_total': float(calib.after_fit),
            'before_age': float(calib.before_age_gof),
            'after_age': float(calib.after_age_gof),
        }
    }

    results_file = thisdir / f'uk_calibration_results_{SYMPTOM_MODEL}.json'
    sc.savejson(results_file, results_dict, indent=2)
    print(f"✓ Results saved to: {results_file}")

    print("\n" + "="*60)
