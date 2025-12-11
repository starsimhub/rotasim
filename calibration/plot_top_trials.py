"""
Compare top trials from hybrid calibration
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import matplotlib.pyplot as plt
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

# Load study
study = optuna.load_study(study_name='rota_hybrid', storage='sqlite:///rota_hybrid.db')

# Select trials to compare
trial_numbers = [41, 46, 49]
print("=" * 80)
print(f"Comparing trials: {trial_numbers}")
print("=" * 80)

# Function to run simulation with trial parameters
def run_simulation_with_params(params, trial_num):
    """Run simulation with specific parameters"""
    print(f"\nRunning simulation for Trial {trial_num}...")

    # Create analyzer
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)

    # Create immunity connector with default values (will be updated after init)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=False  # Will be set to True after init
    )

    # Create simulation
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
        connectors=[immunity_connector],
    )

    # Update base_beta BEFORE initialization
    sim.pars.base_beta = params['base_beta']
    for disease in sim.pars.diseases:
        if isinstance(disease, rs.Rotavirus):
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

    # Store parameters for use in data extraction
    sim._reporting_rate = params['reporting_rate']
    sim._beta0 = params['beta0']
    sim._beta1 = params['beta1']
    sim._beta2 = params['beta2']

    # Initialize
    sim.init()

    # Get immunity connector and update parameters AFTER initialization
    immunity_connector = sim.connectors.rotaimmunityconnector
    immunity_connector.pars['use_fixed_susceptibility'] = True
    immunity_connector.pars['sus_after_1'] = params['sus_after_1']
    immunity_connector.pars['sus_after_2'] = params['sus_after_2']
    immunity_connector.pars['sus_after_3plus'] = params['sus_after_3plus']

    # Initialize adult immunity
    immunity_connector.initialize_immunity(
        min_age=18, max_age=125, min_exposures=5, max_exposures=15
    )

    # Run
    sim.run()

    # Extract results
    df = sim.analyzers['infectedstrainstats'].to_df()

    # Filter to follow-up period
    df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

    # Extract age-specific population counts
    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }

    # Get stored parameters from sim
    reporting_rate = getattr(sim, '_reporting_rate', params['reporting_rate'])
    beta0 = getattr(sim, '_beta0', params['beta0'])
    beta1 = getattr(sim, '_beta1', params['beta1'])
    beta2 = getattr(sim, '_beta2', params['beta2'])

    # Process using age-based symptom model
    overall_incidence, age_distribution = process_incidence_uk_age.process_model(
        df,
        age_counts=age_counts,
        symptom_model='age_and_infection_simple',
        beta0=beta0,
        beta1=beta1,
        beta2=beta2,
        beta3=0,
        reporting_rate=reporting_rate
    )

    print(f"  Incidence: {overall_incidence:.2f} per 100k")
    print(f"  Age distribution: {age_distribution.proportion.values}")

    return overall_incidence, age_distribution


# Run simulations for selected trials
results = {}
for trial_num in trial_numbers:
    trial = study.trials[trial_num]
    results[trial_num] = {
        'params': trial.params,
        'gof': trial.value,
    }
    incidence, age_dist = run_simulation_with_params(trial.params, trial_num)
    results[trial_num]['incidence'] = incidence
    results[trial_num]['age_distribution'] = age_dist

# Create comparison plots
print("\n" + "=" * 80)
print("Creating comparison plots...")
print("=" * 80)

fig, axes = plt.subplots(1, 2, figsize=(16, 7))

# Panel 1: Age Distribution Comparison
ax1 = axes[0]
age_labels = ['0-11\nmonths', '12-23\nmonths', '24-59\nmonths', '5+\nyears']
x = np.arange(len(age_labels))
width = 0.2

target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(4)]

# Plot target
ax1.bar(x - 1.5*width, target_props, width, label='Target (UK Data)',
        color='black', alpha=0.7, edgecolor='black', linewidth=1.5)

# Plot each trial
colors = ['#E74C3C', '#3498DB', '#2ECC71']
for idx, (trial_num, color) in enumerate(zip(trial_numbers, colors)):
    fitted_props = [results[trial_num]['age_distribution'].proportion.iloc[i] * 100 for i in range(4)]
    gof = results[trial_num]['gof']
    ax1.bar(x + (idx - 0.5)*width, fitted_props, width,
           label=f'Trial {trial_num} (GOF={gof:.2f})',
           color=color, alpha=0.8, edgecolor='black', linewidth=1)

ax1.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
ax1.set_title('Age Distribution: Top 3 Trials vs Target', fontsize=13, fontweight='bold', pad=10)
ax1.set_xticks(x)
ax1.set_xticklabels(age_labels, fontsize=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.grid(axis='y', alpha=0.3)
ax1.set_ylim(0, max(target_props) * 1.3)

# Panel 2: Incidence Comparison
ax2 = axes[1]
categories = ['Target'] + [f'Trial {n}' for n in trial_numbers]
incidence_vals = [target_incidence] + [results[n]['incidence'] for n in trial_numbers]
colors_bar = ['black'] + colors

bars = ax2.bar(categories, incidence_vals, color=colors_bar, alpha=0.7,
              edgecolor='black', linewidth=2, width=0.6)

ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
ax2.set_title('Overall Incidence: Top 3 Trials vs Target', fontsize=13, fontweight='bold', pad=10)
ax2.set_ylim(0, max(incidence_vals) * 1.3)
ax2.axhline(y=target_incidence, color='black', linestyle='--', alpha=0.5, linewidth=2)
ax2.grid(axis='y', alpha=0.3)

# Add value labels and GOF
for i, (bar, val) in enumerate(zip(bars, incidence_vals)):
    height = bar.get_height()
    pct_of_target = (val / target_incidence) * 100
    if i == 0:
        label_text = f'{val:.1f}'
    else:
        trial_num = trial_numbers[i-1]
        gof = results[trial_num]['gof']
        label_text = f'{val:.1f}\n({pct_of_target:.0f}% of target)\nGOF: {gof:.2f}'
    ax2.text(bar.get_x() + bar.get_width()/2., height + 1,
             label_text,
             ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.suptitle('Hybrid Model: Top 3 Trials Comparison',
             fontsize=14, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])

# Save
output_file = thisdir / 'top_trials_comparison.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight')
print(f"\n✓ Saved: {output_file}")

output_pdf = thisdir / 'top_trials_comparison.pdf'
plt.savefig(output_pdf, bbox_inches='tight')
print(f"✓ Saved: {output_pdf}")

# Print summary table
print("\n" + "=" * 80)
print("SUMMARY TABLE")
print("=" * 80)
print(f"{'Trial':<8} {'GOF':<10} {'Incidence':<15} {'Age Distribution (0-1/1-2/2-5/5+)':<40}")
print("-" * 80)
print(f"{'Target':<8} {'-':<10} {target_incidence:<15.2f} {'/'.join([f'{target_age_distribution.proportion.iloc[i]*100:.1f}' for i in range(4)]):<40}")
for trial_num in trial_numbers:
    gof = results[trial_num]['gof']
    inc = results[trial_num]['incidence']
    age_props = '/'.join([f"{results[trial_num]['age_distribution'].proportion.iloc[i]*100:.1f}" for i in range(4)])
    print(f"{'#' + str(trial_num):<8} {gof:<10.2f} {inc:<15.2f} {age_props:<40}")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
