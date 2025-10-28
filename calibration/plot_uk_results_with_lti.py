"""
Generate plots for UK calibration results with long-term immunity:
1. Age distribution of long-term immunity development
2. Incidence over time (overall and by age)
"""

import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

thisdir = sc.thispath(__file__)
process_incidence_uk = sc.importbypath(thisdir / 'process_incidence_uk.py')

print("="*60)
print("Generating UK Calibration Plots with Long-Term Immunity")
print("="*60)

# Load best parameters
try:
    best_pars = sc.load(thisdir / 'uk_best_pars_with_lti.obj')
    print("\nLoaded best parameters:")
    for par, val in best_pars.items():
        print(f"  {par}: {val:.6f}")
except:
    print("\nError: Could not load best parameters!")
    print("Please run calibrate_uk_with_lti.py first.")
    exit(1)

# Create simulation with best parameters
print("\n" + "="*60)
print("Running simulation with best parameters...")
print("="*60)

sim = rs.Sim(
    n_agents=10000,  # Larger population for better statistics
    start='2003-01-01',  # 5 year burn-in
    stop='2013-01-01',
    verbose=True,
    scenario='baseline',
    base_beta=best_pars.get('base_beta', 0.16),
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Update immunity parameters
immunity = sim.connectors['rotaimmunityconnector']
immunity.pars.homotypic_immunity_efficacy = best_pars.get('homotypic_immunity_efficacy', 0.5)
immunity.pars.partial_heterotypic_immunity_efficacy = best_pars.get('partial_heterotypic_immunity_efficacy', 0.2)
immunity.pars.complete_heterotypic_immunity_efficacy = best_pars.get('complete_heterotypic_immunity_efficacy', 0.1)
immunity.pars.maternal_immunity_efficacy = best_pars.get('maternal_immunity_efficacy', 0.0)
immunity.pars.long_term_immunity_prob_after_1 = best_pars.get('long_term_immunity_prob_after_1', 0.39)
immunity.pars.long_term_immunity_prob_after_2 = best_pars.get('long_term_immunity_prob_after_2', 0.52)
immunity.pars.long_term_immunity_prob_after_3 = best_pars.get('long_term_immunity_prob_after_3', 0.67)

# Update reporting rate for all diseases
reporting_rate = best_pars.get('reporting_rate', 0.0002)
for disease in sim.diseases.values():
    if isinstance(disease, rs.Rotavirus):
        disease.pars.reporting_rate = reporting_rate

sim.run()

print("\nSimulation complete!")

# ============================================================================
# PLOT 1: Long-Term Immunity Age Distribution
# ============================================================================

print("\n" + "="*60)
print("Generating Plot 1: Long-Term Immunity Age Distribution")
print("="*60)

# Get long-term immunity data
alive = sim.people.alive.values
long_term_immune = immunity.long_term_immune.values[alive]
lti_ages = immunity.long_term_immune_age.values[alive]

n_lti = np.sum(long_term_immune)
print(f"\nLong-term immune agents: {n_lti:,} ({n_lti/np.sum(alive)*100:.1f}%)")

if n_lti > 0:
    lti_age_values = lti_ages[long_term_immune]
    mean_age = np.nanmean(lti_age_values)
    median_age = np.nanmedian(lti_age_values)

    print(f"Mean age at LTI development: {mean_age:.2f} years")
    print(f"Median age at LTI development: {median_age:.2f} years")

    # Create histogram
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(lti_age_values, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
    ax.axvline(mean_age, color='red', linestyle='--', linewidth=2, label=f'Mean: {mean_age:.1f} years')
    ax.axvline(median_age, color='orange', linestyle='--', linewidth=2, label=f'Median: {median_age:.1f} years')
    ax.set_xlabel('Age (years)', fontsize=12)
    ax.set_ylabel('Number of agents', fontsize=12)
    ax.set_title('Age Distribution of Long-Term Immunity Development\n(UK Calibration, 2003-2012)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(thisdir / 'uk_lti_age_distribution.png', dpi=300)
    print(f"\nSaved: {thisdir / 'uk_lti_age_distribution.png'}")
    plt.close()

# ============================================================================
# PLOT 2: Incidence Over Time (Overall and By Age)
# ============================================================================

print("\n" + "="*60)
print("Generating Plot 2: Incidence Over Time")
print("="*60)

# Load analyzer results
analyzer = sim.analyzers[0]
results_df = analyzer.results

# Filter to data period (2008-2012, which is years 5-10 in simulation time)
# Simulation starts on 2003-01-01, so year 5 = 2008-01-01
results_df['Year'] = np.floor(results_df['CollectionTime']).astype(int)
data_period = results_df[(results_df['CollectionTime'] >= 5) & (results_df['CollectionTime'] < 10)]

print(f"Total infections in data period: {len(data_period):,}")

# Map ages to categories
data_period['AgeCat'] = 'Other'
data_period.loc[data_period['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
data_period.loc[data_period['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
data_period.loc[data_period['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
data_period.loc[data_period['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Count first 3 infections per person (symptomatic)
data_period = data_period.sort_values(['id', 'CollectionTime'])
data_period['infection_number'] = data_period.groupby('id').cumcount() + 1
data_symptomatic = data_period[data_period['infection_number'] <= 3].copy()

# Get first infection per person per year per age
data_symptomatic['YearFloat'] = data_symptomatic['CollectionTime']
data_first = data_symptomatic.groupby(['id', 'Year', 'AgeCat']).first().reset_index()

# Calculate population sizes by year and age
# Use all agents (not just infected) to get true population
all_data = results_df[(results_df['CollectionTime'] >= 5) & (results_df['CollectionTime'] < 10)].copy()
all_data['Year'] = np.floor(all_data['CollectionTime']).astype(int)
all_data['AgeCat'] = 'Other'
all_data.loc[all_data['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
all_data.loc[all_data['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
all_data.loc[all_data['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
all_data.loc[all_data['Age'] == '60+', 'AgeCat'] = '>=5 y'

# Take last event per year per person to get population snapshot
pop_snapshots = all_data.sort_values('CollectionTime').groupby(['Year', 'id']).tail(1)
pop_counts = pop_snapshots.groupby(['Year', 'AgeCat']).agg(
    Pop_Age=('id', 'nunique')
).reset_index()

# Count cases by year and age
cases_by_year_age = data_first.groupby(['Year', 'AgeCat']).agg(
    Cases=('id', 'nunique')
).reset_index()

# Merge and calculate incidence rates
incidence_data = pd.merge(cases_by_year_age, pop_counts, on=['Year', 'AgeCat'], how='outer').fillna(0)
incidence_data['Incidence_per_100k'] = (incidence_data['Cases'] / incidence_data['Pop_Age']) * 100000

# Calculate overall incidence (all ages combined)
overall_by_year = data_first.groupby('Year').agg(Cases=('id', 'nunique')).reset_index()
total_pop_by_year = pop_snapshots.groupby('Year').agg(TotalPop=('id', 'nunique')).reset_index()
overall_incidence_data = pd.merge(overall_by_year, total_pop_by_year, on='Year')
overall_incidence_data['Incidence_per_100k'] = (overall_incidence_data['Cases'] / overall_incidence_data['TotalPop']) * 100000

# Convert year to actual calendar year
incidence_data['CalendarYear'] = incidence_data['Year'] + 2003
overall_incidence_data['CalendarYear'] = overall_incidence_data['Year'] + 2003

# Create plots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))

# Plot 1: Overall Incidence Over Time
ax1.plot(overall_incidence_data['CalendarYear'], overall_incidence_data['Incidence_per_100k'],
         marker='o', linewidth=2, markersize=8, color='darkblue', label='Overall')
ax1.set_xlabel('Year', fontsize=12)
ax1.set_ylabel('Incidence per 100,000', fontsize=12)
ax1.set_title('Overall Rotavirus Incidence Over Time\n(UK, 2008-2012)', fontsize=14, fontweight='bold')
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=10)
ax1.set_xticks(overall_incidence_data['CalendarYear'])

# Plot 2: Incidence by Age Group
age_order = ['<1 y', '1-2 y', '2-5 y', '>=5 y']
colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']

for age_cat, color in zip(age_order, colors):
    age_data = incidence_data[incidence_data['AgeCat'] == age_cat]
    if len(age_data) > 0:
        ax2.plot(age_data['CalendarYear'], age_data['Incidence_per_100k'],
                marker='o', linewidth=2, markersize=6, color=color, label=age_cat)

ax2.set_xlabel('Year', fontsize=12)
ax2.set_ylabel('Incidence per 100,000', fontsize=12)
ax2.set_title('Rotavirus Incidence by Age Group Over Time\n(UK, 2008-2012)', fontsize=14, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.legend(fontsize=10, title='Age Group')
ax2.set_xticks(age_data['CalendarYear'].unique())

plt.tight_layout()
plt.savefig(thisdir / 'uk_incidence_over_time.png', dpi=300)
print(f"\nSaved: {thisdir / 'uk_incidence_over_time.png'}")
plt.close()

print("\n" + "="*60)
print("Summary Statistics")
print("="*60)

print(f"\nMean overall incidence (2008-2012): {overall_incidence_data['Incidence_per_100k'].mean():.1f} per 100k")
print("\nMean incidence by age group:")
for age_cat in age_order:
    age_data = incidence_data[incidence_data['AgeCat'] == age_cat]
    if len(age_data) > 0:
        mean_inc = age_data['Incidence_per_100k'].mean()
        print(f"  {age_cat}: {mean_inc:.1f} per 100k")

print("\n" + "="*60)
print("All plots generated successfully!")
print("="*60)
