"""
Verify the calibration results by running a simulation with best parameters
"""

import sciris as sc
import starsim as ss
import rotasim as rs
import optuna
import numpy as np

thisdir = sc.thispath(__file__)
import process_incidence_uk as process_uk

def initialize_uk_ages(sim):
    """Initialize population with UK age distribution (ages in years)"""
    n = len(sim.people)

    # UK age distribution
    age_bins = [
        (0, 1, 0.0126),    # <1 year: 1.26%
        (1, 2, 0.0127),    # 1-2 years: 1.27%
        (2, 5, 0.0366),    # 2-5 years: 3.66%
        (5, 80, 0.9381),   # 5+ years: 93.81%
    ]

    # Sample ages from distribution (in years)
    ages = []
    for low, high, prop in age_bins:
        n_in_bin = int(n * prop)
        if low >= 5:
            bin_ages = np.random.beta(2, 2, n_in_bin) * (high - low) + low
        else:
            bin_ages = np.random.uniform(low, high, n_in_bin)
        ages.extend(bin_ages)

    # Handle rounding
    while len(ages) < n:
        ages.append(np.random.beta(2, 2) * 75 + 5)

    ages = np.array(ages[:n])

    # Set ages (starsim uses years directly)
    sim.people.age[:] = ages

    if sim.pars.verbose:
        print(f"✓ Initialized UK age distribution:")
        print(f"  Age range: {ages.min():.1f} - {ages.max():.1f} years")
        print(f"  Mean age: {ages.mean():.1f} years")

def initialize_adult_immunity(sim, adult_baseline_immunity=0.95):
    """
    Initialize adults with baseline immunity from childhood exposures
    """
    ages = sim.people.age.values  # Already in years
    adult_mask = ages >= 5
    n_adults = adult_mask.sum()

    if n_adults == 0:
        return

    # Get the immunity connector
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is None:
        if sim.pars.verbose:
            print("  Warning: No RotaImmunityConnector found")
        return

    adult_uids = np.where(adult_mask)[0]

    # Record infection history
    immunity_connector.num_recovered_infections[adult_uids] = np.random.choice([2, 3], size=n_adults)
    immunity_connector.has_immunity[adult_uids] = True

    # Set bitmasks for prior exposure
    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            G = disease.G
            P = disease.P
            immunity_connector.exposed_G_bitmask[adult_uids] |= (1 << G)
            immunity_connector.exposed_P_bitmask[adult_uids] |= (1 << P)

    # Set permanent baseline immunity
    immunity_connector.baseline_immunity[adult_uids] = adult_baseline_immunity

    if sim.pars.verbose:
        first_disease = list(sim.diseases.values())[0]
        adult_rel_sus = first_disease.rel_sus[adult_uids]
        child_rel_sus = first_disease.rel_sus[~adult_mask]

        print(f"✓ Initialized {n_adults} adults with baseline immunity:")
        print(f"  Prior infections: 2-3 (typical childhood exposure)")
        print(f"  Cumulative protection: {adult_baseline_immunity*100:.1f}%")
        print(f"  Adult rel_sus: {adult_rel_sus.mean():.3f} (children: {child_rel_sus.mean():.1f})")

def seed_infections_by_age(sim, overall_prevalence=0.002):
    """
    Seed initial infections according to UK case age distribution
    """
    # Target age distribution for infections
    infection_age_dist = [
        (0, 1, 0.138),    # <1 year: 13.8%
        (1, 2, 0.277),    # 1-2 years: 27.7%
        (2, 5, 0.469),    # 2-5 years: 46.9%
        (5, 200, 0.116),  # ≥5 years: 11.6%
    ]

    n_infections = int(len(sim.people) * overall_prevalence)
    if n_infections == 0:
        return

    ages = sim.people.age.values  # Already in years

    # Find agents in each age category
    age_groups = []
    for low, high, target_prop in infection_age_dist:
        mask = (ages >= low) & (ages < high)
        agents_in_group = np.where(mask)[0]
        age_groups.append((low, high, target_prop, agents_in_group))

    # Allocate infections according to target proportions
    infected_agents = []
    for low, high, target_prop, agents_in_group in age_groups:
        n_to_infect = int(n_infections * target_prop)

        if len(agents_in_group) == 0:
            if sim.pars.verbose:
                print(f"  Warning: No agents in age group {low}-{high} years")
            continue

        n_available = len(agents_in_group)
        if n_to_infect > n_available:
            if sim.pars.verbose:
                print(f"  Warning: Only {n_available} agents available in {low}-{high}y (need {n_to_infect})")
            n_to_infect = n_available

        selected = np.random.choice(agents_in_group, size=n_to_infect, replace=False)
        infected_agents.extend(selected)

    # Remove duplicates and limit to target
    infected_agents = list(set(infected_agents))
    if len(infected_agents) > n_infections:
        infected_agents = np.random.choice(infected_agents, size=n_infections, replace=False)

    # Set infections for all rotavirus diseases
    for disease in sim.diseases.values():
        if hasattr(disease, 'G') and hasattr(disease, 'P'):
            disease.infected[:] = False
            disease.susceptible[:] = True
            disease.ti_infected[:] = np.nan

            disease.infected[infected_agents] = True
            disease.susceptible[infected_agents] = False
            disease.ti_infected[infected_agents] = sim.ti

    if sim.pars.verbose:
        print(f"✓ Seeded {len(infected_agents)} infections by age distribution")

def create_sim(sim_pars, seed=0):
    """
    Create a simulation with UK demographics and specified parameters
    """
    # Create base sim (ages in years, aging handled by starsim with births/deaths)
    sim = rs.Sim(
        n_agents=5000,
        start='2003-01-01',  # 5-year burn-in
        stop='2013-01-01',   # 10 years total
        verbose=True,
        rand_seed=seed,
        scenario='single',
        base_beta=sim_pars.get('base_beta', 0.16),
        override_prevalence=0.002,
        analyzers=[rs.InfectedStrainStats()],
        networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
        demographics=[
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
    )

    # Update immunity parameters
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            if 'homotypic_immunity_efficacy' in sim_pars:
                connector.homotypic_immunity_efficacy = sim_pars['homotypic_immunity_efficacy']
            if 'partial_heterotypic_immunity_efficacy' in sim_pars:
                connector.partial_heterotypic_immunity_efficacy = sim_pars['partial_heterotypic_immunity_efficacy']
            if 'complete_heterotypic_immunity_efficacy' in sim_pars:
                connector.complete_heterotypic_immunity_efficacy = sim_pars['complete_heterotypic_immunity_efficacy']
            if 'maternal_immunity_efficacy' in sim_pars:
                connector.maternal_immunity_efficacy = sim_pars['maternal_immunity_efficacy']

    # Store reporting rate
    if 'reporting_rate' in sim_pars:
        sim._reporting_rate = sim_pars['reporting_rate']

    # Initialize sim
    sim.initialize()

    # Apply UK-specific initialization
    initialize_uk_ages(sim)

    # Extract and apply adult baseline immunity
    adult_baseline_immunity = sim_pars.get('adult_baseline_immunity', 0.95)
    initialize_adult_immunity(sim, adult_baseline_immunity=adult_baseline_immunity)

    # Seed infections by age
    seed_infections_by_age(sim, overall_prevalence=0.002)

    return sim

# Load best trial from database
print('='*70)
print('UK CALIBRATION VERIFICATION')
print('='*70)

storage = 'sqlite:///rota.db'
study = optuna.load_study(study_name='rota', storage=storage)

print(f'\nBest Trial: #{study.best_trial.number}')
print(f'Best GOF: {study.best_value:.6f}\n')

# Get best parameters
best_params = study.best_params
print('Best Parameters:')
for k, v in best_params.items():
    if k == 'adult_baseline_immunity':
        print(f'  {k}: {v:.6f} ← PRIMARY PARAMETER')
    else:
        print(f'  {k}: {v:.6f}')

# Load target data
print('\n' + '='*70)
print('TARGET DATA (UK 2008-2012)')
print('='*70)
target_overall, target_age_dist = process_uk.process_data()
print(f'Overall incidence: {target_overall:.2f} per 100k')
print('\nAge distribution:')
for idx, row in target_age_dist.iterrows():
    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(row['ages'], f"{row['ages']}y")
    print(f'  {age_label:4s}: {row["proportion"]*100:5.2f}%')

# Run simulation with best parameters
print('\n' + '='*70)
print('RUNNING SIMULATION WITH BEST PARAMETERS...')
print('='*70)

sim = create_sim(sim_pars=best_params, seed=0)
sim.run()

print('\n✓ Simulation completed successfully')

# Process model output
print('\n' + '='*70)
print('PROCESSING MODEL OUTPUT...')
print('='*70)

# Get infection events
infected_analyzer = None
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        infected_analyzer = analyzer
        break

if infected_analyzer is None:
    print('ERROR: InfectedStrainStats analyzer not found')
    exit(1)

df = infected_analyzer.to_df()
model_overall, model_age_dist = process_uk.process_model(df, verbose=True)

# Apply reporting rate if specified
if hasattr(sim, '_reporting_rate') and sim._reporting_rate is not None:
    print(f'\nApplying reporting rate: {sim._reporting_rate:.6f}')
    model_overall = model_overall * sim._reporting_rate

print('\n' + '='*70)
print('MODEL OUTPUT')
print('='*70)
print(f'Overall incidence: {model_overall:.2f} per 100k')
print('\nAge distribution:')
for idx, row in model_age_dist.iterrows():
    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(row['ages'], f"{row['ages']}y")
    print(f'  {age_label:4s}: {row["proportion"]*100:5.2f}%')

# Calculate fit quality metrics
print('\n' + '='*70)
print('FIT QUALITY METRICS')
print('='*70)

# Overall incidence error
incidence_error = abs(model_overall - target_overall)
incidence_pct_error = (incidence_error / target_overall) * 100
print(f'\nOverall Incidence:')
print(f'  Target:    {target_overall:.2f} per 100k')
print(f'  Model:     {model_overall:.2f} per 100k')
print(f'  Error:     {incidence_error:.2f} per 100k ({incidence_pct_error:.1f}%)')

# Age distribution errors
print(f'\nAge Distribution:')
print(f'  {"Age":<6} {"Target":>8} {"Model":>8} {"Abs Error":>10} {"% Error":>10}')
print(f'  {"-"*6} {"-"*8} {"-"*8} {"-"*10} {"-"*10}')

age_errors = []
for idx in range(len(target_age_dist)):
    target_prop = target_age_dist.iloc[idx]['proportion']
    model_prop = model_age_dist.iloc[idx]['proportion']
    age = target_age_dist.iloc[idx]['ages']

    age_labels = {0: '<1y', 1: '1-2y', 2: '2-5y', 5: '≥5y'}
    age_label = age_labels.get(age, f"{age}y")

    abs_error = abs(model_prop - target_prop)
    pct_error = abs_error * 100
    age_errors.append(abs_error)

    print(f'  {age_label:<6} {target_prop*100:7.2f}% {model_prop*100:7.2f}% {abs_error*100:9.2f}% {pct_error:9.2f}%')

mean_age_error = np.mean(age_errors) * 100
max_age_error = np.max(age_errors) * 100

print(f'\n  Mean absolute error: {mean_age_error:.2f}%')
print(f'  Max absolute error:  {max_age_error:.2f}%')

# Overall GOF breakdown
print(f'\nGOF Calculation:')
print(f'  Incidence component: (|{model_overall:.2f} - {target_overall:.2f}| / {target_overall:.2f}) * 100 = {incidence_pct_error:.2f}')
print(f'  Age dist component:  Mean of absolute percentage errors = {mean_age_error:.2f}')
total_gof = incidence_pct_error + mean_age_error
print(f'  Total GOF: {total_gof:.2f}')
print(f'  (Optuna reported: {study.best_value:.2f})')

print('\n' + '='*70)
print('FIT QUALITY ASSESSMENT')
print('='*70)

# Assess fit quality
if incidence_pct_error < 10:
    print('✓ Overall incidence: EXCELLENT fit (<10% error)')
elif incidence_pct_error < 20:
    print('✓ Overall incidence: GOOD fit (10-20% error)')
elif incidence_pct_error < 30:
    print('○ Overall incidence: ACCEPTABLE fit (20-30% error)')
else:
    print('✗ Overall incidence: POOR fit (>30% error)')

if mean_age_error < 5:
    print('✓ Age distribution: EXCELLENT fit (<5% mean error)')
elif mean_age_error < 10:
    print('✓ Age distribution: GOOD fit (5-10% mean error)')
elif mean_age_error < 15:
    print('○ Age distribution: ACCEPTABLE fit (10-15% mean error)')
else:
    print('✗ Age distribution: POOR fit (>15% mean error)')

print('\n' + '='*70)
print('✓ Verification complete!')
print('='*70)
