"""
Re-evaluate existing hybrid calibration trials with new GOF metric
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import pandas as pd
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("=" * 80)
print("RE-EVALUATING TRIALS WITH NEW GOF METRIC")
print("=" * 80)
print(f"\nTarget incidence: {target_incidence:.2f} per 100,000")
print("Target age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Load study
study = optuna.load_study(study_name='rota_hybrid', storage='sqlite:///rota_hybrid.db')

# Get completed trials, sorted by old GOF
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed_trials.sort(key=lambda t: t.value)

print(f"\nTotal completed trials: {len(completed_trials)}")
print(f"Evaluating top 10 trials...")

# Function to compute old GOF
def compute_old_gof(incidence, age_dist):
    """Original GOF: relative errors"""
    inc_gof = abs(incidence - target_incidence) / target_incidence
    age_gof = sum(abs(age_dist.proportion.iloc[i] - target_age_distribution.proportion.iloc[i]) /
                  target_age_distribution.proportion.iloc[i] for i in range(4))
    return inc_gof + age_gof

# Function to compute new GOF
def compute_new_gof(incidence, age_dist):
    """New GOF: log-scale incidence + squared errors for age"""
    eps = 1e-6
    inc_gof = (np.log(target_incidence + eps) - np.log(incidence + eps)) ** 2
    age_gof = sum((target_age_distribution.proportion.iloc[i] - age_dist.proportion.iloc[i]) ** 2
                  for i in range(4))
    return 10 * age_gof + inc_gof

# Function to run simulation (same as plot script)
def run_simulation(params):
    """Run simulation with specific parameters"""
    # Create analyzer
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)

    # Create immunity connector
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)

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

    # Store parameters
    sim._reporting_rate = params['reporting_rate']
    sim._beta0 = params['beta0']
    sim._beta1 = params['beta1']
    sim._beta2 = params['beta2']

    # Initialize
    sim.init()

    # Update immunity connector AFTER initialization
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
    df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

    # Get age counts
    ages_years = sim.people.age.values
    age_counts = {
        '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
        '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
        '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
        '>=5 y': int((ages_years >= 5).sum()),
    }

    # Get stored parameters
    reporting_rate = getattr(sim, '_reporting_rate', params['reporting_rate'])
    beta0 = getattr(sim, '_beta0', params['beta0'])
    beta1 = getattr(sim, '_beta1', params['beta1'])
    beta2 = getattr(sim, '_beta2', params['beta2'])

    # Process
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

    return overall_incidence, age_distribution

# Evaluate top 10 trials
results = []
for i, trial in enumerate(completed_trials[:10]):
    print(f"\nEvaluating Trial #{trial.number} ({i+1}/10)...")

    # Run simulation
    incidence, age_dist = run_simulation(trial.params)

    # Compute both GOFs
    old_gof = compute_old_gof(incidence, age_dist)
    new_gof = compute_new_gof(incidence, age_dist)

    results.append({
        'trial': trial.number,
        'old_gof_stored': trial.value,
        'old_gof_computed': old_gof,
        'new_gof': new_gof,
        'incidence': incidence,
        'age_0_1': age_dist.proportion.iloc[0],
        'age_1_2': age_dist.proportion.iloc[1],
        'age_2_5': age_dist.proportion.iloc[2],
        'age_5plus': age_dist.proportion.iloc[3],
    })

    print(f"  Incidence: {incidence:.2f} per 100k (target: {target_incidence:.2f})")
    print(f"  Old GOF: {old_gof:.4f} (stored: {trial.value:.4f})")
    print(f"  New GOF: {new_gof:.4f}")

# Create comparison table
df_results = pd.DataFrame(results)
df_results['old_rank'] = df_results['old_gof_computed'].rank()
df_results['new_rank'] = df_results['new_gof'].rank()
df_results['rank_change'] = df_results['old_rank'] - df_results['new_rank']

print("\n" + "=" * 80)
print("COMPARISON: OLD vs NEW GOF")
print("=" * 80)
print("\nSorted by OLD GOF:")
df_old = df_results.sort_values('old_gof_computed')
print(df_old[['trial', 'incidence', 'old_gof_computed', 'new_gof', 'old_rank', 'new_rank', 'rank_change']].to_string(index=False))

print("\n" + "-" * 80)
print("Sorted by NEW GOF:")
df_new = df_results.sort_values('new_gof')
print(df_new[['trial', 'incidence', 'old_gof_computed', 'new_gof', 'old_rank', 'new_rank', 'rank_change']].to_string(index=False))

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
best_old = df_results.loc[df_results['old_gof_computed'].idxmin()]
best_new = df_results.loc[df_results['new_gof'].idxmin()]

print(f"\nBest trial by OLD GOF: #{int(best_old['trial'])} (GOF={best_old['old_gof_computed']:.4f}, inc={best_old['incidence']:.2f})")
print(f"Best trial by NEW GOF: #{int(best_new['trial'])} (GOF={best_new['new_gof']:.4f}, inc={best_new['incidence']:.2f})")

if best_old['trial'] != best_new['trial']:
    print(f"\n⚠ NEW GOF selects different best trial!")
    print(f"  Old best (#{int(best_old['trial'])}): inc={best_old['incidence']:.2f} (new_rank={int(best_old['new_rank'])})")
    print(f"  New best (#{int(best_new['trial'])}): inc={best_new['incidence']:.2f} (old_rank={int(best_new['old_rank'])})")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
