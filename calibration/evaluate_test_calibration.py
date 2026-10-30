"""
Evaluate top trials from test calibration with new GOF metric
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
print("EVALUATING TEST CALIBRATION TRIALS")
print("=" * 80)
print(f"\nTarget incidence: {target_incidence:.2f} per 100,000")
print("Target age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Load test study
study = optuna.load_study(study_name='rota_hybrid_test', storage='sqlite:///rota_hybrid_test.db')

# Get completed trials, sorted by new GOF
completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
completed_trials.sort(key=lambda t: t.value)

print(f"\nTotal completed trials: {len(completed_trials)}")
print(f"Evaluating top 5 trials...")

# Function to compute new GOF
def compute_new_gof(incidence, age_dist):
    """New GOF: log-scale incidence + squared errors for age"""
    eps = 1e-6
    inc_gof = (np.log(target_incidence + eps) - np.log(incidence + eps)) ** 2
    age_gof = sum((target_age_distribution.proportion.iloc[i] - age_dist.proportion.iloc[i]) ** 2
                  for i in range(4))
    return 10 * age_gof + inc_gof

# Function to run simulation
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

    # Process
    overall_incidence, age_distribution = process_incidence_uk_age.process_model(
        df,
        age_counts=age_counts,
        symptom_model='age_and_infection_simple',
        beta0=params['beta0'],
        beta1=params['beta1'],
        beta2=params['beta2'],
        beta3=0,
        reporting_rate=params['reporting_rate']
    )

    return overall_incidence, age_distribution

# Evaluate top 5 trials
results = []
for i, trial in enumerate(completed_trials[:5]):
    print(f"\nEvaluating Trial #{trial.number} ({i+1}/5)...")

    # Run simulation
    incidence, age_dist = run_simulation(trial.params)

    # Compute GOF
    new_gof = compute_new_gof(incidence, age_dist)

    results.append({
        'trial': trial.number,
        'gof_stored': trial.value,
        'gof_computed': new_gof,
        'incidence': incidence,
        'pct_of_target': (incidence / target_incidence) * 100,
        'age_0_1': age_dist.proportion.iloc[0],
        'age_1_2': age_dist.proportion.iloc[1],
        'age_2_5': age_dist.proportion.iloc[2],
        'age_5plus': age_dist.proportion.iloc[3],
    })

    print(f"  Incidence: {incidence:.2f} per 100k ({(incidence/target_incidence)*100:.0f}% of target {target_incidence:.2f})")
    print(f"  GOF: {new_gof:.4f} (stored: {trial.value:.4f})")

# Create results table
df_results = pd.DataFrame(results)

print("\n" + "=" * 80)
print("TEST CALIBRATION: TOP 5 TRIALS")
print("=" * 80)
print(df_results[['trial', 'incidence', 'pct_of_target', 'gof_computed']].to_string(index=False))

print("\n" + "=" * 80)
print("COMPARISON WITH RE-EVALUATION RESULTS")
print("=" * 80)
print("\nFrom re-evaluation of original calibration:")
print("  Trial #49: 20.27 per 100k (74% of target) - Best by new GOF")
print("  Trial #41: 60.15 per 100k (218% of target) - Best by old GOF")
print("\nFrom test calibration (20 trials with new GOF):")
best_test = df_results.iloc[0]
print(f"  Trial #{int(best_test['trial'])}: {best_test['incidence']:.2f} per 100k ({best_test['pct_of_target']:.0f}% of target)")

print("\n" + "=" * 80)
print("SUMMARY")
print("=" * 80)
print(f"Target incidence: {target_incidence:.2f} per 100,000")
print(f"\nBest from original calibration (new GOF): Trial #49 = {20.27:.2f} per 100k")
print(f"Best from test calibration: Trial #{int(best_test['trial'])} = {best_test['incidence']:.2f} per 100k")

if best_test['incidence'] < 20.27:
    print(f"\n✓ Test calibration found BETTER trial! ({best_test['incidence']:.2f} vs 20.27)")
else:
    print(f"\n⚠ Original calibration still has best trial (20.27 vs {best_test['incidence']:.2f})")

print("\n" + "=" * 80)
print("DONE")
print("=" * 80)
