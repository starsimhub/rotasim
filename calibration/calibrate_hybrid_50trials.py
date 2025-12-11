"""
Hybrid calibration: 50 trials with new GOF metric

This model combines:
1. Age-based symptom probability: P(symptomatic) = logistic(beta0 + beta1*age + beta2*age^2)
2. Fitted susceptibility by infection number: sus_after_1, sus_after_2, sus_after_3plus

New GOF metric:
- GOF = 10 * GOF_age + GOF_incidence
- GOF_incidence = (log(target) - log(model))^2
- GOF_age = sum((proportion_target - proportion_model)^2) for all age groups

Run on VM with: python calibrate_hybrid_50trials.py
"""
import sys
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna
import json
from datetime import datetime

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("HYBRID CALIBRATION: 50 Trials with New GOF Metric")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\nNew GOF Formula:")
print("  GOF = 10 * GOF_age + GOF_incidence")
print("  GOF_incidence = (log(target) - log(model))^2")
print("  GOF_age = sum((proportion_target - proportion_model)^2)")
print("\nCalibrating 8 parameters:")
print("  1. reporting_rate")
print("  2. base_beta")
print("  3-5. beta0, beta1, beta2 (age symptom model)")
print("  6-8. sus_after_1, sus_after_2, sus_after_3plus (fitted immunity)")
print("=" * 80)


class HybridCalibration(ss.Calibration):
    """
    Calibration combining age-based symptoms with fitted immunity
    """

    def __init__(self, sim, data, calib_pars=None, total_trials=50, debug=False, **kwargs):
        calib_pars_dict = calib_pars if calib_pars else {}
        super().__init__(sim=sim, calib_pars=calib_pars_dict, **kwargs)
        self.data = data
        self.target_incidence, self.target_age_distribution = data
        self.total_trials = total_trials
        self.debug = debug

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """Run simulation with given parameters"""
        # Make a copy of base sim
        sim = sc.dcp(self.sim)

        # Extract parameters
        if sim_pars is None:
            sim_pars = {}

        reporting_rate = sim_pars.get('reporting_rate', 0.0025)
        base_beta = sim_pars.get('base_beta', 3.0)
        beta0 = sim_pars.get('beta0', 0.5)
        beta1 = sim_pars.get('beta1', -0.1)
        beta2 = sim_pars.get('beta2', -0.01)
        sus_after_1 = sim_pars.get('sus_after_1', 0.80)
        sus_after_2 = sim_pars.get('sus_after_2', 0.65)
        sus_after_3plus = sim_pars.get('sus_after_3plus', 0.50)

        if trial is not None and self.debug:
            print(f"\nTrial {trial}:")
            print(f"  Age params: beta0={beta0:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}")
            print(f"  Immunity: sus_1={sus_after_1:.3f}, sus_2={sus_after_2:.3f}, sus_3+={sus_after_3plus:.3f}")
            print(f"  Other: reporting={reporting_rate:.6f}, beta={base_beta:.4f}")

        # Update base_beta before initialization
        sim.pars.base_beta = base_beta
        for disease in sim.pars.diseases:
            if isinstance(disease, rs.Rotavirus):
                disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

        # Store parameters for use in sim_to_df
        sim._reporting_rate = reporting_rate
        sim._beta0 = beta0
        sim._beta1 = beta1
        sim._beta2 = beta2

        # Initialize the simulation
        sim.init()

        # Get immunity connector and update to use fixed susceptibility with fitted values
        immunity_connector = sim.connectors.rotaimmunityconnector
        immunity_connector.pars['use_fixed_susceptibility'] = True
        immunity_connector.pars['sus_after_1'] = sus_after_1
        immunity_connector.pars['sus_after_2'] = sus_after_2
        immunity_connector.pars['sus_after_3plus'] = sus_after_3plus

        # Initialize adult immunity
        immunity_connector.initialize_immunity(
            min_age=18, max_age=125, min_exposures=5, max_exposures=15
        )

        # Run simulation
        sim.run()

        return sim

    def sim_to_df(self, sim):
        """Convert simulation results to data frame format"""
        df = sim.analyzers['infectedstrainstats'].to_df()

        # Filter to follow-up period (years 5-10)
        df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

        # Extract age-specific population counts
        ages_years = sim.people.age.values
        age_counts = {
            '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
            '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
            '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
            '>=5 y': int((ages_years >= 5).sum()),
        }

        # Get stored parameters
        reporting_rate = getattr(sim, '_reporting_rate', 0.0025)
        beta0 = getattr(sim, '_beta0', 0.5)
        beta1 = getattr(sim, '_beta1', -0.1)
        beta2 = getattr(sim, '_beta2', -0.01)

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

        return overall_incidence, age_distribution

    def compute_gof(self, sim):
        """Compute goodness of fit with NEW GOF metric

        GOF = 10 * GOF_age + GOF_incidence
        where:
        - GOF_incidence = (log(target) - log(model))^2
        - GOF_age = sum((proportion_target - proportion_model)^2) for all age groups
        - Proportions are on 0-1 scale (not percentages)
        """
        overall_incidence, age_distribution = self.sim_to_df(sim)

        # Incidence GOF (log-scale squared error)
        eps = 1e-6
        log_target = np.log(self.target_incidence + eps)
        log_fitted = np.log(overall_incidence + eps)
        incidence_gof = (log_target - log_fitted) ** 2

        # Age distribution GOF (sum of squared errors across age groups)
        age_gof = 0.0
        for i in range(len(self.target_age_distribution)):
            target_prop = self.target_age_distribution.proportion.iloc[i]
            fitted_prop = age_distribution.proportion.iloc[i]
            age_gof += (target_prop - fitted_prop) ** 2

        # Total GOF: 10x weight on age distribution
        total_gof = 10 * age_gof + incidence_gof

        if self.debug:
            print(f"  Incidence: target={self.target_incidence:.2f}, "
                  f"fitted={overall_incidence:.2f}, GOF={incidence_gof:.4f}")
            print(f"  Age GOF: {age_gof:.4f}")
            print(f"  Total GOF: {total_gof:.4f}")

        return total_gof

    def trial_to_sim_pars(self, trial):
        """Convert trial to simulation parameters with monotonicity constraints"""
        # Sample parameters
        reporting_rate = trial.suggest_float('reporting_rate', 0.0001, 0.01, log=True)
        base_beta = trial.suggest_float('base_beta', 0.05, 0.5, log=True)
        beta0 = trial.suggest_float('beta0', -5.0, 2.0)
        beta1 = trial.suggest_float('beta1', -1.0, 1.0)
        beta2 = trial.suggest_float('beta2', -0.5, 0.5)

        # Sample immunity levels with monotonicity constraint
        # sus_after_3plus <= sus_after_2 <= sus_after_1
        sus_after_3plus = trial.suggest_float('sus_after_3plus', 0.1, 1.0)
        sus_after_2 = trial.suggest_float('sus_after_2', sus_after_3plus, 1.0)
        sus_after_1 = trial.suggest_float('sus_after_1', sus_after_2, 1.0)

        return {
            'reporting_rate': reporting_rate,
            'base_beta': base_beta,
            'beta0': beta0,
            'beta1': beta1,
            'beta2': beta2,
            'sus_after_1': sus_after_1,
            'sus_after_2': sus_after_2,
            'sus_after_3plus': sus_after_3plus,
        }

    def trial_pars_to_sim_pars(self, trial_pars=None, which='best'):
        """Convert trial parameters to simulation parameters"""
        return trial_pars

    def run_trial(self, trial):
        """Run a single calibration trial"""
        # Convert trial to simulation parameters
        sim_pars = self.trial_to_sim_pars(trial)

        # Run simulation
        sim = self.run_sim(calib_pars=None, sim_pars=sim_pars, trial=trial.number)

        # Compute goodness of fit
        gof = self.compute_gof(sim)

        print(f"Trial {trial.number}: GOF = {gof:.4f}")

        return gof

    def calibrate(self):
        """Run calibration using Optuna"""
        # Create Optuna study
        study = optuna.create_study(
            study_name='rota_hybrid_50trials',
            direction='minimize',
            storage='sqlite:///rota_hybrid_50trials.db',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=12345),
        )

        print(f"\nRunning calibration ({self.total_trials} trials)...")
        print("=" * 60)

        study.optimize(self.run_trial, n_trials=self.total_trials)

        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)

        return study


# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("\nTarget Data:")
print(f"  Incidence: {target_incidence:.2f} per 100,000")
print("  Age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    print(f"    {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Create base simulation
analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)
immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)

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

# Create calibration
calib = HybridCalibration(
    sim=sim,
    data=(target_incidence, target_age_distribution),
    total_trials=50,
    debug=False,  # Set to False for cleaner output on VM
)

# Run calibration
print("\n" + "=" * 80)
print("Starting calibration with 50 trials...")
print("=" * 80)

study = calib.calibrate()

# Show best trial
best_trial = study.best_trial
print(f"\nBest Trial: #{best_trial.number}")
print(f"Best GOF: {best_trial.value:.4f}")
print("\nBest Parameters:")
for key, value in best_trial.params.items():
    if key.startswith('sus_'):
        protection = (1 - value) * 100
        print(f"  {key:<20}: {value:.6f} ({protection:.2f}% protection)")
    else:
        print(f"  {key:<20}: {value:.6f}")

# Save results to JSON
results = {
    'completed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_trials': 50,
    'best_trial_number': best_trial.number,
    'best_gof': best_trial.value,
    'best_params': best_trial.params,
    'target_incidence': target_incidence,
    'target_age_distribution': target_age_distribution.proportion.tolist(),
}

output_file = thisdir / 'calibration_50trials_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nResults saved to: {output_file}")

# Show top 5 trials
print("\n" + "=" * 80)
print("Top 5 Trials:")
print("=" * 80)
trials = sorted(study.trials, key=lambda t: t.value if t.value is not None else float('inf'))
for i, trial in enumerate(trials[:5]):
    if trial.value is not None:
        print(f"\n#{trial.number}: GOF = {trial.value:.4f}")
        print(f"  reporting_rate={trial.params['reporting_rate']:.6f}, base_beta={trial.params['base_beta']:.4f}")
        print(f"  beta0={trial.params['beta0']:.4f}, beta1={trial.params['beta1']:.4f}, beta2={trial.params['beta2']:.4f}")
        print(f"  sus: {trial.params['sus_after_1']:.3f}/{trial.params['sus_after_2']:.3f}/{trial.params['sus_after_3plus']:.3f}")

print("\n" + "=" * 80)
print("DONE")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
