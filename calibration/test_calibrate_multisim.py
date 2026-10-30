"""
TEST: Hybrid calibration with MultiSim - 2 trials × 5 replicates

Quick test to verify the MultiSim calibration workflow works correctly
before running the full 50 trials × 20 replicates version.

Run with: python test_calibrate_multisim.py
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
print("TEST: MULTISIM CALIBRATION - 2 Trials × 5 Replicates")
print("=" * 80)
print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()


class HybridCalibrationMultiSim(ss.Calibration):
    """
    Calibration combining age-based symptoms with fitted immunity using MultiSim
    """

    def __init__(self, sim, data, calib_pars=None, total_trials=2, n_reps=5, debug=True, **kwargs):
        calib_pars_dict = calib_pars if calib_pars else {}
        super().__init__(sim=sim, calib_pars=calib_pars_dict, **kwargs)
        self.data = data
        self.target_incidence, self.target_age_distribution = data
        self.total_trials = total_trials
        self.n_reps = n_reps
        self.debug = debug

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None, n_reps=None):
        """Run simulation(s) with given parameters"""
        if n_reps is None:
            n_reps = self.n_reps

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
        sim._sus_after_1 = sus_after_1
        sim._sus_after_2 = sus_after_2
        sim._sus_after_3plus = sus_after_3plus

        # If single replicate, just initialize and return the sim
        if n_reps == 1:
            sim.init()

            immunity_connector = sim.connectors.rotaimmunityconnector
            immunity_connector.pars['use_fixed_susceptibility'] = True
            immunity_connector.pars['sus_after_1'] = sus_after_1
            immunity_connector.pars['sus_after_2'] = sus_after_2
            immunity_connector.pars['sus_after_3plus'] = sus_after_3plus

            immunity_connector.initialize_immunity(
                min_age=18, max_age=125, min_exposures=5, max_exposures=15
            )

            return sim

        # For multiple replicates, create list of sims with different seeds
        print(f"  Creating {n_reps} simulation copies...")
        rand_seeds = np.random.randint(0, 1e6, n_reps)

        sims = []
        for i, seed in enumerate(rand_seeds):
            # Make a fresh copy of the sim for this replicate
            sim_copy = sc.dcp(sim)
            sim_copy.pars.rand_seed = seed

            # Initialize this sim
            sim_copy.init()

            # Update immunity parameters
            immunity_connector = sim_copy.connectors.rotaimmunityconnector
            immunity_connector.pars['use_fixed_susceptibility'] = True
            immunity_connector.pars['sus_after_1'] = sus_after_1
            immunity_connector.pars['sus_after_2'] = sus_after_2
            immunity_connector.pars['sus_after_3plus'] = sus_after_3plus

            immunity_connector.initialize_immunity(
                min_age=18, max_age=125, min_exposures=5, max_exposures=15
            )

            sims.append(sim_copy)

        # Create MultiSim from initialized sims
        print(f"  Creating MultiSim...")
        ms = ss.MultiSim(sims=sims)

        return ms

    def sim_to_df(self, sim):
        """Convert simulation results to data frame format"""
        df = sim.analyzers['infectedstrainstats'].to_df()
        df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)]

        ages_years = sim.people.age.values
        age_counts = {
            '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
            '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
            '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
            '>=5 y': int((ages_years >= 5).sum()),
        }

        reporting_rate = getattr(sim, '_reporting_rate', 0.0025)
        beta0 = getattr(sim, '_beta0', 0.5)
        beta1 = getattr(sim, '_beta1', -0.1)
        beta2 = getattr(sim, '_beta2', -0.01)

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

    def compute_gof_single(self, sim):
        """Compute goodness of fit for a single simulation"""
        overall_incidence, age_distribution = self.sim_to_df(sim)

        eps = 1e-6
        log_target = np.log(self.target_incidence + eps)
        log_fitted = np.log(overall_incidence + eps)
        incidence_gof = (log_target - log_fitted) ** 2

        age_gof = 0.0
        for i in range(len(self.target_age_distribution)):
            target_prop = self.target_age_distribution.proportion.iloc[i]
            fitted_prop = age_distribution.proportion.iloc[i]
            age_gof += (target_prop - fitted_prop) ** 2

        total_gof = 10 * age_gof + incidence_gof

        return total_gof

    def compute_gof(self, sim_or_multisim):
        """Compute goodness of fit"""
        if isinstance(sim_or_multisim, ss.MultiSim):
            print(f"  Computing GOF for {len(sim_or_multisim.sims)} replicates...")
            gof_values = []
            for i, sim in enumerate(sim_or_multisim.sims):
                gof = self.compute_gof_single(sim)
                gof_values.append(gof)
                if self.debug:
                    print(f"    Replicate {i+1}: GOF = {gof:.4f}")

            median_gof = np.median(gof_values)

            if self.debug:
                print(f"  GOF Summary:")
                print(f"    Median: {median_gof:.4f}")
                print(f"    Mean: {np.mean(gof_values):.4f} ± {np.std(gof_values):.4f}")
                print(f"    Range: [{np.min(gof_values):.4f}, {np.max(gof_values):.4f}]")

            return median_gof
        else:
            return self.compute_gof_single(sim_or_multisim)

    def trial_to_sim_pars(self, trial):
        """Convert trial to simulation parameters"""
        reporting_rate = trial.suggest_float('reporting_rate', 0.0001, 0.01, log=True)
        base_beta = trial.suggest_float('base_beta', 0.05, 0.5, log=True)
        beta0 = trial.suggest_float('beta0', -5.0, 2.0)
        beta1 = trial.suggest_float('beta1', -1.0, 1.0)
        beta2 = trial.suggest_float('beta2', -0.5, 0.5)

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

    def run_trial(self, trial):
        """Run a single calibration trial with multiple replicates"""
        sim_pars = self.trial_to_sim_pars(trial)

        sim_or_multisim = self.run_sim(calib_pars=None, sim_pars=sim_pars, trial=trial.number)

        print(f"  Running simulation(s)...")
        if isinstance(sim_or_multisim, ss.MultiSim):
            sim_or_multisim.run()
        else:
            sim_or_multisim.run()

        gof = self.compute_gof(sim_or_multisim)

        print(f"Trial {trial.number}: Median GOF = {gof:.4f}\n")

        return gof

    def calibrate(self):
        """Run calibration using Optuna"""
        study = optuna.create_study(
            study_name='rota_test_multisim',
            direction='minimize',
            sampler=optuna.samplers.TPESampler(seed=12345),
        )

        print(f"Running TEST calibration ({self.total_trials} trials × {self.n_reps} replicates)...")
        print("=" * 60)

        study.optimize(self.run_trial, n_trials=self.total_trials)

        print("\n" + "=" * 60)
        print("TEST CALIBRATION COMPLETE")
        print("=" * 60)

        return study


# Get target data
target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

print("Target Data:")
print(f"  Incidence: {target_incidence:.2f} per 100,000")
print("  Age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    print(f"    {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")
print()

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

# Create calibration with MultiSim (TEST: 2 trials, 5 replicates)
calib = HybridCalibrationMultiSim(
    sim=sim,
    data=(target_incidence, target_age_distribution),
    total_trials=2,
    n_reps=5,
    debug=True,
)

# Run calibration
print("=" * 80)
print("Starting TEST calibration...")
print("=" * 80)
print()

study = calib.calibrate()

# Show results
print("\nTest Results:")
print("=" * 60)
for trial in study.trials:
    if trial.value is not None:
        print(f"\nTrial {trial.number}: Median GOF = {trial.value:.4f}")
        print(f"  reporting_rate={trial.params['reporting_rate']:.6f}")
        print(f"  base_beta={trial.params['base_beta']:.4f}")
        print(f"  beta0={trial.params['beta0']:.4f}, beta1={trial.params['beta1']:.4f}, beta2={trial.params['beta2']:.4f}")
        print(f"  sus: {trial.params['sus_after_1']:.3f}/{trial.params['sus_after_2']:.3f}/{trial.params['sus_after_3plus']:.3f}")

best_trial = study.best_trial
print(f"\nBest Trial: #{best_trial.number} with GOF = {best_trial.value:.4f}")

# Save test results
results = {
    'completed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'total_trials': 2,
    'n_reps_per_trial': 5,
    'best_trial_number': best_trial.number,
    'best_gof': best_trial.value,
    'best_params': best_trial.params,
}

output_file = thisdir / 'test_calibration_results.json'
with open(output_file, 'w') as f:
    json.dump(results, f, indent=2)
print(f"\nTest results saved to: {output_file}")

print("\n" + "=" * 80)
print("TEST COMPLETE - Script is working correctly!")
print(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
