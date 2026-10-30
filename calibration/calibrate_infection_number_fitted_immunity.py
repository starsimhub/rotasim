"""
Calibrate infection_number model with FITTED susceptibility levels after each infection

This approach treats the immunity levels themselves as calibration parameters,
with the constraint that susceptibility must decrease (immunity increase) with
more infections: sus_after_1 >= sus_after_2 >= sus_after_3plus

This provides more flexibility than the exponential model while maintaining
biological plausibility.
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

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')


class UKAgeCalibrationFittedImmunity(ss.Calibration):
    """
    Calibration class for UK age-stratified incidence data with fitted immunity levels

    Calibrates 5 parameters:
    - reporting_rate: Probability of reporting given symptomatic
    - base_beta: Base transmission rate
    - sus_after_1: Susceptibility after 1 infection (monotonicity: >= sus_after_2)
    - sus_after_2: Susceptibility after 2 infections (monotonicity: >= sus_after_3plus)
    - sus_after_3plus: Susceptibility after 3+ infections (lowest bound)
    """

    def __init__(self, sim, data, calib_pars=None, total_trials=100, debug=False, **kwargs):
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

        reporting_rate = sim_pars.get('reporting_rate', 0.015)
        base_beta = sim_pars.get('base_beta', 2.0)
        sus_after_1 = sim_pars.get('sus_after_1', 0.80)
        sus_after_2 = sim_pars.get('sus_after_2', 0.65)
        sus_after_3plus = sim_pars.get('sus_after_3plus', 0.50)

        if trial is not None:
            print(f"\nTrial {trial}: "
                  f"reporting_rate={reporting_rate:.6f}, "
                  f"beta={base_beta:.4f}, "
                  f"sus_1={sus_after_1:.3f}, "
                  f"sus_2={sus_after_2:.3f}, "
                  f"sus_3+={sus_after_3plus:.3f}")

        # Update base_beta before initialization
        sim.pars.base_beta = base_beta
        for disease in sim.pars.diseases:
            if isinstance(disease, rs.Rotavirus):
                disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

        # Store reporting_rate for use in sim_to_df
        sim._reporting_rate = reporting_rate

        # Initialize the simulation
        sim.init()

        # Get immunity connector and update to use fixed susceptibility with fitted values
        immunity_connector = sim.connectors.rotaimmunityconnector
        immunity_connector.pars['use_fixed_susceptibility'] = True
        immunity_connector.pars['sus_after_1'] = sus_after_1
        immunity_connector.pars['sus_after_2'] = sus_after_2
        immunity_connector.pars['sus_after_3plus'] = sus_after_3plus

        # Initialize adult immunity (ages 18-125 with multiple prior exposures)
        immunity_connector.initialize_immunity(
            min_age=18, max_age=125, min_exposures=5, max_exposures=15
        )

        # Run simulation
        sim.run()

        return sim

    def sim_to_df(self, sim):
        """Extract incidence and age distribution from simulation results"""
        # Access analyzer instance and call its to_df() method
        analyzer = sim.analyzers['infectedstrainstats']
        df = analyzer.to_df()

        # Filter to follow-up period (years 5-10 of simulation = 2008-2012)
        # CollectionTime is in years, not days
        follow_up_start_years = 5
        follow_up_end_years = 10
        df = df[(df['CollectionTime'] >= follow_up_start_years) & (df['CollectionTime'] < follow_up_end_years)]

        # Get reporting rate (stored as attribute by run_sim)
        reporting_rate = getattr(sim, '_reporting_rate', 0.015)

        # Calculate reported cases based on severity
        df['reported'] = np.random.random(len(df)) < (reporting_rate * df['severity'])
        reported_df = df[df['reported']].copy()

        # Extract age-specific population counts
        ages_years = sim.people.age.values
        age_counts = {
            '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
            '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
            '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
            '>=5 y': int((ages_years >= 5).sum()),
        }

        # Calculate overall incidence
        total_reported = len(reported_df)
        total_pop = sum(age_counts.values())
        overall_incidence = (total_reported / total_pop) * 100000 / 5  # per 100k per year

        # Calculate age distribution (proportions)
        # Map analyzer age categories to UK calibration age groups
        # Analyzer uses: '0-2', '2-4', '4-6', '6-12', '12-24', '24-36', '36-48', '48-60', '60+' (months)
        # UK groups need: <1y (0-11 months), 1-2y (12-23 months), 2-5y (24-59 months), >=5y (60+ months)

        age_dist = {}
        if total_reported > 0:
            # <1 y: 0-11 months = analyzer categories '0-2', '2-4', '4-6', '6-12' (up to but not including 12 months)
            age_dist['<1 y'] = len(reported_df[reported_df['Age'].isin(['0-2', '2-4', '4-6', '6-12'])]) / total_reported

            # 1-2 y: 12-23 months = analyzer category '12-24'
            age_dist['1-2 y'] = len(reported_df[reported_df['Age'] == '12-24']) / total_reported

            # 2-5 y: 24-59 months = analyzer categories '24-36', '36-48', '48-60'
            age_dist['2-5 y'] = len(reported_df[reported_df['Age'].isin(['24-36', '36-48', '48-60'])]) / total_reported

            # >=5 y: 60+ months = analyzer category '60+'
            age_dist['>=5 y'] = len(reported_df[reported_df['Age'] == '60+']) / total_reported
        else:
            age_dist = {'<1 y': 0.0, '1-2 y': 0.0, '2-5 y': 0.0, '>=5 y': 0.0}

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
        """Compute goodness of fit as sum of relative errors"""
        overall_incidence, age_distribution = self.sim_to_df(sim)

        # Incidence GOF (relative error)
        incidence_gof = abs(overall_incidence - self.target_incidence) / self.target_incidence

        # Age distribution GOF (sum of relative errors across age groups)
        age_gof = 0.0
        for i in range(len(self.target_age_distribution)):
            target_prop = self.target_age_distribution.proportion.iloc[i]
            fitted_prop = age_distribution.proportion.iloc[i]
            age_gof += abs(fitted_prop - target_prop) / target_prop

        total_gof = incidence_gof + age_gof

        if self.debug:
            print(f"\n  Incidence: target={self.target_incidence:.2f}, "
                  f"fitted={overall_incidence:.2f}, GOF={incidence_gof:.4f}")
            print(f"  Age GOF: {age_gof:.4f}")
            print(f"  Total GOF: {total_gof:.4f}")

        return total_gof

    def trial_to_sim_pars(self, trial):
        """Convert optuna trial to simulation parameters with monotonicity constraints"""
        # Sample parameters with constraints
        reporting_rate = trial.suggest_float('reporting_rate',
                                            self.calib_pars['reporting_rate'][1],
                                            self.calib_pars['reporting_rate'][2])

        base_beta = trial.suggest_float('base_beta',
                                        self.calib_pars['base_beta'][1],
                                        self.calib_pars['base_beta'][2])

        # Sample immunity levels with monotonicity constraint:
        # sus_after_1 >= sus_after_2 >= sus_after_3plus
        # This ensures immunity increases with infection count

        # Sample sus_after_3plus first (lowest level)
        sus_after_3plus = trial.suggest_float('sus_after_3plus',
                                               self.calib_pars['sus_after_3plus'][1],
                                               self.calib_pars['sus_after_3plus'][2])

        # Sample sus_after_2 (must be >= sus_after_3plus)
        sus_after_2 = trial.suggest_float('sus_after_2',
                                          sus_after_3plus,  # Lower bound is sus_after_3plus
                                          self.calib_pars['sus_after_2'][2])

        # Sample sus_after_1 (must be >= sus_after_2)
        sus_after_1 = trial.suggest_float('sus_after_1',
                                          sus_after_2,  # Lower bound is sus_after_2
                                          self.calib_pars['sus_after_1'][2])

        return {
            'reporting_rate': reporting_rate,
            'base_beta': base_beta,
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

        print(f"  → GOF: {gof:.4f}")

        return gof

    def calibrate(self, calib_pars=None):
        """Run calibration using Optuna"""
        if calib_pars:
            self.calib_pars = calib_pars

        # Create Optuna study
        study = optuna.create_study(
            study_name='rota',
            direction='minimize',
            storage='sqlite:///rota.db',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=12345),
        )

        print(f"\nRunning calibration ({self.total_trials} trials)...")
        print("=" * 60)

        study.optimize(self.run_trial, n_trials=self.total_trials)

        print("\n" + "=" * 60)
        print("CALIBRATION COMPLETE")
        print("=" * 60)
        print(f"\nBest trial: {study.best_trial.number}")
        print(f"Best GOF: {study.best_value:.4f}")
        print("\nBest parameters:")
        for key, value in study.best_params.items():
            print(f"  {key}: {value:.6f}")

        # Store results
        self.best_pars = study.best_params

        return study


def main():
    print("=" * 60)
    print("UK Calibration: infection_number with FITTED immunity levels")
    print("=" * 60)
    print("\nFitting discrete immunity levels with monotonicity constraint:")
    print("  sus_after_1 >= sus_after_2 >= sus_after_3plus")
    print("  (Lower susceptibility = higher protection)")

    # Get target data
    target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

    print(f"\nUK Demographics:")
    print(f"  Birth rate: 13/1000")
    print(f"  Death rate: 6/1000")
    print(f"  Follow-up: 5 years (2008-2012)")

    # Create analyzer with infection-based severity
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

    # Create immunity connector (will be configured with fitted values during calibration)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,  # Use fixed model with fitted values
        sus_after_1=0.80,  # Initial guess
        sus_after_2=0.65,  # Initial guess
        sus_after_3plus=0.50,  # Initial guess
    )

    # Create base simulation
    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
    base_sim = rs.Sim(
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
        rand_seed=12345,
    )

    print("\n" + "=" * 60)
    print(f"Calibration target data:")
    print(f"Overall incidence: {target_incidence:.1f} per 100k")
    print(f"\nAge distribution (proportions):")
    for i, row in target_age_distribution.iterrows():
        print(f"  {row.ages:>2}: {row.proportion:>6.2%}")

    # Define calibration parameters
    # Format: [best_guess, min, max]
    calib_pars = sc.objdict(
        reporting_rate=[0.01, 0.001, 0.05],  # Reporting probability
        base_beta=[2.5, 1.0, 8.0],  # Transmission rate (expect intermediate between 1.7-7.0)
        sus_after_1=[0.85, 0.50, 1.0],  # Susceptibility after 1 infection (50%-100%)
        sus_after_2=[0.70, 0.30, 0.95],  # Susceptibility after 2 infections (30%-95%)
        sus_after_3plus=[0.50, 0.10, 0.90],  # Susceptibility after 3+ infections (10%-90%)
    )

    print("\n" + "=" * 60)
    print("Calibration parameters:")
    for key, (best, low, high) in calib_pars.items():
        if key.startswith('sus_'):
            # Show as protection for clarity
            prot_best = (1 - best) * 100
            prot_low = (1 - high) * 100  # Note: reversed for susceptibility
            prot_high = (1 - low) * 100
            print(f"  {key}: best={best:.3f} ({prot_best:.0f}% protection), "
                  f"range=[{low:.3f}, {high:.3f}] ({prot_low:.0f}%-{prot_high:.0f}% protection)")
        else:
            print(f"  {key}: best={best}, range=[{low}, {high}]")

    # Create calibration object
    calib = UKAgeCalibrationFittedImmunity(
        sim=base_sim,
        data=(target_incidence, target_age_distribution),
        calib_pars=calib_pars,
        total_trials=50,  # Use 50 trials for good coverage
        debug=False,
    )

    # Run calibration
    study = calib.calibrate(calib_pars=calib_pars)

    # Save results
    results = {
        'model': 'infection_number_fitted_immunity',
        'description': 'Fitted immunity levels with monotonicity constraint',
        'best_parameters': study.best_params,
        'best_gof': float(study.best_value),
        'n_trials': len(study.trials),
        'target_incidence': float(target_incidence),
        'calibration_pars': {k: list(v) for k, v in calib_pars.items()},
    }

    results_file = thisdir / 'uk_calibration_results_infection_number_fitted_immunity.json'
    sc.savejson(results_file, results, indent=2)
    print(f"\n✓ Results saved to: {results_file}")

    # Print comparison with previous models
    print("\n" + "=" * 60)
    print("Comparison with previous approaches:")
    print("=" * 60)

    print("\nStandard exponential model:")
    print("  Beta: ~1.74")
    print("  Protection growth: Continuous exponential (rate=0.1)")
    print("  Issue: Only 12% of target incidence")

    print("\nFixed susceptibility model:")
    print("  Beta: 6.97")
    print("  Protection levels: 33%, 50%, 64% (fixed)")
    print("  Issue: Implausibly high beta, simulations hung")

    print("\nFitted immunity model (this approach):")
    print(f"  Beta: {study.best_params['base_beta']:.2f}")
    sus_1 = study.best_params['sus_after_1']
    sus_2 = study.best_params['sus_after_2']
    sus_3 = study.best_params['sus_after_3plus']
    print(f"  Protection levels: {(1-sus_1)*100:.1f}%, {(1-sus_2)*100:.1f}%, {(1-sus_3)*100:.1f}% (fitted)")
    print(f"  GOF: {study.best_value:.4f}")
    print("  Advantage: Data-driven immunity levels, monotonicity enforced")


if __name__ == '__main__':
    main()
