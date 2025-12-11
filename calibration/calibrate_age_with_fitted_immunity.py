"""
Hybrid calibration: Age-based symptoms + Fitted immunity

This model combines:
1. Age-based symptom probability: P(symptomatic) = logistic(beta0 + beta1*age + beta2*age^2)
2. Fitted susceptibility by infection number: sus_after_1, sus_after_2, sus_after_3plus

Parameters to calibrate (8 total):
- reporting_rate: Probability of reporting given symptomatic
- base_beta: Base transmission rate
- beta0, beta1, beta2: Age-based symptom model parameters
- sus_after_1, sus_after_2, sus_after_3plus: Fitted immunity levels

Fixed parameters:
- severity = 0.2 (constant, reporting_rate adjusted accordingly)
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
import matplotlib.pyplot as plt

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')

print("=" * 80)
print("HYBRID CALIBRATION: Age-Based Symptoms + Fitted Immunity")
print("=" * 80)
print("\nModel Features:")
print("  - Age affects symptom probability (beta0 + beta1*age + beta2*age^2)")
print("  - Infection history affects immunity with fitted parameters")
print("  - FIXED severity = 0.2 (reporting_rate adjusted accordingly)")
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

    def __init__(self, sim, data, calib_pars=None, total_trials=20, debug=False, **kwargs):
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

        if trial is not None:
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
        follow_up_start_years = 5
        follow_up_end_years = 10
        df = df[(df['CollectionTime'] >= follow_up_start_years) & (df['CollectionTime'] < follow_up_end_years)]

        # Get parameters
        reporting_rate = getattr(sim, '_reporting_rate', 0.0025)
        beta0 = getattr(sim, '_beta0', 0.5)
        beta1 = getattr(sim, '_beta1', -0.1)
        beta2 = getattr(sim, '_beta2', -0.01)

        # Extract age-specific population counts
        ages_years = sim.people.age.values
        age_counts = {
            '<1 y': int(((ages_years >= 0) & (ages_years < 1)).sum()),
            '1-2 y': int(((ages_years >= 1) & (ages_years < 2)).sum()),
            '2-5 y': int(((ages_years >= 2) & (ages_years < 5)).sum()),
            '>=5 y': int((ages_years >= 5).sum()),
        }

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
        """Compute goodness of fit with log-scale incidence and squared errors for age

        GOF = 10 * GOF_age + GOF_incidence
        where:
        - GOF_incidence = (log(target) - log(model))^2
        - GOF_age = sum((proportion_target - proportion_model)^2) for all age groups
        - Proportions are on 0-1 scale (not percentages)
        """
        import numpy as np
        overall_incidence, age_distribution = self.sim_to_df(sim)

        # Incidence GOF (log-scale squared error)
        # Add small epsilon to avoid log(0)
        eps = 1e-6
        log_target = np.log(self.target_incidence + eps)
        log_fitted = np.log(overall_incidence + eps)
        incidence_gof = (log_target - log_fitted) ** 2

        # Age distribution GOF (sum of squared errors across age groups)
        # Proportions are already on 0-1 scale
        age_gof = 0.0
        for i in range(len(self.target_age_distribution)):
            target_prop = self.target_age_distribution.proportion.iloc[i]
            fitted_prop = age_distribution.proportion.iloc[i]
            age_gof += (target_prop - fitted_prop) ** 2

        # Total GOF: 10x weight on age distribution
        total_gof = 10 * age_gof + incidence_gof

        if self.debug:
            print(f"\n  Incidence: target={self.target_incidence:.2f}, "
                  f"fitted={overall_incidence:.2f}, GOF={incidence_gof:.4f}")
            print(f"  Age GOF: {age_gof:.4f}")
            print(f"  Total GOF: {total_gof:.4f}")

        return total_gof

    def trial_to_sim_pars(self, trial):
        """Convert optuna trial to simulation parameters with monotonicity constraints"""
        # Sample parameters
        reporting_rate = trial.suggest_float('reporting_rate',
                                            self.calib_pars['reporting_rate'][1],
                                            self.calib_pars['reporting_rate'][2])

        base_beta = trial.suggest_float('base_beta',
                                        self.calib_pars['base_beta'][1],
                                        self.calib_pars['base_beta'][2])

        # Age symptom parameters
        beta0 = trial.suggest_float('beta0',
                                   self.calib_pars['beta0'][1],
                                   self.calib_pars['beta0'][2])

        beta1 = trial.suggest_float('beta1',
                                   self.calib_pars['beta1'][1],
                                   self.calib_pars['beta1'][2])

        beta2 = trial.suggest_float('beta2',
                                   self.calib_pars['beta2'][1],
                                   self.calib_pars['beta2'][2])

        # Sample immunity levels with monotonicity constraint:
        # sus_after_1 >= sus_after_2 >= sus_after_3plus
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

        print(f"  → GOF: {gof:.4f}")

        return gof

    def calibrate(self, calib_pars=None):
        """Run calibration using Optuna"""
        if calib_pars:
            self.calib_pars = calib_pars

        # Create Optuna study
        study = optuna.create_study(
            study_name='rota_hybrid',
            direction='minimize',
            storage='sqlite:///rota_hybrid.db',
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
            if key.startswith('sus_'):
                protection = (1 - value) * 100
                print(f"  {key}: {value:.6f} ({protection:.2f}% protection)")
            else:
                print(f"  {key}: {value:.6f}")

        # Store results
        self.best_pars = study.best_params

        return study


def main():
    # Get target data
    target_incidence, target_age_distribution = process_incidence_uk_age.process_data()

    print(f"\nTarget incidence: {target_incidence:.2f} per 100,000")
    print("Target age distribution:")
    age_labels = ['0-11 months', '12-23 months', '24-59 months', '5+ years']
    for i, label in enumerate(age_labels):
        print(f"  {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

    # Define calibration parameters
    # Note: severity fixed at 0.2, so reporting_rate range adjusted accordingly
    # Previous: severity=0.05, reporting=[0.001, 0.03] → product=[0.00005, 0.0015]
    # Now: severity=0.2, reporting=[0.00025, 0.01] → product=[0.00005, 0.002]
    calib_pars = dict(
        reporting_rate=[0.0025, 0.0003, 0.01],  # Adjusted for severity=0.2, upper bound increased
        base_beta=[3.0, 2.0, 6.0],
        beta0=[0.5, -2, 3],              # Symptom intercept
        beta1=[-0.1, -0.5, 0.3],         # Age linear effect
        beta2=[-0.01, -0.05, 0.02],      # Age quadratic effect
        sus_after_1=[0.85, 0.5, 1.0],    # After 1st infection
        sus_after_2=[0.7, 0.3, 0.95],    # After 2nd infection
        sus_after_3plus=[0.5, 0.1, 0.9], # After 3+ infections
    )

    print("\n" + "-" * 80)
    print("Calibration Parameter Ranges:")
    print("-" * 80)
    print("FIXED: severity = 0.2")
    for param, (guess, lower, upper) in calib_pars.items():
        print(f"  {param:<20}: [{lower:.4f}, {upper:.4f}] (guess: {guess:.4f})")

    # Create analyzer with FIXED severity = 0.2
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)

    # Create immunity connector (will be configured with fitted values during calibration)
    immunity_connector = rs.RotaImmunityConnector(
        use_fixed_susceptibility=True,
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
    )

    # Create calibration object
    n_trials = 30
    print(f"\n{'=' * 80}")
    print(f"Starting calibration with {n_trials} trials...")
    print(f"{'=' * 80}\n")

    calib = HybridCalibration(
        sim=base_sim,
        data=(target_incidence, target_age_distribution),
        calib_pars=calib_pars,
        total_trials=n_trials,
        debug=False,
    )

    # Run calibration
    study = calib.calibrate()

    # Extract results
    best_pars = study.best_params
    best_gof = study.best_value

    print("\n" + "=" * 80)
    print("CALIBRATION COMPLETE")
    print("=" * 80)
    print(f"\nBest GOF: {best_gof:.6f}")
    print("\nBest parameters:")
    for k, v in best_pars.items():
        if k.startswith('sus_'):
            protection = (1 - v) * 100
            print(f"  {k:<20}: {v:.6f} ({protection:.2f}% protection)")
        else:
            print(f"  {k:<20}: {v:.6f}")

    # Save calibration results
    output_file = thisdir / 'uk_calibration_results_hybrid_age_fitted_immunity.json'
    results = {
        'model': 'hybrid_age_fitted_immunity',
        'description': 'Age-based symptoms + fitted immunity by infection number (severity=0.2)',
        'n_trials': n_trials,
        'best_parameters': best_pars,
        'best_gof': best_gof,
        'target_incidence': float(target_incidence),
        'fixed_severity': 0.2,
        'calibration_pars': {k: list(v) for k, v in calib_pars.items()},
    }

    sc.savejson(output_file, results, indent=2)
    print(f"\n✓ Calibration results saved to: {output_file}")

    # Generate figure for MLE
    print("\n" + "=" * 80)
    print("EVALUATING BEST MODEL AND GENERATING FIGURE")
    print("=" * 80)

    print("\nRunning simulation with best parameters...")
    sim = calib.run_sim(calib_pars=None, sim_pars=best_pars, trial=None)
    overall_incidence, age_distribution = calib.sim_to_df(sim)

    print("✓ Simulation complete")

    # Display results
    print("\nMLE Results:")
    print(f"  Incidence: {overall_incidence:.2f} per 100k (target: {target_incidence:.2f})")
    print("  Age distribution:")
    for i, label in enumerate(age_labels):
        print(f"    {label:<15}: {age_distribution.proportion.iloc[i]*100:>6.2f}% (target: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%)")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Panel A: Age Distribution
    ax1 = axes[0]
    x = np.arange(len(age_labels))
    width = 0.35

    target_props = [target_age_distribution.proportion.iloc[i] * 100 for i in range(4)]
    fitted_props = [age_distribution.proportion.iloc[i] * 100 for i in range(4)]

    bars1 = ax1.bar(x - width/2, target_props, width, label='Target (UK Data)',
                    color='#3498DB', alpha=0.8, edgecolor='black', linewidth=1.5)
    bars2 = ax1.bar(x + width/2, fitted_props, width, label='MLE Fit',
                    color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax1.set_ylabel('Proportion of Cases (%)', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Age Group', fontsize=12, fontweight='bold')
    ax1.set_title('A. Age Distribution of Reported Cases', fontsize=13, fontweight='bold', pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(['0-11\nmonths', '12-23\nmonths', '24-59\nmonths', '5+\nyears'], fontsize=10)
    ax1.legend(fontsize=11, loc='upper left')
    ax1.grid(axis='y', alpha=0.3)
    ax1.set_ylim(0, max(max(target_props), max(fitted_props)) * 1.2)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 1,
                    f'{height:.1f}%', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Panel B: Incidence Comparison
    ax2 = axes[1]
    incidence_vals = [target_incidence, overall_incidence]
    colors = ['#3498DB', '#E74C3C']
    bars = ax2.bar(['Target\n(UK Data)', 'MLE Fit'],
                   incidence_vals, color=colors, alpha=0.8, edgecolor='black', linewidth=2, width=0.5)

    ax2.set_ylabel('Incidence (per 100,000)', fontsize=12, fontweight='bold')
    ax2.set_title('B. Overall Incidence Comparison', fontsize=13, fontweight='bold', pad=10)
    ax2.set_ylim(0, max(incidence_vals) * 1.3)
    ax2.axhline(y=target_incidence, color='blue', linestyle='--', alpha=0.5, linewidth=2)
    ax2.grid(axis='y', alpha=0.3)

    # Add value labels
    for bar, val in zip(bars, incidence_vals):
        height = bar.get_height()
        pct_of_target = (val / target_incidence) * 100
        ax2.text(bar.get_x() + bar.get_width()/2., height + 1.5,
                 f'{val:.1f}\n({pct_of_target:.0f}% of target)',
                 ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add GOF annotation
    incidence_gof = abs(overall_incidence - target_incidence) / target_incidence
    age_gof = ((age_distribution.proportion - target_age_distribution.proportion).abs() / target_age_distribution.proportion).sum()
    error = overall_incidence - target_incidence
    error_pct = (error / target_incidence) * 100
    ax2.text(0.5, 0.15,
            f'GOF: {best_gof:.4f}\nIncidence error: {error:+.1f} ({error_pct:+.1f}%)\nAge GOF: {age_gof:.4f}',
            transform=ax2.transAxes, fontsize=10, ha='center',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.6))

    # Overall title
    fig.suptitle('Hybrid Model: Age-Based Symptoms + Fitted Immunity (MLE Fit)',
                 fontsize=14, fontweight='bold', y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # Save figure
    figure_file = thisdir / 'hybrid_calibration_mle_fit.png'
    plt.savefig(figure_file, dpi=300, bbox_inches='tight')
    print(f"\n✓ Figure saved to: {figure_file}")

    figure_pdf = thisdir / 'hybrid_calibration_mle_fit.pdf'
    plt.savefig(figure_pdf, bbox_inches='tight')
    print(f"✓ Figure saved to: {figure_pdf}")

    # Save detailed results with fit metrics
    results['mle_evaluation'] = {
        'incidence': float(overall_incidence),
        'age_distribution': age_distribution.proportion.tolist(),
        'incidence_gof': float(incidence_gof),
        'age_gof': float(age_gof),
    }

    sc.savejson(output_file, results, indent=2)

    print("\n" + "=" * 80)
    print("CALIBRATION AND EVALUATION COMPLETE")
    print("=" * 80)
    print("\nResults:")
    print(f"  Best GOF: {best_gof:.6f}")
    print(f"  Incidence error: {error:+.1f} per 100k ({error_pct:+.1f}%)")
    print(f"  Age GOF: {age_gof:.4f}")
    print("=" * 80)


if __name__ == '__main__':
    main()
