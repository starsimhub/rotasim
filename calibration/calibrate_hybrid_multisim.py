"""
Hybrid calibration with MultiSim: 50 trials × 20 replicates each

This model combines:
1. Age-based symptom probability: P(symptomatic) = logistic(beta0 + beta1*age + beta2*age^2)
2. Fitted susceptibility by infection number: sus_after_1, sus_after_2, sus_after_3plus

Key improvements over single-run calibration:
- Runs 20 independent simulations per parameter set to reduce stochastic variation
- Uses MEDIAN GOF across replicates for more robust parameter selection
- Supports PARALLEL execution across cluster nodes via Optuna's multi-worker system

Parallelization:
- WITHIN trials: 20 simulations run in parallel via MultiSim (uses all CPUs on node)
- ACROSS trials: Multiple workers can run different trials simultaneously

GOF metric:
- GOF = 10 * GOF_age + GOF_incidence
- GOF_incidence = (log(target) - log(model))^2
- GOF_age = sum((proportion_target - proportion_model)^2) for all age groups

Usage:
  Single worker:  python calibrate_hybrid_multisim.py
  Multi-worker:   python calibrate_hybrid_multisim.py --n-trials 10 --total-trials 50
                  (run multiple instances with different --worker-id values)

  On cluster with SLURM array jobs:
    sbatch --array=1-5 job_script.sh
    (each array task runs: python calibrate_hybrid_multisim.py --n-trials 10 --total-trials 50 --worker-id $SLURM_ARRAY_TASK_ID)
"""
import sys
import logging
import argparse
import os
from datetime import datetime

# Parse command line arguments FIRST
parser = argparse.ArgumentParser(description='Run hybrid calibration with MultiSim')
parser.add_argument('--n-trials', type=int, default=50,
                    help='Number of trials for THIS worker to run (default: 50)')
parser.add_argument('--total-trials', type=int, default=None,
                    help='Total trials across ALL workers (default: same as n-trials)')
parser.add_argument('--n-reps', type=int, default=20,
                    help='Number of simulation replicates per trial (default: 20)')
parser.add_argument('--n-jobs', type=int, default=1,
                    help='Number of parallel trials via Optuna study.optimize (default: 1, use -1 for all CPUs)')
parser.add_argument('--worker-id', type=str, default=None,
                    help='Worker ID for logging (default: auto-generated)')
parser.add_argument('--n-cpus-per-trial', type=int, default=None,
                    help='Number of CPUs for MultiSim parallelization within each trial (default: all available)')
parser.add_argument('--db-path', type=str, default='rota_hybrid_multisim.db',
                    help='Path to Optuna database (default: rota_hybrid_multisim.db)')
args = parser.parse_args()

# Set worker ID
if args.worker_id is None:
    # Try to get SLURM array task ID if running on SLURM
    args.worker_id = os.environ.get('SLURM_ARRAY_TASK_ID',
                                     os.environ.get('PBS_ARRAYID',
                                     datetime.now().strftime("%H%M%S")))

# Configure logging FIRST before any other imports
log_file = f'calibrate_hybrid_multisim_worker{args.worker_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

logger.info("="*80)
logger.info("CALIBRATION SCRIPT STARTED")
logger.info(f"Worker ID: {args.worker_id}")
logger.info(f"Log file: {log_file}")
logger.info(f"Trials for this worker: {args.n_trials}")
if args.total_trials:
    logger.info(f"Total trials across all workers: {args.total_trials}")
logger.info(f"Replicates per trial: {args.n_reps}")
logger.info(f"Parallel jobs (n_jobs): {args.n_jobs if args.n_jobs != -1 else 'all CPUs'}")
logger.info(f"CPUs per trial: {args.n_cpus_per_trial if args.n_cpus_per_trial else 'all available'}")
logger.info(f"Database: {args.db_path}")
logger.info(f"Python: {sys.version}")
logger.info("="*80)

problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)
    logger.info(f"Removed {problem_path} from sys.path")

logger.info("Importing packages...")
try:
    import numpy as np
    logger.info("  - numpy imported")
    import sciris as sc
    logger.info("  - sciris imported")
    import starsim as ss
    logger.info(f"  - starsim imported (v{ss.__version__})")
    import rotasim as rs
    logger.info("  - rotasim imported")
    import optuna
    logger.info(f"  - optuna imported (v{optuna.__version__})")
    import json
    logger.info("  - json imported")
except Exception as e:
    logger.error(f"Failed to import packages: {e}", exc_info=True)
    sys.exit(1)

logger.info("Setting up paths...")
try:
    thisdir = sc.thispath(__file__)
    logger.info(f"  - thisdir: {thisdir}")
    sys.path.insert(0, str(thisdir))
    process_incidence_uk_age = sc.importbypath(thisdir / 'process_incidence_uk_age.py')
    logger.info("  - process_incidence_uk_age imported")
except Exception as e:
    logger.error(f"Failed to set up paths: {e}", exc_info=True)
    sys.exit(1)

logger.info("=" * 80)
logger.info("HYBRID CALIBRATION WITH MULTISIM - CLUSTER-PARALLEL VERSION")
logger.info("=" * 80)
logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info("\nMultiSim Configuration:")
logger.info(f"  - {args.n_reps} replicates per trial (parallelized WITHIN trial)")
logger.info("  - Median GOF across replicates for parameter selection")
logger.info("  - Multiple workers can run trials in parallel (coordinated via Optuna DB)")
logger.info("\nGOF Formula:")
logger.info("  GOF = 10 * GOF_age + GOF_incidence")
logger.info("  GOF_incidence = (log(target) - log(model))^2")
logger.info("  GOF_age = sum((proportion_target - proportion_model)^2)")
logger.info("\nCalibrating 8 parameters:")
logger.info("  1. reporting_rate")
logger.info("  2. base_beta")
logger.info("  3-5. beta0, beta1, beta2 (age symptom model)")
logger.info("  6-8. sus_after_1, sus_after_2, sus_after_3plus (fitted immunity)")
logger.info("=" * 80)


class HybridCalibrationMultiSim(ss.Calibration):
    """
    Calibration combining age-based symptoms with fitted immunity using MultiSim
    """

    def __init__(self, sim, data, calib_pars=None, total_trials=50, n_reps=20, n_jobs=6, n_cpus=None, debug=False, **kwargs):
        calib_pars_dict = calib_pars if calib_pars else {}
        super().__init__(sim=sim, calib_pars=calib_pars_dict, **kwargs)
        self.data = data
        self.target_incidence, self.target_age_distribution = data
        self.total_trials = total_trials
        self.n_reps = n_reps
        self.n_jobs = n_jobs  # Number of parallel trials (passed to study.optimize)
        self.n_cpus = n_cpus  # CPUs for MultiSim parallelization within each trial
        self.debug = debug

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None, n_reps=None):
        """Run simulation(s) with given parameters

        If n_reps > 1, returns a MultiSim object with multiple replicates
        If n_reps == 1, returns a single Sim object
        """
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

        if trial is not None:
            logger.info(f"\nTrial {trial}:")
            logger.info(f"  Age params: beta0={beta0:.4f}, beta1={beta1:.4f}, beta2={beta2:.4f}")
            logger.info(f"  Immunity: sus_1={sus_after_1:.3f}, sus_2={sus_after_2:.3f}, sus_3+={sus_after_3plus:.3f}")
            logger.info(f"  Other: reporting={reporting_rate:.6f}, beta={base_beta:.4f}")

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

            return sim

        # For multiple replicates, create list of sims with different seeds
        # Generate random seeds for each replicate
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

            # Initialize adult immunity
            immunity_connector.initialize_immunity(
                min_age=18, max_age=125, min_exposures=5, max_exposures=15
            )

            sims.append(sim_copy)

        # Create MultiSim from initialized sims
        ms = ss.MultiSim(sims=sims)

        return ms

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

    def compute_gof_single(self, sim):
        """Compute goodness of fit for a single simulation

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

        return total_gof

    def compute_gof(self, sim_or_multisim):
        """Compute goodness of fit

        If input is a Sim, compute GOF directly
        If input is a MultiSim, compute GOF for each replicate and return MEDIAN
        """
        # Check if this is a MultiSim
        if isinstance(sim_or_multisim, ss.MultiSim):
            # Compute GOF for each replicate
            gof_values = []
            for sim in sim_or_multisim.sims:
                gof = self.compute_gof_single(sim)
                gof_values.append(gof)

            # Return median GOF across replicates
            median_gof = np.median(gof_values)

            logger.info(f"  GOF across {len(gof_values)} replicates:")
            logger.info(f"    Median: {median_gof:.4f}")
            logger.info(f"    Mean: {np.mean(gof_values):.4f} ± {np.std(gof_values):.4f}")
            logger.info(f"    Range: [{np.min(gof_values):.4f}, {np.max(gof_values):.4f}]")

            return median_gof
        else:
            # Single simulation
            return self.compute_gof_single(sim_or_multisim)

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
        """Run a single calibration trial with multiple replicates"""
        logger.info(f"Starting trial {trial.number}...")
        try:
            # Convert trial to simulation parameters
            sim_pars = self.trial_to_sim_pars(trial)
            logger.info(f"  Parameters generated for trial {trial.number}")

            # Run simulation (returns MultiSim with n_reps replicates)
            logger.info(f"  Initializing {self.n_reps} simulations...")
            sim_or_multisim = self.run_sim(calib_pars=None, sim_pars=sim_pars, trial=trial.number)
            logger.info(f"  Simulations initialized")

            # Run the simulation(s)
            logger.info(f"  Running simulations...")
            if isinstance(sim_or_multisim, ss.MultiSim):
                sim_or_multisim.run()
            else:
                sim_or_multisim.run()
            logger.info(f"  Simulations complete")

            # Compute goodness of fit (median across replicates if MultiSim)
            logger.info(f"  Computing GOF...")
            gof = self.compute_gof(sim_or_multisim)

            logger.info(f"Trial {trial.number}: Median GOF = {gof:.4f}")

            return gof
        except Exception as e:
            logger.error(f"Trial {trial.number} FAILED with error: {e}", exc_info=True)
            raise

    def calibrate(self, db_path='rota_hybrid_multisim.db'):
        """Run calibration using Optuna

        Args:
            db_path: Path to SQLite database for Optuna storage
        """
        logger.info("Creating/loading Optuna study...")
        try:
            # Create Optuna study (or load if exists)
            # Multiple workers can safely connect to the same database
            study = optuna.create_study(
                study_name='rota_hybrid_multisim',
                direction='minimize',
                storage=f'sqlite:///{db_path}',
                load_if_exists=True,  # Critical for multi-worker: load existing study
                sampler=optuna.samplers.TPESampler(seed=12345),
            )
            logger.info("  Optuna study created/loaded successfully")
            logger.info(f"  Storage: sqlite:///{db_path}")
            logger.info(f"  Study name: {study.study_name}")
            logger.info(f"  Existing trials in database: {len(study.trials)}")
        except Exception as e:
            logger.error(f"Failed to create Optuna study: {e}", exc_info=True)
            raise

        logger.info(f"\nThis worker will run up to {self.total_trials} trials")
        logger.info(f"Parallel trials (n_jobs): {self.n_jobs if self.n_jobs != -1 else 'all CPUs'}")
        logger.info(f"Each trial: {self.n_reps} replicates (parallelized within trial)")
        logger.info(f"CPUs per trial: {self.n_cpus if self.n_cpus else 'all available'}")
        logger.info("=" * 60)

        try:
            # Run trials with n_jobs for parallel execution via study.optimize
            # n_jobs=1: sequential (one trial at a time)
            # n_jobs>1: run N trials in parallel
            # n_jobs=-1: use all available CPUs for parallel trials
            study.optimize(self.run_trial, n_trials=self.total_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            logger.info("Calibration interrupted by user")
        except Exception as e:
            logger.error(f"Calibration failed: {e}", exc_info=True)
            raise

        logger.info("\n" + "=" * 60)
        logger.info(f"WORKER COMPLETE - Ran {len([t for t in study.trials if t.state.name == 'COMPLETE'])} trials")
        logger.info("=" * 60)

        return study


# Get target data
logger.info("Loading target data...")
try:
    target_incidence, target_age_distribution = process_incidence_uk_age.process_data()
    logger.info("  Target data loaded successfully")
except Exception as e:
    logger.error(f"Failed to load target data: {e}", exc_info=True)
    sys.exit(1)

logger.info("\nTarget Data:")
logger.info(f"  Incidence: {target_incidence:.2f} per 100,000")
logger.info("  Age distribution:")
for i, label in enumerate(['0-11 months', '12-23 months', '24-59 months', '5+ years']):
    logger.info(f"    {label:<15}: {target_age_distribution.proportion.iloc[i]*100:>6.2f}%")

# Create base simulation
logger.info("Creating base simulation...")
try:
    analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    logger.info("  Analyzer and immunity connector created")

    people = ss.People(n_agents=100000, age_data=thisdir / 'uk_age_data.csv')
    logger.info("  People initialized (n=100,000)")

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
    logger.info("  Base simulation created successfully")
except Exception as e:
    logger.error(f"Failed to create base simulation: {e}", exc_info=True)
    sys.exit(1)

# Create calibration with MultiSim
logger.info("Creating calibration object...")
try:
    calib = HybridCalibrationMultiSim(
        sim=sim,
        data=(target_incidence, target_age_distribution),
        total_trials=args.n_trials,  # Trials for THIS worker
        n_reps=args.n_reps,  # Replicates per trial
        n_jobs=args.n_jobs,  # Parallel trials via study.optimize
        n_cpus=args.n_cpus_per_trial,  # CPUs for within-trial parallelization
        debug=False,
    )
    logger.info("  Calibration object created")
except Exception as e:
    logger.error(f"Failed to create calibration object: {e}", exc_info=True)
    sys.exit(1)

# Run calibration
total_trials_msg = args.total_trials if args.total_trials else args.n_trials
logger.info("\n" + "=" * 80)
logger.info(f"Starting calibration: this worker will run up to {args.n_trials} trials")
logger.info(f"Total target across all workers: {total_trials_msg} trials")
logger.info(f"Each trial: {args.n_reps} replicates")
logger.info("=" * 80)

try:
    study = calib.calibrate(db_path=args.db_path)
except Exception as e:
    logger.error(f"Calibration failed: {e}", exc_info=True)
    logger.info(f"\nLog file saved to: {log_file}")
    sys.exit(1)

# Show best trial across ALL workers (reads from shared database)
logger.info("\n" + "=" * 60)
logger.info("RESULTS FROM SHARED DATABASE (all workers)")
logger.info("=" * 60)
logger.info(f"Total completed trials in database: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")

best_trial = study.best_trial
logger.info(f"\nBest Trial: #{best_trial.number}")
logger.info(f"Best GOF (median across {args.n_reps} replicates): {best_trial.value:.4f}")
logger.info("\nBest Parameters:")
for key, value in best_trial.params.items():
    if key.startswith('sus_'):
        protection = (1 - value) * 100
        logger.info(f"  {key:<20}: {value:.6f} ({protection:.2f}% protection)")
    else:
        logger.info(f"  {key:<20}: {value:.6f}")

# Save results to JSON (only save if this worker found the best trial or if specified)
logger.info("Saving results to JSON...")
try:
    completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
    results = {
        'completed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'worker_id': args.worker_id,
        'total_completed_trials': len(completed_trials),
        'n_reps_per_trial': args.n_reps,
        'best_trial_number': best_trial.number,
        'best_gof': best_trial.value,
        'best_params': best_trial.params,
        'target_incidence': target_incidence,
        'target_age_distribution': target_age_distribution.proportion.tolist(),
    }

    output_file = thisdir / f'calibration_multisim_results_worker{args.worker_id}.json'
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Results saved to: {output_file}")
except Exception as e:
    logger.error(f"Failed to save results: {e}", exc_info=True)

# Show top 5 trials (from all workers)
logger.info("\n" + "=" * 80)
logger.info("Top 5 Trials (from shared database):")
logger.info("=" * 80)
completed_trials = [t for t in study.trials if t.state.name == 'COMPLETE']
trials = sorted(completed_trials, key=lambda t: t.value if t.value is not None else float('inf'))
for i, trial in enumerate(trials[:5]):
    if trial.value is not None:
        logger.info(f"\n#{trial.number}: Median GOF = {trial.value:.4f}")
        logger.info(f"  reporting_rate={trial.params['reporting_rate']:.6f}, base_beta={trial.params['base_beta']:.4f}")
        logger.info(f"  beta0={trial.params['beta0']:.4f}, beta1={trial.params['beta1']:.4f}, beta2={trial.params['beta2']:.4f}")
        logger.info(f"  sus: {trial.params['sus_after_1']:.3f}/{trial.params['sus_after_2']:.3f}/{trial.params['sus_after_3plus']:.3f}")

logger.info("\n" + "=" * 80)
logger.info(f"WORKER {args.worker_id} DONE")
logger.info(f"Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
logger.info(f"Total trials completed (all workers): {len(completed_trials)}")
logger.info(f"Log file: {log_file}")
logger.info("=" * 80)
