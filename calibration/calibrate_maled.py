"""
MAL-ED birth-cohort calibration with MultiSim.

Mirrors calibrate_hybrid_multisim.py (same 8 parameters, same sim setup, same
Optuna+MultiSim machinery, same TPE sampler seed for reproducibility) but
targets MAL-ED incidence-by-age + age-at-first-infection per-site (Bangladesh
or Pakistan).

GOF (from process_incidence_maled.gof):
  GOF_inc   = sum over MAL-ED age bins of (log(IR_target) - log(IR_model))^2
  GOF_first = ((med_target - med_model)^2 + 0.5*((Q25_diff)^2 + (Q75_diff)^2)) / med_target^2
  GOF       = w_inc * GOF_inc + w_first * GOF_first   (equal weight by default)

Usage:
  python calibrate_maled.py --site bangladesh --n-trials 50 --n-reps 20
  python calibrate_maled.py --site pakistan   --n-trials 50 --n-reps 20

Multi-worker:
  python calibrate_maled.py --site bangladesh --n-trials 10 --total-trials 50 --worker-id 1
"""
import sys
import logging
import argparse
import os
from datetime import datetime

# -------- CLI --------
parser = argparse.ArgumentParser(description='Run MAL-ED calibration with MultiSim')
parser.add_argument('--site', type=str, required=True,
                    choices=['bangladesh', 'pakistan'],
                    help='MAL-ED site to calibrate against')
parser.add_argument('--n-trials', type=int, default=50,
                    help='Number of trials for THIS worker (default: 50)')
parser.add_argument('--total-trials', type=int, default=None,
                    help='Total trials across ALL workers (default: same as n-trials)')
parser.add_argument('--n-reps', type=int, default=20,
                    help='Replicates per trial (default: 20)')
parser.add_argument('--n-jobs', type=int, default=1,
                    help='Parallel trials via Optuna study.optimize (default: 1, -1 for all CPUs)')
parser.add_argument('--worker-id', type=str, default=None,
                    help='Worker ID for logging (default: auto-generated)')
parser.add_argument('--n-cpus-per-trial', type=int, default=None,
                    help='CPUs for MultiSim parallelization within each trial')
parser.add_argument('--db-path', type=str, default=None,
                    help='Optuna SQLite path (default: rota_maled_<site>.db)')
parser.add_argument('--w-inc', type=float, default=1.0,
                    help='GOF weight on incidence-by-age component (default: 1.0)')
parser.add_argument('--w-first', type=float, default=1.0,
                    help='GOF weight on first-infection-age component (default: 1.0)')
args = parser.parse_args()

if args.db_path is None:
    args.db_path = f'rota_maled_{args.site}.db'
if args.worker_id is None:
    args.worker_id = os.environ.get('SLURM_ARRAY_TASK_ID',
                     os.environ.get('PBS_ARRAYID',
                                    datetime.now().strftime('%H%M%S')))

# -------- Logging --------
log_file = f'calibrate_maled_{args.site}_worker{args.worker_id}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)
logger.info("=" * 80)
logger.info(f"MAL-ED CALIBRATION ({args.site.title()})")
logger.info(f"Worker ID: {args.worker_id}")
logger.info(f"Log file: {log_file}")
logger.info(f"Trials for this worker: {args.n_trials}")
if args.total_trials:
    logger.info(f"Total trials across all workers: {args.total_trials}")
logger.info(f"Replicates per trial: {args.n_reps}")
logger.info(f"Parallel jobs: {args.n_jobs if args.n_jobs != -1 else 'all CPUs'}")
logger.info(f"GOF weights: w_inc={args.w_inc}, w_first={args.w_first}")
logger.info(f"Database: {args.db_path}")
logger.info("=" * 80)

# Strip the parent path that pulls in a different rotasim install (same trick
# as calibrate_hybrid_multisim).
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

# -------- Imports --------
import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna
import json

thisdir = sc.thispath(__file__)
sys.path.insert(0, str(thisdir))
process_incidence_maled = sc.importbypath(thisdir / 'process_incidence_maled.py')

# -------- Targets --------
logger.info(f"Loading MAL-ED targets for {args.site}...")
TARGETS = process_incidence_maled.load_targets(args.site)
logger.info("Target IR by age (per 100 person-months):")
for bin_label in process_incidence_maled.MALED_AGE_BINS:
    ir = TARGETS['ir_by_age'].loc[bin_label, 'IR']
    pt = TARGETS['ir_by_age'].loc[bin_label, 'PT']
    cs = TARGETS['ir_by_age'].loc[bin_label, 'cases']
    logger.info(f"  {bin_label:<8}  cases={cs:>3}  PT={pt:>5}  IR={ir:.3f}")
fi = TARGETS['first_infection']
logger.info(f"Target first-infection quartiles (months): "
            f"Q25={fi['q25']:.2f}, median={fi['median']:.2f}, Q75={fi['q75']:.2f} "
            f"(n_events={fi['n_events']}/{fi['n_total']})")

# Calibration window matches calibrate_hybrid_multisim: years 5-10.
CAL_WINDOW = (5.0, 10.0)
CAL_WINDOW_MONTHS = (CAL_WINDOW[1] - CAL_WINDOW[0]) * 12.0


# -------- Calibration class --------
class MALEDCalibration(ss.Calibration):
    def __init__(self, sim, targets, total_trials=50, n_reps=20, n_jobs=1, n_cpus=None,
                 w_inc=1.0, w_first=1.0, **kwargs):
        super().__init__(sim=sim, calib_pars={}, **kwargs)
        self.targets = targets
        self.total_trials = total_trials
        self.n_reps = n_reps
        self.n_jobs = n_jobs
        self.n_cpus = n_cpus
        self.w_inc = w_inc
        self.w_first = w_first

    def _build_sim(self, sim_pars, rand_seed=None):
        """Make an initialised sim with the given parameter overrides."""
        sim = sc.dcp(self.sim)
        if rand_seed is not None:
            sim.pars.rand_seed = rand_seed

        # Transmission scaling.
        sim.pars.base_beta = sim_pars['base_beta']
        for disease in sim.pars.diseases:
            if isinstance(disease, rs.Rotavirus):
                disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

        # Stash params on the sim for sim_to_summary to read later.
        sim._reporting_rate  = sim_pars['reporting_rate']
        sim._beta0           = sim_pars['beta0']
        sim._beta1           = sim_pars['beta1']
        sim._beta2           = sim_pars['beta2']

        sim.init()

        # Fixed-susceptibility immunity (same as hybrid UK calibration).
        ic = sim.connectors.rotaimmunityconnector
        ic.pars['use_fixed_susceptibility'] = True
        ic.pars['sus_after_1']     = sim_pars['sus_after_1']
        ic.pars['sus_after_2']     = sim_pars['sus_after_2']
        ic.pars['sus_after_3plus'] = sim_pars['sus_after_3plus']
        ic.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)

        return sim

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None, n_reps=None):
        if n_reps is None:
            n_reps = self.n_reps
        sim_pars = sim_pars or {}

        if trial is not None:
            logger.info(f"\nTrial {trial}:")
            logger.info(f"  Age params: beta0={sim_pars['beta0']:.4f}, beta1={sim_pars['beta1']:.4f}, beta2={sim_pars['beta2']:.4f}")
            logger.info(f"  Immunity: sus_1={sim_pars['sus_after_1']:.3f}, sus_2={sim_pars['sus_after_2']:.3f}, sus_3+={sim_pars['sus_after_3plus']:.3f}")
            logger.info(f"  Other: reporting={sim_pars['reporting_rate']:.6f}, beta={sim_pars['base_beta']:.4f}")

        if n_reps == 1:
            return self._build_sim(sim_pars)

        rand_seeds = np.random.randint(0, int(1e6), n_reps)
        sims = [self._build_sim(sim_pars, rand_seed=int(s)) for s in rand_seeds]
        return ss.MultiSim(sims=sims)

    def sim_to_summary(self, sim):
        """Run process_incidence_maled.process_model on this sim's analyzer output."""
        df = sim.analyzers['infectedstrainstats'].to_df()
        pt = process_incidence_maled.compute_person_months_steady_state(
            ages_years=sim.people.age.values,
            window_months=CAL_WINDOW_MONTHS,
        )
        return process_incidence_maled.process_model(
            df,
            person_months_by_bin=pt,
            symptom_model='age_and_infection_simple',
            beta0=sim._beta0, beta1=sim._beta1, beta2=sim._beta2,
            reporting_rate=sim._reporting_rate,
            calibration_window=CAL_WINDOW,
            censor_at_months=36.0,
        )

    def compute_gof_single(self, sim):
        model_out = self.sim_to_summary(sim)
        g = process_incidence_maled.gof(model_out, self.targets,
                                        w_inc=self.w_inc, w_first=self.w_first)
        return g['gof'], g, model_out

    def compute_gof(self, sim_or_multisim):
        if isinstance(sim_or_multisim, ss.MultiSim):
            gofs, breakdowns = [], []
            for sim in sim_or_multisim.sims:
                gof_total, g, _ = self.compute_gof_single(sim)
                gofs.append(gof_total)
                breakdowns.append(g)
            median_gof = float(np.median(gofs))
            median_inc = float(np.median([b['gof_incidence']        for b in breakdowns]))
            median_fi  = float(np.median([b['gof_first_infection'] for b in breakdowns]))
            logger.info(f"  GOF across {len(gofs)} replicates:")
            logger.info(f"    median total = {median_gof:.4f} "
                        f"(inc={median_inc:.4f}, first_inf={median_fi:.4f})")
            logger.info(f"    mean total   = {np.mean(gofs):.4f} ± {np.std(gofs):.4f}")
            logger.info(f"    range total  = [{np.min(gofs):.4f}, {np.max(gofs):.4f}]")
            return median_gof
        gof_total, _, _ = self.compute_gof_single(sim_or_multisim)
        return gof_total

    def trial_to_sim_pars(self, trial):
        # Same 8-parameter space and same monotonicity constraint as the UK hybrid fit.
        reporting_rate  = trial.suggest_float('reporting_rate', 0.0001, 0.01, log=True)
        base_beta       = trial.suggest_float('base_beta',      0.05, 0.5,    log=True)
        beta0           = trial.suggest_float('beta0',          -5.0, 2.0)
        beta1           = trial.suggest_float('beta1',          -1.0, 1.0)
        beta2           = trial.suggest_float('beta2',          -0.5, 0.5)
        sus_after_3plus = trial.suggest_float('sus_after_3plus', 0.1, 1.0)
        sus_after_2     = trial.suggest_float('sus_after_2',     sus_after_3plus, 1.0)
        sus_after_1     = trial.suggest_float('sus_after_1',     sus_after_2,     1.0)
        return dict(
            reporting_rate=reporting_rate, base_beta=base_beta,
            beta0=beta0, beta1=beta1, beta2=beta2,
            sus_after_1=sus_after_1, sus_after_2=sus_after_2,
            sus_after_3plus=sus_after_3plus,
        )

    def trial_pars_to_sim_pars(self, trial_pars=None, which='best'):
        return trial_pars

    def run_trial(self, trial):
        logger.info(f"Starting trial {trial.number}...")
        try:
            sim_pars = self.trial_to_sim_pars(trial)
            sim_or_ms = self.run_sim(sim_pars=sim_pars, trial=trial.number)
            sim_or_ms.run()
            gof_total = self.compute_gof(sim_or_ms)
            logger.info(f"Trial {trial.number}: Median GOF = {gof_total:.4f}")
            return gof_total
        except Exception as e:
            logger.error(f"Trial {trial.number} FAILED: {e}", exc_info=True)
            raise

    def calibrate(self, db_path):
        study = optuna.create_study(
            study_name=f'rota_maled_{args.site}',
            direction='minimize',
            storage=f'sqlite:///{db_path}',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=12345),
        )
        logger.info(f"Optuna study loaded ({study.study_name}, {len(study.trials)} existing trials)")
        try:
            study.optimize(self.run_trial, n_trials=self.total_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            logger.info("Calibration interrupted by user")
        return study


# -------- Base simulation (mirrors UK setup) --------
logger.info("Creating base simulation...")
analyzer = rs.InfectedStrainStats(use_infection_based_severity=False, constant_severity=0.2)
immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
# Reuse UK age data file for now; switch to a MAL-ED-specific demographic file later if needed.
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
logger.info("Base simulation created")

# -------- Run --------
calib = MALEDCalibration(
    sim=sim, targets=TARGETS,
    total_trials=args.n_trials, n_reps=args.n_reps,
    n_jobs=args.n_jobs, n_cpus=args.n_cpus_per_trial,
    w_inc=args.w_inc, w_first=args.w_first,
)
study = calib.calibrate(db_path=args.db_path)

# -------- Summarise --------
completed = [t for t in study.trials if t.state.name == 'COMPLETE']
logger.info("=" * 60)
logger.info(f"RESULTS ({args.site.title()})")
logger.info("=" * 60)
logger.info(f"Total completed trials: {len(completed)}")

best = study.best_trial
logger.info(f"\nBest trial: #{best.number}")
logger.info(f"Best GOF (median across {args.n_reps} replicates): {best.value:.4f}")
for key, value in best.params.items():
    if key.startswith('sus_'):
        logger.info(f"  {key:<20}: {value:.6f} ({(1-value)*100:.2f}% protection)")
    else:
        logger.info(f"  {key:<20}: {value:.6f}")

results = {
    'site': args.site,
    'completed_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'worker_id': args.worker_id,
    'total_completed_trials': len(completed),
    'n_reps_per_trial': args.n_reps,
    'best_trial_number': best.number,
    'best_gof': best.value,
    'best_params': best.params,
    'target_ir_by_age': TARGETS['ir_by_age']['IR'].to_dict(),
    'target_first_infection': TARGETS['first_infection'],
    'gof_weights': {'w_inc': args.w_inc, 'w_first': args.w_first},
}
out_path = thisdir / f'calibration_maled_{args.site}_worker{args.worker_id}.json'
with open(out_path, 'w') as f:
    json.dump(results, f, indent=2)
logger.info(f"Results saved to: {out_path}")

logger.info("\nTop 5 trials:")
sorted_trials = sorted(completed, key=lambda t: t.value if t.value is not None else float('inf'))
for i, t in enumerate(sorted_trials[:5]):
    logger.info(f"  #{t.number}: GOF={t.value:.4f}  "
                f"beta={t.params['base_beta']:.3f}  rep={t.params['reporting_rate']:.5f}  "
                f"sus=[{t.params['sus_after_1']:.2f},{t.params['sus_after_2']:.2f},{t.params['sus_after_3plus']:.2f}]")

logger.info("\n" + "=" * 80)
logger.info(f"WORKER {args.worker_id} DONE ({args.site}, {datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
logger.info("=" * 80)
