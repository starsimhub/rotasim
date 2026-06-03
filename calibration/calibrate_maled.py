"""
MAL-ED birth-cohort calibration with multiprocessing.Pool (spawn).

Same calibration target as before (per-site IR-by-age + first-infection
quartiles), same 7-parameter Optuna+TPE search, but workers are now spawned
fresh per replicate rather than forked from a parent that pre-built 20 sims.

Why the change: the previous MultiSim approach built 20 sim objects in the
parent process before forking, so each worker inherited all 20 via CoW.
That hit a memory ceiling (~340 GB) and got OOM-killed on shared VMs. With
spawn-based pool, each worker starts clean and builds only the one sim it
runs (~5-8 GB per worker, ~50-80 GB total for n_reps=10).

Usage:
  python calibrate_maled.py --site bangladesh --n-trials 50 --n-reps 20
  python calibrate_maled.py --site bangladesh --smoke

Multi-worker (Optuna-coordinated):
  python calibrate_maled.py --site bangladesh --n-trials 10 --total-trials 50 --worker-id 1
"""
import sys
import os
import json
import logging
import argparse
from datetime import datetime
from multiprocessing import get_context

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs
import optuna

# ---------------------------------------------------------------------------
# Module-level setup that workers also need.
# ---------------------------------------------------------------------------
# Strip an alternate sibling install of rotasim from sys.path if present
# (same trick as the previous version).
_PROBLEM_PATH = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if _PROBLEM_PATH in sys.path:
    sys.path.remove(_PROBLEM_PATH)

thisdir = sc.thispath(__file__)
if str(thisdir) not in sys.path:
    sys.path.insert(0, str(thisdir))
process_incidence_maled = sc.importbypath(thisdir / 'process_incidence_maled.py')

# MAL-ED reporting is ~100% (intensive TAC sampling), so we keep both the
# reporting filter and severity filter at 1.0 -- the only thing between
# infections and observed symptomatic cases is the age-symptom probability.
FIXED_REPORTING_RATE    = 1.0
FIXED_CONSTANT_SEVERITY = 1.0

# Site-specific demographics (modern, MAL-ED enrollment era ~2011-2014).
SITE_DEMOGRAPHICS = {
    'bangladesh': dict(birth_rate=19, death_rate=6),
    'pakistan':   dict(birth_rate=27, death_rate=7),
}

MALED_SITES = list(SITE_DEMOGRAPHICS.keys())


# ---------------------------------------------------------------------------
# Worker function: runs in a spawned subprocess. Must be picklable and
# self-sufficient. Builds its own sim from `sim_config`, applies trial params,
# runs, returns the small `model_out` dict back to the parent.
# ---------------------------------------------------------------------------
def _run_one_replicate(args):
    sim_config, sim_pars, rand_seed, cal_window = args

    # Build base sim from picklable config (no pre-built sim objects crossing
    # the process boundary -- spawn cannot pickle a fully-initialised Sim
    # anyway, and even if it could, this is the whole point of the refactor).
    analyzer = rs.InfectedStrainStats(
        use_infection_based_severity=False,
        constant_severity=sim_config['constant_severity'],
    )
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    people = ss.People(n_agents=sim_config['n_agents'],
                       age_data=sim_config['age_data_path'])
    sim = rs.Sim(
        n_agents=sim_config['n_agents'],
        start=sim_config['start'],
        stop=sim_config['stop'],
        verbose=False,
        scenario='single',
        people=people,
        analyzers=[analyzer],
        networks=ss.RandomNet(n_contacts=sim_config['n_contacts']),
        demographics=[
            ss.Births(birth_rate=ss.peryear(sim_config['birth_rate'])),
            ss.Deaths(death_rate=ss.peryear(sim_config['death_rate'])),
        ],
        interventions=[],
        connectors=[immunity_connector],
        rand_seed=rand_seed,
    )

    # Apply trial parameters (transmission + per-disease beta scaling).
    sim.pars.base_beta = sim_pars['base_beta']
    for disease in sim.pars.diseases:
        if isinstance(disease, rs.Rotavirus):
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)

    sim.init()

    # Configure fitted-susceptibility immunity after init.
    ic = sim.connectors.rotaimmunityconnector
    ic.pars['use_fixed_susceptibility'] = True
    ic.pars['sus_after_1']     = sim_pars['sus_after_1']
    ic.pars['sus_after_2']     = sim_pars['sus_after_2']
    ic.pars['sus_after_3plus'] = sim_pars['sus_after_3plus']
    # Maternal immunity (passive protection from mother, decays with age).
    # Defaults to OFF (0.0 efficacy) if not in sim_pars -- lets us re-run old
    # trials that pre-date this parameter via evaluate_trial.py.
    ic.pars['maternal_immunity_efficacy']  = sim_pars.get('maternal_immunity_efficacy', 0.0)
    ic.pars['maternal_immunity_half_life'] = ss.days(sim_pars.get('maternal_immunity_half_life_days', 90.0))
    # Slow (breastfeeding) maternal component; defaults to OFF for trials predating it.
    ic.pars['maternal_immunity_efficacy_slow']  = sim_pars.get('maternal_immunity_efficacy_slow', 0.0)
    ic.pars['maternal_immunity_half_life_slow'] = ss.days(sim_pars.get('maternal_immunity_half_life_slow_days', 270.0))
    ic.initialize_immunity(min_age=18, max_age=125,
                           min_exposures=5, max_exposures=15)

    sim.run()

    # Reduce the sim's analyzer output to the small model_out dict we need
    # downstream. Everything large stays inside the worker process and is
    # released when the worker is reused for the next task or torn down.
    df = sim.analyzers['infectedstrainstats'].to_df()
    pt = process_incidence_maled.compute_person_months_steady_state(
        ages_years=sim.people.age.values,
        window_months=(cal_window[1] - cal_window[0]) * 12.0,
    )
    model_out = process_incidence_maled.process_model(
        df,
        person_months_by_bin=pt,
        symptom_model='age_and_infection_simple',
        beta0=sim_pars['beta0'],
        beta1=sim_pars['beta1'],
        beta2=sim_pars['beta2'],
        reporting_rate=sim_config['reporting_rate'],
        calibration_window=cal_window,
        censor_at_months=36.0,
        p_asymp_detect=sim_config.get('p_asymp_detect', 0.4),
    )
    return model_out


# ---------------------------------------------------------------------------
# Calibration object. No longer subclasses ss.Calibration -- we use Optuna
# directly. The class just holds the configuration and per-trial state.
# ---------------------------------------------------------------------------
class MALEDCalibration:
    def __init__(self, sim_config, targets, cal_window, total_trials, n_reps,
                 n_jobs, n_cpus, w_inc, w_first, db_path, study_name, logger,
                 fit_target='joint', p_asymp_detect=0.4):
        self.sim_config     = sim_config
        self.targets        = targets
        self.cal_window     = cal_window
        self.total_trials   = total_trials
        self.n_reps         = n_reps
        self.n_jobs         = n_jobs
        self.n_cpus         = n_cpus
        self.w_inc          = w_inc
        self.w_first        = w_first
        self.db_path        = db_path
        self.study_name     = study_name
        self.logger         = logger
        self.fit_target     = fit_target
        self.p_asymp_detect = p_asymp_detect

    def _trial_to_sim_pars(self, trial):
        # 11-parameter space (reporting_rate fixed at 1.0; two-phase maternal immunity).
        # Monotonicity: sus_after_3plus <= sus_after_2 <= sus_after_1.
        base_beta       = trial.suggest_float('base_beta',      0.05, 0.5,    log=True)
        beta0           = trial.suggest_float('beta0',          -5.0, 2.0)
        beta1           = trial.suggest_float('beta1',          -1.0, 1.0)
        beta2           = trial.suggest_float('beta2',          -0.5, 0.5)
        sus_after_3plus = trial.suggest_float('sus_after_3plus', 0.1, 1.0)
        sus_after_2     = trial.suggest_float('sus_after_2',     sus_after_3plus, 1.0)
        sus_after_1     = trial.suggest_float('sus_after_1',     sus_after_2, 1.0)
        # Two-phase maternal immunity:
        #   FAST = transplacental IgG (high efficacy, short half-life ~weeks-2mo)
        #   SLOW = breastfeeding IgA  (can be weaker, longer half-life ~4-18mo)
        # Half-life ranges are disjoint (fast <= 90d < slow) so the two components
        # can't swap roles, keeping the decomposition identifiable. Slow efficacy
        # range starts at 0 so the optimizer can fall back to single-phase.
        maternal_immunity_efficacy            = trial.suggest_float('maternal_immunity_efficacy',            0.5, 0.99)
        maternal_immunity_half_life_days      = trial.suggest_float('maternal_immunity_half_life_days',     15.0, 90.0)
        maternal_immunity_efficacy_slow       = trial.suggest_float('maternal_immunity_efficacy_slow',       0.0, 0.8)
        maternal_immunity_half_life_slow_days = trial.suggest_float('maternal_immunity_half_life_slow_days', 120.0, 540.0)
        return dict(
            base_beta=base_beta,
            beta0=beta0, beta1=beta1, beta2=beta2,
            sus_after_1=sus_after_1, sus_after_2=sus_after_2,
            sus_after_3plus=sus_after_3plus,
            maternal_immunity_efficacy=maternal_immunity_efficacy,
            maternal_immunity_half_life_days=maternal_immunity_half_life_days,
            maternal_immunity_efficacy_slow=maternal_immunity_efficacy_slow,
            maternal_immunity_half_life_slow_days=maternal_immunity_half_life_slow_days,
        )

    def _run_replicates(self, sim_pars):
        """Spawn n_reps workers; each builds and runs one sim; collect summaries."""
        seeds = np.random.randint(0, int(1e6), self.n_reps).tolist()
        # Pass p_asymp_detect into sim_config so the worker uses the same value.
        cfg = dict(self.sim_config, p_asymp_detect=self.p_asymp_detect)
        args_list = [(cfg, sim_pars, int(s), self.cal_window) for s in seeds]

        n_workers = min(self.n_reps, self.n_cpus or self.n_reps)
        ctx = get_context('spawn')
        with ctx.Pool(processes=n_workers) as pool:
            model_outs = pool.map(_run_one_replicate, args_list)
        return model_outs

    def _aggregate(self, model_outs):
        """Compute per-rep GOF, return (median_total, breakdown, per-rep details)."""
        gofs, breakdowns = [], []
        for mo in model_outs:
            g = process_incidence_maled.gof(mo, self.targets,
                                            w_inc=self.w_inc, w_first=self.w_first,
                                            fit_target=self.fit_target)
            gofs.append(g['gof'])
            breakdowns.append(g)
        return dict(
            median_total = float(np.median(gofs)),
            mean_total   = float(np.mean(gofs)),
            std_total    = float(np.std(gofs)),
            min_total    = float(np.min(gofs)),
            max_total    = float(np.max(gofs)),
            median_inc   = float(np.median([b['gof_incidence']        for b in breakdowns])),
            median_first = float(np.median([b['gof_first_infection'] for b in breakdowns])),
            per_rep_gofs  = gofs,
            per_rep_inc   = [float(b['gof_incidence'])       for b in breakdowns],
            per_rep_first = [float(b['gof_first_infection']) for b in breakdowns],
        )

    def _store_per_rep(self, trial, agg, model_outs):
        """Persist full per-replicate model output into the trial's user_attrs so
        downstream analysis (e.g. error bars on IR-by-age and first-infection
        quartiles) never needs to re-run the sims. Small payload: n_reps x 4 IR
        values + n_reps x 3 quartiles + per-rep GOF components."""
        bins = process_incidence_maled.MALED_AGE_BINS
        trial.set_user_attr('age_bins', bins)
        trial.set_user_attr(
            'ir_by_age_per_rep',
            [[float(mo['ir_by_age'].loc[b, 'IR']) for b in bins] for mo in model_outs])
        trial.set_user_attr(
            'first_inf_per_rep',
            [[float(mo['first_infection'][k]) for k in ('q25', 'median', 'q75')]
             for mo in model_outs])
        trial.set_user_attr('per_rep_gof_inc', agg['per_rep_inc'])
        trial.set_user_attr('per_rep_gof_first', agg['per_rep_first'])
        trial.set_user_attr('per_rep_gof_total', agg['per_rep_gofs'])

    def _log_summary(self, agg, model_outs, sim_pars, trial_num):
        # Median IR per bin and median first-inf quartile across reps.
        bins = process_incidence_maled.MALED_AGE_BINS
        median_ir = {b: float(np.median([mo['ir_by_age'].loc[b, 'IR']
                                          for mo in model_outs])) for b in bins}
        median_q  = {k: float(np.median([mo['first_infection'][k] for mo in model_outs]))
                     for k in ('q25', 'median', 'q75')}
        self.logger.info(f"  GOF across {len(model_outs)} replicates:")
        self.logger.info(f"    median total = {agg['median_total']:.4f} "
                         f"(inc={agg['median_inc']:.4f}, first_inf={agg['median_first']:.4f})")
        self.logger.info(f"    mean total   = {agg['mean_total']:.4f} ± {agg['std_total']:.4f}")
        self.logger.info(f"    range total  = [{agg['min_total']:.4f}, {agg['max_total']:.4f}]")
        self.logger.info(f"  Median model IR (per 100 PM): "
                         + ", ".join(f"{b}={median_ir[b]:.3f}" for b in bins))
        self.logger.info(f"  Median model first-inf (mo): "
                         f"Q25={median_q['q25']:.2f}, "
                         f"med={median_q['median']:.2f}, "
                         f"Q75={median_q['q75']:.2f}")

    def run_trial(self, trial):
        self.logger.info(f"Starting trial {trial.number}...")
        try:
            sim_pars = self._trial_to_sim_pars(trial)
            self.logger.info(f"\nTrial {trial.number}:")
            self.logger.info(f"  Age params: beta0={sim_pars['beta0']:.4f}, "
                             f"beta1={sim_pars['beta1']:.4f}, beta2={sim_pars['beta2']:.4f}")
            self.logger.info(f"  Immunity: sus_1={sim_pars['sus_after_1']:.3f}, "
                             f"sus_2={sim_pars['sus_after_2']:.3f}, "
                             f"sus_3+={sim_pars['sus_after_3plus']:.3f}")
            self.logger.info(f"  Maternal FAST: efficacy={sim_pars['maternal_immunity_efficacy']:.3f}, "
                             f"half_life={sim_pars['maternal_immunity_half_life_days']:.1f} days")
            self.logger.info(f"  Maternal SLOW: efficacy={sim_pars['maternal_immunity_efficacy_slow']:.3f}, "
                             f"half_life={sim_pars['maternal_immunity_half_life_slow_days']:.1f} days")
            self.logger.info(f"  Transmission: beta={sim_pars['base_beta']:.4f} "
                             f"(reporting fixed at {FIXED_REPORTING_RATE})")

            model_outs = self._run_replicates(sim_pars)
            agg = self._aggregate(model_outs)
            self._log_summary(agg, model_outs, sim_pars, trial.number)
            self._store_per_rep(trial, agg, model_outs)
            self.logger.info(f"Trial {trial.number}: Median GOF = {agg['median_total']:.4f}")
            return agg['median_total']
        except Exception as e:
            self.logger.error(f"Trial {trial.number} FAILED: {e}", exc_info=True)
            raise

    def calibrate(self):
        # n_startup_trials shrinks with prior trials so resumes don't waste cycles
        # re-sampling the same first random point each restart.
        try:
            existing = len(optuna.load_study(study_name=self.study_name,
                                              storage=f'sqlite:///{self.db_path}').trials)
        except KeyError:
            existing = 0
        n_startup = max(0, 10 - existing)

        study = optuna.create_study(
            study_name=self.study_name,
            direction='minimize',
            storage=f'sqlite:///{self.db_path}',
            load_if_exists=True,
            sampler=optuna.samplers.TPESampler(seed=12345, n_startup_trials=n_startup),
        )
        self.logger.info(f"Optuna study loaded ({study.study_name}, "
                         f"{len(study.trials)} existing trials, "
                         f"n_startup_trials for this worker={n_startup})")
        try:
            study.optimize(self.run_trial, n_trials=self.total_trials, n_jobs=self.n_jobs)
        except KeyboardInterrupt:
            self.logger.info("Calibration interrupted by user")
        return study


# ---------------------------------------------------------------------------
# Main: parses CLI, sets up logging, runs the calibration.
# Wrapped in __main__ guard so spawned workers can reimport this module
# without re-running argparse / logging / db creation.
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description='MAL-ED calibration (spawn-pool architecture)')
    parser.add_argument('--site', type=str, required=True, choices=MALED_SITES,
                        help='MAL-ED site to calibrate against')
    parser.add_argument('--n-trials', type=int, default=50)
    parser.add_argument('--total-trials', type=int, default=None)
    parser.add_argument('--n-reps', type=int, default=20)
    parser.add_argument('--n-jobs', type=int, default=1,
                        help='Parallel trials via Optuna study.optimize')
    parser.add_argument('--worker-id', type=str, default=None)
    parser.add_argument('--n-cpus-per-trial', type=int, default=None,
                        help='Max worker processes for the within-trial Pool')
    parser.add_argument('--db-path', type=str, default=None)
    parser.add_argument('--w-inc', type=float, default=1.0)
    parser.add_argument('--w-first', type=float, default=1.0)
    parser.add_argument('--fit-target', type=str, default='joint',
                        choices=['joint', 'symptomatic_ir', 'first_infection'],
                        help='Which GOF component(s) to optimize against. '
                             '"joint" = incidence + first-inf; '
                             '"symptomatic_ir" = only the per-age-bin IR target; '
                             '"first_infection" = only the age-at-first-detection target.')
    parser.add_argument('--p-asymp-detect', type=float, default=0.4,
                        help='Probability MAL-ED detects an asymptomatic infection '
                             'via monthly stool (default 0.4 = ~shedding/collection_interval).')
    parser.add_argument('--smoke', action='store_true',
                        help='Tiny end-to-end smoke test (5k agents, 5y sim, 1 trial, 1 rep).')
    args = parser.parse_args()

    if args.smoke:
        args.n_trials = 1
        args.n_reps   = 1
        args.n_jobs   = 1
        args.db_path  = f'rota_maled_smoke_{args.site}.db'

    if args.db_path is None:
        # Encode fit-target in default DB name so different modes don't share a study.
        suffix = '' if args.fit_target == 'joint' else f'_{args.fit_target}'
        args.db_path = f'rota_maled_{args.site}{suffix}.db'
    if args.worker_id is None:
        args.worker_id = os.environ.get('SLURM_ARRAY_TASK_ID',
                          os.environ.get('PBS_ARRAYID',
                                          datetime.now().strftime('%H%M%S')))

    log_file = (f'calibrate_maled_{args.site}_worker{args.worker_id}_'
                f'{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[logging.FileHandler(log_file), logging.StreamHandler(sys.stdout)],
    )
    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info(f"MAL-ED CALIBRATION ({args.site.title()}) — spawn-pool architecture")
    logger.info(f"Worker ID: {args.worker_id}")
    logger.info(f"Log file: {log_file}")
    logger.info(f"Trials for this worker: {args.n_trials}")
    if args.total_trials:
        logger.info(f"Total trials across all workers: {args.total_trials}")
    logger.info(f"Replicates per trial: {args.n_reps}")
    logger.info(f"Parallel trials: {args.n_jobs}")
    logger.info(f"CPUs per trial: {args.n_cpus_per_trial or args.n_reps}")
    logger.info(f"GOF weights: w_inc={args.w_inc}, w_first={args.w_first}")
    logger.info(f"Fit target: {args.fit_target}")
    logger.info(f"P(asymp detect by MAL-ED): {args.p_asymp_detect:.2f}")
    logger.info(f"Database: {args.db_path}")
    logger.info("=" * 80)

    # Targets
    targets = process_incidence_maled.load_targets(args.site)
    logger.info(f"Loaded MAL-ED targets for {args.site}")
    logger.info("Target IR by age (per 100 person-months):")
    for bin_label in process_incidence_maled.MALED_AGE_BINS:
        ir = targets['ir_by_age'].loc[bin_label, 'IR']
        pt = targets['ir_by_age'].loc[bin_label, 'PT']
        cs = targets['ir_by_age'].loc[bin_label, 'cases']
        logger.info(f"  {bin_label:<8}  cases={cs:>3}  PT={pt:>5}  IR={ir:.3f}")
    fi = targets['first_infection']
    logger.info(f"Target first-infection (mo): Q25={fi['q25']:.2f}, "
                f"med={fi['median']:.2f}, Q75={fi['q75']:.2f} "
                f"(n_events={fi['n_events']}/{fi['n_total']})")

    # Sim window
    if args.smoke:
        sim_start, sim_stop = '2003-01-01', '2008-01-01'
        n_agents = 5_000
        cal_window = (2.0, 5.0)
    else:
        sim_start, sim_stop = '2003-01-01', '2013-01-01'
        n_agents = 100_000
        cal_window = (5.0, 10.0)
    logger.info(f"Sim: n_agents={n_agents}, range={sim_start}..{sim_stop}, "
                f"calibration window (years)={cal_window}")

    # Demographics
    demo = SITE_DEMOGRAPHICS[args.site]
    logger.info(f"Demographics ({args.site}): birth_rate={demo['birth_rate']}/1000/y, "
                f"death_rate={demo['death_rate']}/1000/y")
    logger.info(f"Fixed reporting_rate={FIXED_REPORTING_RATE}, "
                f"constant_severity={FIXED_CONSTANT_SEVERITY}")

    # Picklable sim configuration that workers reconstruct from
    sim_config = dict(
        n_agents=n_agents,
        start=sim_start,
        stop=sim_stop,
        n_contacts=7,
        birth_rate=demo['birth_rate'],
        death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY,
        reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'),
    )

    suffix = '' if args.fit_target == 'joint' else f'_{args.fit_target}'
    study_name = f'rota_maled_{args.site}{suffix}'
    calib = MALEDCalibration(
        sim_config=sim_config,
        targets=targets,
        cal_window=cal_window,
        total_trials=args.n_trials,
        n_reps=args.n_reps,
        n_jobs=args.n_jobs,
        n_cpus=args.n_cpus_per_trial,
        w_inc=args.w_inc,
        w_first=args.w_first,
        db_path=args.db_path,
        study_name=study_name,
        logger=logger,
        fit_target=args.fit_target,
        p_asymp_detect=args.p_asymp_detect,
    )
    study = calib.calibrate()

    # Summarise
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
        'target_ir_by_age': targets['ir_by_age']['IR'].to_dict(),
        'target_first_infection': targets['first_infection'],
        'gof_weights': {'w_inc': args.w_inc, 'w_first': args.w_first},
    }
    out_path = thisdir / f'calibration_maled_{args.site}_worker{args.worker_id}.json'
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Results saved to: {out_path}")

    logger.info("\nTop 5 trials:")
    sorted_trials = sorted(completed, key=lambda t: t.value if t.value is not None else float('inf'))
    for t in sorted_trials[:5]:
        logger.info(f"  #{t.number}: GOF={t.value:.4f}  "
                    f"beta={t.params['base_beta']:.3f}  "
                    f"age_symp=[{t.params['beta0']:.2f},{t.params['beta1']:.2f},{t.params['beta2']:.2f}]  "
                    f"sus=[{t.params['sus_after_1']:.2f},{t.params['sus_after_2']:.2f},"
                    f"{t.params['sus_after_3plus']:.2f}]")

    logger.info("\n" + "=" * 80)
    logger.info(f"WORKER {args.worker_id} DONE ({args.site}, "
                f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')})")
    logger.info("=" * 80)


if __name__ == '__main__':
    main()
