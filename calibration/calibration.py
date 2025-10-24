"""
Define the calibration class
"""
import os
import numpy as np
import sciris as sc
import optuna as op
import matplotlib.pyplot as plt
import starsim as ss

# Local imports ... should tidy up later
thisdir = sc.thispath(__file__)
import rotasim as rs
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')


__all__ = ['Calibration', 'compute_gof']


def compute_gof(actual, predicted, normalize=True, use_frac=True, use_squared=False, eps=1e-9):
    """
    Calculate the goodness of fit. By default use normalized absolute error, but
    highly customizable. For example, mean squared error is equivalent to
    setting normalize=False, use_squared=True, as_scalar='mean'.

    Args:
        actual      (arr):   array of actual (data) points
        predicted   (arr):   corresponding array of predicted (model) points
        normalize   (bool):  whether to divide the values by the largest value in either series
        use_frac    (bool):  convert to fractional mismatches rather than absolute
        use_squared (bool):  square the mismatches
        eps         (float): to avoid divide-by-zero

    Returns:
        gofs (arr): array of goodness-of-fit values, or a single value if as_scalar is True

    **Examples**::

        x1 = np.cumsum(np.random.random(100))
        x2 = np.cumsum(np.random.random(100))

        e1 = compute_gof(x1, x2) # Default, normalized absolute error
        e2 = compute_gof(x1, x2, normalize=False, use_frac=False) # Fractional error
        e3 = compute_gof(x1, x2, normalize=False, use_squared=True, as_scalar='mean') # Mean squared error
    """

    # Handle inputs
    actual    = np.array(sc.dcp(actual), dtype=float)
    predicted = np.array(sc.dcp(predicted), dtype=float)

    # Key step -- calculate the mismatch!
    gofs = abs(np.array(actual) - np.array(predicted))

    if normalize and not use_frac:
        actual_max = abs(actual).max()
        if actual_max > 0:
            gofs /= actual_max

    if use_frac:
        if (actual<0).any() or (predicted<0).any():
            print('Warning: Calculating fractional errors for non-positive quantities is ill-advised!')
        else:
            # Divide by actual to get fractional error relative to observed data
            # This treats over- and under-estimation symmetrically
            gofs /= (actual + eps)

    return gofs


class Calibration(sc.prettyobj):
    """
    A class to handle calibration of RotaABM simulations. Uses the Optuna hyperparameter
    optimization library (optuna.org).

    Args:
        sim          (RotaABM): the simulation to calibrate
        data         (df)   : dataframe of the data to calibrate to
        calib_pars   (dict) : a dictionary of the parameters to calibrate of the format dict(key1=[best, low, high])
        n_trials     (int)  : the number of trials per worker
        n_workers    (int)  : the number of parallel workers (default: maximum number of available CPUs)
        total_trials (int)  : if n_trials is not supplied, calculate by dividing this number by n_workers
        weights      (dict) : the relative weights of each data source
        debug        (bool) : if True, do not run in parallel
        verbose      (bool) : whether to print details of the calibration

    Returns:
        A Calibration object
    """
    def __init__(self, sim, data, calib_pars, n_trials=None, n_workers=None, total_trials=100,
                 die=False, debug=False, verbose=True):

        # Handle run arguments
        name = 'rota'
        db_name = f'{name}.db'
        storage = f'sqlite:///{db_name}'
        if n_workers is None:
            n_workers = sc.cpu_count()
        if total_trials is not None:
            n_trials = int(np.ceil(total_trials/n_workers))
        kw = dict(n_trials=int(n_trials), n_workers=int(n_workers), debug=debug,
                  name=name, db_name=db_name, storage=storage)
        self.run_args = sc.objdict(kw)

        # Store calibration settings
        self.known_pars = ['reassortment_rate', 'rel_beta', 'reporting_rate', 'maternal_immunity_efficacy', 'maternal_immunity_half_life',
                          'homotypic_immunity_efficacy', 'partial_heterotypic_immunity_efficacy', 'complete_heterotypic_immunity_efficacy']

        # Handle other inputs
        self.sim        = sim
        # data should be a tuple of (overall_incidence, age_distribution)
        if isinstance(data, tuple):
            self.overall_incidence, self.age_distribution = data
        else:
            # Backwards compatibility: assume single dataframe with 'inci' column
            self.overall_incidence = data['inci'].mean()
            total = data['inci'].sum()
            self.age_distribution = sc.dataframe(dict(ages=data['ages'], proportion=data['inci']/total))
        self.calib_pars = calib_pars
        self.die        = die
        self.verbose    = verbose
        self.before_sim = None
        self.after_sim  = None
        self.calibrated = False
        return

    def run_sim(self, calib_pars=None, sim_pars=None, trial=None):
        """ Create and run a simulation """

        # Convert calib_pars (best, low, high) to a guess
        if calib_pars is not None:
            sim_pars = self.trial_to_sim_pars(calib_pars=calib_pars, trial=trial)

        # Update sim with new pars
        sim = self.translate_pars(sim_pars=sim_pars)

        # Run sim
        sim.run()
        return sim

    def calib_to_sim_pars(self):
        """ Pull out "best" from the list of calibration pars """
        sim_pars = sc.objdict()
        for par,(best,low,high) in self.calib_pars.items():
            sim_pars[par] = best
        return sim_pars

    def trial_to_sim_pars(self, calib_pars, trial):
        """ Take in an optuna trial and sample from pars """
        calib_pars = sc.mergedicts(calib_pars) # To allow None
        sim_pars = sc.objdict()
        for par, (best,low,high) in calib_pars.items():
            val = trial.suggest_float(par, low, high)
            sim_pars[par] = val
        return sim_pars

    def translate_pars(self, sim_pars):
        """ Take the nested dict of calibration pars and modify the sim """
        sim_pars = sc.mergedicts(sim_pars) # To allow None
        sim = sc.dcp(self.sim)

        # Store parameters to apply after initialization
        pars_to_apply = {}

        pars = list(sim_pars.keys())
        for par in pars:
            val = sim_pars.pop(par)
            if par in self.known_pars:
                pars_to_apply[par] = val
            else:
                errormsg = f'Do not know how to handle parameter "{par}"'
                raise NotImplementedError(errormsg)

        # Initialize the sim first so diseases and connectors are created
        sim.init()

        # Now apply the parameters after initialization
        for par, val in pars_to_apply.items():
            if par == 'rel_beta':
                # In V2, modify base_beta which affects all strains
                if hasattr(sim, '_base_beta'):
                    sim._base_beta = sim._base_beta * val
                # Also need to update disease betas
                if hasattr(sim, 'diseases'):
                    for disease in sim.diseases.values():
                        if hasattr(disease, 'pars') and hasattr(disease.pars, 'beta'):
                            # Multiply the beta by the relative factor
                            disease.pars.beta = disease.pars.beta * val
            elif par == 'reassortment_rate':
                # In V2, set on the RotaReassortmentConnector (called reassortment_prob in V2)
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaReassortmentConnector':
                            if hasattr(connector, 'pars') and hasattr(connector.pars, 'reassortment_prob'):
                                # Update the probability parameter - need to properly initialize it
                                new_dist = ss.bernoulli(p=val)
                                new_dist.init(sim.people, sim=sim)  # Initialize with people and sim
                                connector.pars.reassortment_prob = new_dist
                            break
            elif par == 'reporting_rate':
                # reporting_rate is applied during post-processing, not to sim
                # Store it as an attribute on sim for use in compute_fit
                sim._reporting_rate = val
            elif par == 'maternal_immunity_efficacy':
                # Set on RotaImmunityConnector
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaImmunityConnector':
                            connector.pars.maternal_immunity_efficacy = val
                            break
            elif par == 'maternal_immunity_half_life':
                # Set on RotaImmunityConnector (in days)
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaImmunityConnector':
                            connector.pars.maternal_immunity_half_life = val
                            break
            elif par == 'homotypic_immunity_efficacy':
                # Set on RotaImmunityConnector
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaImmunityConnector':
                            connector.pars.homotypic_immunity_efficacy = val
                            break
            elif par == 'partial_heterotypic_immunity_efficacy':
                # Set on RotaImmunityConnector
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaImmunityConnector':
                            connector.pars.partial_heterotypic_immunity_efficacy = val
                            break
            elif par == 'complete_heterotypic_immunity_efficacy':
                # Set on RotaImmunityConnector
                if hasattr(sim, 'connectors'):
                    for connector in sim.connectors.values():
                        if type(connector).__name__ == 'RotaImmunityConnector':
                            connector.pars.complete_heterotypic_immunity_efficacy = val
                            break
            else:
                setattr(sim, par, val) # Set the new value for other known pars

        return sim

    def compute_fit(self, sim, full=False):
        """
        Compute goodness-of-fit for both overall incidence and age distribution

        Strategy:
        - reporting_rate fits overall incidence magnitude
        - maternal_immunity and rel_beta fit age distribution shape

        Returns combined GOF that weights both objectives
        """
        # Get simulation results
        sim_overall_incidence, sim_age_distribution = self.sim_to_df(sim)

        # Handle case where simulation died out
        if sim_overall_incidence is None or sim_age_distribution is None or len(sim_age_distribution) == 0:
            penalty = 1e6
            if full:
                return penalty, penalty, penalty  # total, incidence_gof, age_dist_gof
            else:
                return penalty

        # 1. Compute GOF for overall incidence (single value)
        target_incidence = self.overall_incidence
        incidence_gof = abs(sim_overall_incidence - target_incidence) / (target_incidence + 1e-9)

        # 2. Compute GOF for age distribution (proportions)
        target_proportions = self.age_distribution.proportion.values
        sim_proportions = sim_age_distribution.proportion.values

        # Handle mismatched shapes
        if len(sim_proportions) != len(target_proportions):
            if len(sim_proportions) < len(target_proportions):
                padding = np.zeros(len(target_proportions) - len(sim_proportions))
                sim_proportions = np.concatenate([sim_proportions, padding])
            else:
                sim_proportions = sim_proportions[:len(target_proportions)]

        # Use sum of absolute differences for proportions (they sum to 1)
        age_dist_gof = np.abs(sim_proportions - target_proportions).sum()

        # Combined GOF: weight both objectives equally
        # Incidence GOF is already fractional (normalized)
        # Age dist GOF is sum of absolute proportion differences (max = 2 if completely wrong)
        # Normalize age_dist_gof to similar scale as incidence_gof
        total_gof = incidence_gof + age_dist_gof

        if full:
            return total_gof, incidence_gof, age_dist_gof
        else:
            return total_gof

    @staticmethod
    def sim_to_df(sim):
        """
        Convert the sim output to data format

        Returns:
            overall_incidence: float - overall incidence per 100k
            age_distribution: dataframe - proportions by age
        """
        # Extract infection data from InfectedStrainStats analyzer
        infected_analyzer = None
        for analyzer in sim.analyzers.values():
            if type(analyzer).__name__ == 'InfectedStrainStats':
                infected_analyzer = analyzer
                break

        if infected_analyzer is None:
            raise ValueError("InfectedStrainStats analyzer not found in simulation. Please add it to the sim.analyzers list.")

        # Get the infection events dataframe
        df = infected_analyzer.to_df()

        # Process the data using the process_incidence module
        # Returns (overall_incidence, age_distribution)
        overall_incidence, age_distribution = process_incidence.process_model(df)

        # Apply reporting rate if specified (represents surveillance capture rate)
        # This scales the OVERALL incidence but doesn't affect age distribution shape
        if hasattr(sim, '_reporting_rate') and sim._reporting_rate is not None:
            reporting_rate = sim._reporting_rate
            overall_incidence = overall_incidence * reporting_rate

        return overall_incidence, age_distribution

    def run_trial(self, trial):
        """ Define the objective for Optuna """
        sim = self.run_sim(calib_pars=self.calib_pars, trial=trial)
        fit = self.compute_fit(sim)
        return fit

    def load_study(self):
        """ Load a study from disk """
        args = self.run_args
        study = op.load_study( storage=args.storage, study_name=args.name)
        return study

    def worker(self):
        """ Run a single worker """
        if self.verbose:
            op.logging.set_verbosity(op.logging.DEBUG)
        else:
            op.logging.set_verbosity(op.logging.ERROR)
        study = self.load_study()
        output = study.optimize(self.run_trial, n_trials=self.run_args.n_trials)
        return output

    def run_workers(self):
        """ Run multiple workers in parallel """
        if self.run_args.n_workers > 1 and not self.run_args.debug: # Normal use case: run in parallel
            output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
        else: # Special case: just run one
            output = [self.worker()]
        return output

    def remove_db(self):
        """ Remove the database file """
        if os.path.exists(self.run_args.db_name):
            os.remove(self.run_args.db_name)
            print(f'Removed existing calibration file {self.run_args.db_name}')
        return

    def make_study(self):
        """ Make a study, deleting one if it already exists """
        self.remove_db()
        output = op.create_study(storage=self.run_args.storage, study_name=self.run_args.name)
        return output

    def calibrate(self, **kwargs):
        """
        Perform calibration.

        Args:
            kwargs (dict): if supplied, overwrite stored run_args (n_trials, n_workers, etc.)
        """
        # Load and validate calibration parameters
        self.run_args.update(kwargs) # Update optuna settings

        # Run the optimization
        self.T = sc.timer()
        self.make_study()
        self.run_workers() # Actually run!

        # Load and parse results
        study = self.load_study()
        self.best_pars = sc.objdict(study.best_params)
        self.parse_study(study)
        if self.verbose:
            print('Best pars:', self.best_pars)

        # Tidy up
        self.remove_db()
        self.calibrated = True
        self.T.toc()
        return self

    def check_fit(self, verbose=True):
        """ Run before and after simulations to validate the fit """
        if verbose: print('Checking fit...')
        before_pars = self.calib_to_sim_pars()
        self.before_sim = self.run_sim(sim_pars=before_pars)
        self.after_sim  = self.run_sim(sim_pars=self.best_pars)

        # Get simulation results (returns tuples of (overall_incidence, age_distribution))
        self.before_overall_incidence, self.before_age_distribution = self.sim_to_df(self.before_sim)
        self.after_overall_incidence, self.after_age_distribution = self.sim_to_df(self.after_sim)

        # For backwards compatibility, store dataframes with both metrics
        self.before_df = self.before_age_distribution  # Age distribution dataframe
        self.after_df = self.after_age_distribution

        # Get full GOF breakdown (total_gof, incidence_gof, age_dist_gof)
        self.before_fit, self.before_incidence_gof, self.before_age_gof = self.compute_fit(self.before_sim, full=True)
        self.after_fit, self.after_incidence_gof, self.after_age_gof = self.compute_fit(self.after_sim, full=True)

        if verbose:
            print(f'\nFit with original pars:')
            print(f'  Total GOF:       {self.before_fit:n}')
            print(f'  Incidence GOF:   {self.before_incidence_gof:n}')
            print(f'  Age dist GOF:    {self.before_age_gof:n}')
            print(f'\nFit with best-fit pars:')
            print(f'  Total GOF:       {self.after_fit:n}')
            print(f'  Incidence GOF:   {self.after_incidence_gof:n}')
            print(f'  Age dist GOF:    {self.after_age_gof:n}')

            if self.after_fit <= self.before_fit:
                print('\n✓ Calibration improved fit')
            else:
                print('\n✗ Calibration did not improve fit')
        return self.before_fit, self.after_fit

    def parse_study(self, study):
        """Parse the study into a data frame -- called automatically """
        best = study.best_params
        self.best_pars = best

        if self.verbose: print('Making results structure...')
        results = []
        n_trials = len(study.trials)
        failed_trials = []
        for trial in study.trials:
            data = {'index':trial.number, 'mismatch': trial.value}
            for key,val in trial.params.items():
                data[key] = val
            if data['mismatch'] is None:
                failed_trials.append(data['index'])
            else:
                results.append(data)
        if self.verbose: print(f'Processed {n_trials} trials; {len(failed_trials)} failed')

        keys = ['index', 'mismatch'] + list(best.keys())
        data = sc.objdict().make(keys=keys, vals=[])
        for i,r in enumerate(results):
            for key in keys:
                if key not in r:
                    warnmsg = f'Key {key} is missing from trial {i}, replacing with default'
                    print(warnmsg)
                    r[key] = best[key]
                data[key].append(r[key])
        self.study_data = data
        self.df = sc.dataframe.from_dict(data)
        self.df = self.df.sort_values(by=['mismatch']) # Sort
        return

    def to_json(self, filename=None, indent=2, **kwargs):
        """ Convert the results to JSON """
        order = np.argsort(self.df['mismatch'])
        json = []
        for o in order:
            row = self.df.iloc[o,:].to_dict()
            rowdict = dict(index=row.pop('index'), mismatch=row.pop('mismatch'), pars={})
            for key,val in row.items():
                rowdict['pars'][key] = val
            json.append(rowdict)
        self.json = json
        if filename:
            return sc.savejson(filename, json, indent=indent, **kwargs)
        else:
            return json

    def plot_sims(self, **kwargs):
        """ Plot sims, before and after calibration """
        data = self.data
        if not hasattr(self, 'after_fit'):
            self.check_fit(verbose=False)

        fig = plt.figure()

        # Plot raw values
        plt.subplot(3,1,1)
        for label,df in dict(Data=data, Before=self.before_df, After=self.after_df).items():
            plt.scatter(df.ages, df.inci, label=label)
        plt.ylim(bottom=0)
        plt.xlabel('Age')
        plt.ylabel('Incidence')
        plt.legend()

        # Plot goodness-of-fit
        plt.subplot(3,1,2)
        for i,label,gofs in sc.odict(Before=self.before_gofs, After=self.after_gofs).enumitems():
            x = np.arange(len(gofs))
            dx = 0.3
            plt.bar(x+i*dx, gofs, label=label, width=dx)
        plt.ylim(bottom=0)
        plt.xlabel('Age')
        plt.ylabel('Goodness-of-fit')
        plt.legend()

        # Plot fit
        plt.subplot(3,1,3)
        plt.barh(y=[1,0], width=[self.before_fit, self.after_fit])
        plt.yticks([1,0], ['Before', 'After'])
        plt.xlabel('Total mismatch')
        return fig

    def plot_trend(self, best_thresh=None, fig_kw=None):
        """ Plot the trend in best mismatch over trials """
        df = self.df.sort_values('index') # Make a copy of the dataframe, sorted by trial number
        mismatch = sc.dcp(df['mismatch'].values)
        best_mismatch = np.zeros(len(mismatch))
        for i in range(len(mismatch)):
            best_mismatch[i] = mismatch[:i+1].min()
        smoothed_mismatch = sc.smooth(mismatch)
        fig = plt.figure(**sc.mergedicts(fig_kw))

        ax1 = plt.subplot(2,1,1)
        plt.plot(mismatch, alpha=0.2, label='Original')
        plt.plot(smoothed_mismatch, lw=3, label='Smoothed')
        plt.plot(best_mismatch, lw=3, label='Best')

        ax2 = plt.subplot(2,1,2)
        max_mismatch = mismatch.min()*best_thresh if best_thresh is not None else np.inf
        inds = sc.findinds(mismatch<=max_mismatch)
        plt.plot(best_mismatch, lw=3, label='Best')
        plt.scatter(inds, mismatch[inds], c=mismatch[inds], label='Trials')
        for ax in [ax1, ax2]:
            plt.sca(ax)
            plt.grid(True)
            plt.legend()
            sc.setylim()
            sc.setxlim()
            plt.xlabel('Trial number')
            plt.ylabel('Mismatch')

        sc.figlayout()
        return fig


if __name__ == '__main__':

    # Run in debug mode (serial)
    debug = False
    total_trials = 20

    # Create the base sim with InfectedStrainStats analyzer
    # Need longer simulation and demographics to sustain infections for calibration
    sim = rs.Sim(
        n_agents = 10_000,
        start = "2000-01-01",
        stop = "2010-01-01",  # 10 years to cover years 1-9 needed for calibration
        verbose = False,
        scenario = "baseline",  # Use baseline scenario with multiple strains
        analyzers = [rs.InfectedStrainStats()],  # Add the infection tracking analyzer
        demographics = [  # Add demographics to sustain population
            ss.Births(birth_rate=ss.peryear(25)),  # 25 per 1000 per year
            ss.Deaths(death_rate=ss.peryear(10)),  # 10 per 1000 per year
        ],
    )

    # Convert the data
    data = process_incidence.process_data()

    # Specify the calibration parameters
    calib_pars = sc.objdict(
        rel_beta = [0.01, 0.001, 0.1],  # Expanded range to allow much lower transmission
        reassortment_rate = [0.10, 0.05, 0.15]
    )

    # Run the calibration
    calib = Calibration(
        sim = sim,
        data = data,
        calib_pars = calib_pars,
        total_trials = total_trials,
        debug = debug,
    )


    calib.calibrate()
    calib.check_fit()

    # Plot and save results
    print('\nGenerating plots...')
    fig1 = calib.plot_sims()
    fig1.savefig('calibration_fit.png', dpi=150, bbox_inches='tight')
    print('  Saved: calibration_fit.png')

    fig2 = calib.plot_trend()
    fig2.savefig('calibration_trend.png', dpi=150, bbox_inches='tight')
    print('  Saved: calibration_trend.png')

    plt.show()  # Display plots if running interactively