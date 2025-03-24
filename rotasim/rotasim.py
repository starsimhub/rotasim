"""
Rotasim model

Usage:
    import rotasim as rs
    sim = rs.Sim()
    sim.run()

TODO:
    - Figure out how to make host vaccination more efficient
    - Replace host with array
    - Replace pathogen with array
    - Replace random with numpy
    - Replace math with numpy
"""

import csv
import random as rnd
import numpy as np
import sciris as sc
import starsim as ss
from . import rotasim_genetics as rg



__all__ = ['Sim']


### Sim class
class Sim(ss.Sim):
    """
    Run the simulation
    """

    def __init__(self,
            n_agents = 10_000,
            timelimit = 10,
            start = 0,
            verbose = 0,
            to_csv = True,
            rand_seed = 1,
            **kwargs,
        ):
        """
        Create the simulation.

        Args:
            defaults (list): a list of parameters matching the command-line inputs; see below
            verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
        """

        if 'connectors' not in kwargs:
            kwargs['connectors'] = rg.Rota()


        super().__init__(n_agents=n_agents, start=start, stop=start+timelimit, unit='year', dt=1/365, verbose=verbose, rand_seed=rand_seed, **kwargs)



        # Update with any keyword arguments
        # for k,v in kwargs.items():
        #     if k in args:
        #         args[k] = v
        #     else:
        #         KeyError(k)

        # Loop over command line input arguments, if provided
        # Using sys.argv in this way breaks when using pytest because it passes two args instead of one (runner and script)
        # for i,arg in enumerate(sys.argv[1:]):
        #     args[i] = arg

        if verbose:
            print(f'Creating simulation with N={n_agents}, timelimit={timelimit} and parameters:')
            # print(args)


        return


    def init(self, force=False):
        """
        Set up the variables for the run
        """
        if force or not self.initialized:

            if self.pars.people is None:
                self.pars.people = ss.People(n_agents=self.pars.n_agents)

            super().init(force=force)

        return self


    def to_df(self):
        """ Convert results to a dataframe """
        cols = self.results.columns
        res = self.results.infected_all
        df = sc.dataframe(data=res, columns=cols)
        self.df = df
        return df



if __name__ == '__main__':
    sim = Sim(n_agents=10_000, timelimit=10)
    sim.run()
    print("done!")

