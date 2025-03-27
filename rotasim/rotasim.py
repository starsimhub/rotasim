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
            unit = 'year',
            dt = 1/365,
            verbose = 0,
            to_csv = True,
            rand_seed = 1,
            rota_kwargs = {},
            **kwargs,
        ):
        """
        Create the simulation.

        Args:
            n_agents (int): the number of agents in the simulation
            timelimit (int): the number of time units to run the simulation for
            start (int): the starting date of the simulation
            unit (str): the unit of time to use
            dt (float): the time step to use
            verbose (bool): the "verbosity" of the output: if False, print nothing; if None, print the timestep; if True, print out results
            to_csv (bool): whether to save the results to a CSV file
            rand_seed (int): the random seed to use
            rota_kwargs (dict): custom parameters for the Rota class
            kwargs (dict): additional Sim keyword arguments,
        """
        # N is the old name for n_agents, replace it with the new key if it's present
        if 'N' in kwargs:
            n_agents = kwargs.pop('N')

        # If the Rota module isn't provided, create it
        if 'connectors' not in kwargs:
            kwargs['connectors'] = rg.Rota(to_csv=to_csv, **rota_kwargs)

        super().__init__(n_agents=n_agents, start=start, stop=start+timelimit, unit=unit, dt=dt, verbose=verbose, rand_seed=rand_seed, use_aging=True, **kwargs)

        if verbose:
            print(f'Creating simulation with n_agents={n_agents}, timelimit={timelimit} and parameters:')

        return


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

