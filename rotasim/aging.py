"""
Custom aging module for rotasim

Starsim 3.0.2 does not automatically age the population despite documentation claims.
This module explicitly increments agent ages at each timestep.
"""

import starsim as ss
import numpy as np

__all__ = ['Aging']


class Aging(ss.Demographics):
    """
    Demographics module that increments ages at each timestep

    Usage:
        sim = ss.Sim(demographics=[Aging(), ss.Births(...), ss.Deaths(...)])
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    def step(self):
        """Increment all agent ages by dt"""
        # Get timestep size in days
        if hasattr(self.sim.pars, 'dt'):
            dt = self.sim.pars.dt
            if hasattr(dt, 'days'):
                dt_days = dt.days  # If dt is a timedelta-like object
            else:
                dt_days = float(dt)  # If dt is numeric (already in days)
        else:
            # Default to 1 day if dt not found
            dt_days = 1.0

        # Increment all ages by dt_days
        self.sim.people.age[:] += dt_days

        return
