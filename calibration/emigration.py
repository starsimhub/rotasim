"""
Custom migration module for starsim
Implements age-specific migration (emigration or immigration, primarily adults)
"""
import starsim as ss
import numpy as np


class Emigration(ss.Demographics):
    """
    Age-specific migration module

    Adds or removes agents from the population based on age-dependent migration rates.
    Primarily affects adults (>=5 years) to simulate migration patterns.

    Positive rates = immigration (adds agents)
    Negative rates = emigration (removes agents)

    Args:
        emigration_rate: net migration rate per 1000 per year for adults (default: 10)
                        Positive = immigration, Negative = emigration
        age_threshold: age above which migration occurs (default: 5 years)

    Example:
        # Emigration (negative migration)
        emigr = Emigration(emigration_rate=10, age_threshold=5)  # Removes agents

        # Immigration (positive migration)
        immigr = Emigration(emigration_rate=-4, age_threshold=5)  # Adds agents

        sim = ss.Sim(demographics=[births, deaths, emigr])
    """

    def __init__(self, emigration_rate=10, age_threshold=5, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            emigration_rate=emigration_rate,  # per 1000 per year (negative = immigration)
            age_threshold=age_threshold,  # years
        )

    def step(self):
        """Apply age-dependent emigration at each timestep"""
        # Get ages
        ages = self.sim.people.age

        # Calculate emigration probability for adults
        # Convert annual rate to per-timestep probability
        # emigration_rate is per 1000 per year
        annual_prob = self.pars.emigration_rate / 1000

        # Convert to per-timestep (assuming daily timesteps)
        if hasattr(self.sim.pars, 'dt'):
            dt_days = self.sim.pars.dt if isinstance(self.sim.pars.dt, (int, float)) else 1.0
        else:
            dt_days = 1.0  # Default to daily

        per_timestep_prob = annual_prob / 365.25 * dt_days

        # Apply only to adults (age >= threshold)
        eligible = (ages >= self.pars.age_threshold) & self.sim.people.alive

        # Sample who emigrates
        if eligible.sum() > 0:
            emigrate_probs = np.zeros(len(self.sim.people))
            emigrate_probs[eligible] = per_timestep_prob

            # Bernoulli trial
            will_emigrate = np.random.random(len(self.sim.people)) < emigrate_probs

            # Remove emigrating agents (mark as not alive)
            n_emigrated = will_emigrate.sum()
            if n_emigrated > 0:
                self.sim.people.alive[will_emigrate] = False
        else:
            pass  # No emigration this timestep
