"""
Immunity between rotavirus strains, and from mothers
"""

# Third-party imports
import numpy as np
import starsim as ss

# Local imports
from .rotavirus import Rotavirus

__all__ = ["RotaImmunityConnector", "erlang_survival"]


def erlang_survival(t, n_stages, mean_duration):
    """
    The share of an Erlang(n_stages) waiting time still running at time t

    This is the waning of a maternal class split into several compartments, as in rotavirus
    transmission models (e.g. Pitzer et al.): `n_stages=1` is exponential decay with the given
    mean, and larger values give a plateau followed by a sharper drop.

    Args:
        t (array): time, in the same units as mean_duration
        n_stages (int): number of stages
        mean_duration (float): mean of the waiting time
    """
    n_stages = int(n_stages)
    x = n_stages / mean_duration * np.asarray(t, dtype=float)
    total = np.ones_like(x)
    term = np.ones_like(x)
    for k in range(1, n_stages):
        term = term * x / k
        total += term
    return np.exp(-x) * total


class RotaImmunityConnector(ss.Connector):
    """
    Sets each agent's susceptibility to each strain from their infection history, age, and strains seen

    For a strain, an agent's protection is the larger of two things:

    - Acquired immunity: `factor * efficacy`. The efficacy depends on the closest match among the strains
      they have recovered from (homotypic, partial heterotypic if G or P is shared, or complete heterotypic).
      The factor is the larger of their baseline protection, which depends on how many infection episodes
      they have had and does not wane, and the waning immunity from the matching strains.
    - Maternal immunity, which depends on age.

    The strains an agent has seen are held as bitmasks, one bit per genotype, so that matching is vectorized.

    Args:
        pars (dict): parameters
        homotypic_immunity_efficacy (float): protection after recovering from the same G,P
        partial_heterotypic_immunity_efficacy (float): protection after recovering from a strain sharing G or P
        complete_heterotypic_immunity_efficacy (float): protection after recovering only from strains sharing neither
        naive_immunity_efficacy (float): protection of agents with no prior infections
        use_fixed_susceptibility (bool): whether baseline protection is set by sus_after_1, sus_after_2, and sus_after_3plus, rather than saturating with baseline_immunity_exponential_rate
        sus_after_1 (float): susceptibility after 1 infection episode
        sus_after_2 (float): susceptibility after 2 infection episodes
        sus_after_3plus (float): susceptibility after 3 or more infection episodes
        baseline_immunity_exponential_rate (float): baseline protection is 1 - exp(-rate * episodes)
        maternal_immunity_model (str): how maternal immunity wanes: 'erlang' (the same for everyone of an age) or 'titer' (from each infant's antibody titer)
        maternal_immunity_efficacy (float): protection at birth; 0 turns maternal immunity off
        maternal_immunity_half_life (ss.Duration): half-life, which sets the mean duration if that is not given
        maternal_immunity_n_stages (int): stages of Erlang waning (1 is exponential)
        maternal_immunity_mean_duration (ss.Duration): mean duration of Erlang waning
        maternal_immunity_efficacy_slow (float): a second, slower component, e.g. from breastfeeding (Erlang model only)
        maternal_immunity_half_life_slow (ss.Duration): half-life of the slow component
        maternal_titer_median (float): median initial titer, as a multiple of the titer that gives half protection (titer model)
        maternal_titer_gsd (float): geometric standard deviation of initial titers
        maternal_titer_half_life (ss.Duration): half-life of the titer
        maternal_hill_slope (float): slope of the Hill curve from titer to protection
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()

        self.define_pars(
            homotypic_immunity_efficacy=1.0,
            partial_heterotypic_immunity_efficacy=0.5,
            complete_heterotypic_immunity_efficacy=0.3,
            naive_immunity_efficacy=0.0,
            use_fixed_susceptibility=False,
            sus_after_1=0.67,
            sus_after_2=0.50,
            sus_after_3plus=0.36,
            baseline_immunity_exponential_rate=0.1,
            maternal_immunity_model="erlang",
            maternal_immunity_efficacy=0.0,
            maternal_immunity_half_life=ss.days(90),
            maternal_immunity_n_stages=1,
            maternal_immunity_mean_duration=None,
            maternal_immunity_efficacy_slow=0.0,
            maternal_immunity_half_life_slow=ss.days(270),
            maternal_titer_median=8.0,
            maternal_titer_gsd=2.0,
            maternal_titer_half_life=ss.days(35),
            maternal_hill_slope=3.0,
        )
        self.update_pars(pars=pars, **kwargs)

        self.define_states(
            ss.IntArr("exposed_GP_bitmask", default=0),  # The (G,P) pairs each agent has recovered from
            ss.IntArr("exposed_G_bitmask", default=0),  # The G genotypes each agent has recovered from
            ss.IntArr("exposed_P_bitmask", default=0),  # The P genotypes each agent has recovered from
            ss.BoolArr("has_immunity", default=False),  # Whether the agent has any prior infection
            ss.FloatArr("num_recovered_infections", default=0.0),  # Infection episodes completed
            ss.FloatArr("num_current_infections", default=0.0),  # Strains the agent has now
            ss.FloatArr("maternal_titer_z", default=ss.normal(0, 1)),  # Each infant's initial titer, as a draw from a standard normal
        )

        self.rota_diseases = []

    def init_pre(self, sim):
        """Find the Rotavirus diseases, and give each genotype its bit"""
        super().init_pre(sim)
        self.rota_diseases = [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]

        unique_G = sorted(set(d.G for d in self.rota_diseases))
        unique_P = sorted(set(d.P for d in self.rota_diseases))
        unique_GP = sorted(set(d.strain for d in self.rota_diseases))
        max_bits = np.iinfo(ss.dtypes.int).bits - 1
        if max(len(unique_G), len(unique_P), len(unique_GP), 0) > max_bits:
            errormsg = f"Too many genotypes for the bitmasks: {len(unique_G)} G, {len(unique_P)} P, {len(unique_GP)} G,P pairs; the maximum is {max_bits} of each"
            raise ValueError(errormsg)

        self.masks = []  # The (G, P, GP) bits of each disease
        self.partners = []  # The positions of the diseases that share only G or only P with each disease
        for d in self.rota_diseases:
            d.immunity = self
            self.masks.append((1 << unique_G.index(d.G), 1 << unique_P.index(d.P), 1 << unique_GP.index(d.strain)))
            self.partners.append([i for i, other in enumerate(self.rota_diseases) if (other.G == d.G) ^ (other.P == d.P)])

    def baseline_protection(self):
        """The protection from past infection episodes, which does not wane"""
        p = self.pars
        episodes = self.num_recovered_infections.values
        if p.use_fixed_susceptibility:
            susceptibility = np.select([episodes >= 3, episodes >= 2, episodes >= 1], [p.sus_after_3plus, p.sus_after_2, p.sus_after_1], default=1.0)
            return 1 - susceptibility
        return 1 - np.exp(-p.baseline_immunity_exponential_rate * episodes)

    def maternal_protection(self):
        """The protection from maternal antibodies, or 0 if there is none"""
        p = self.pars
        if p.maternal_immunity_efficacy == 0 and p.maternal_immunity_efficacy_slow == 0:
            return 0
        ages = self.sim.people.age.values
        if p.maternal_immunity_model == "titer":
            titer0 = p.maternal_titer_median * np.exp(np.log(p.maternal_titer_gsd) * self.maternal_titer_z.values)
            titer = titer0 * np.exp(-np.log(2) * ages / p.maternal_titer_half_life.years)
            hill = titer**p.maternal_hill_slope
            return p.maternal_immunity_efficacy * hill / (hill + 1)  # A titer of 1 gives half protection
        elif p.maternal_immunity_model == "erlang":
            mean_duration = p.maternal_immunity_mean_duration or p.maternal_immunity_half_life / np.log(2)
            fast = p.maternal_immunity_efficacy * erlang_survival(ages, p.maternal_immunity_n_stages, mean_duration.years)
            slow = p.maternal_immunity_efficacy_slow * np.exp(-np.log(2) * ages / p.maternal_immunity_half_life_slow.years)
            return 1 - (1 - fast) * (1 - slow)
        raise ValueError(f"maternal_immunity_model must be 'erlang' or 'titer', not {p.maternal_immunity_model!r}")

    def step(self):
        """Set each strain's rel_sus"""
        p = self.pars
        baseline = self.baseline_protection()
        maternal = self.maternal_protection()
        waned = np.array([d.waned_immunity_efficacy.values for d in self.rota_diseases])
        has_immunity = self.has_immunity.values
        G_bits = self.exposed_G_bitmask.values
        P_bits = self.exposed_P_bitmask.values
        GP_bits = self.exposed_GP_bitmask.values

        for i, disease in enumerate(self.rota_diseases):
            G_mask, P_mask, GP_mask = self.masks[i]
            exact = (GP_bits & GP_mask) != 0
            partial = (((G_bits & G_mask) != 0) | ((P_bits & P_mask) != 0)) & ~exact

            conditions = [~has_immunity, exact, partial]
            choices = [p.naive_immunity_efficacy, p.homotypic_immunity_efficacy, p.partial_heterotypic_immunity_efficacy]
            efficacy = np.select(conditions, choices, default=p.complete_heterotypic_immunity_efficacy)

            factor = baseline.copy()
            factor[exact] = np.maximum(factor[exact], waned[i, exact])
            if self.partners[i]:
                factor[partial] = np.maximum(factor[partial], waned[self.partners[i]][:, partial].max(axis=0))

            disease.rel_sus[:] = 1 - np.maximum(factor * efficacy, maternal)

    def record_infection(self, uids):
        """Count new infections, called by Rotavirus when agents are infected"""
        self.num_current_infections[uids] += 1

    def record_recovery(self, disease, uids):
        """Note the strain each agent has recovered from, called by Rotavirus when agents recover"""
        if not len(uids):
            return

        G_mask, P_mask, GP_mask = self.masks[self.rota_diseases.index(disease)]
        self.exposed_G_bitmask[uids] = self.exposed_G_bitmask[uids] | G_mask
        self.exposed_P_bitmask[uids] = self.exposed_P_bitmask[uids] | P_mask
        self.exposed_GP_bitmask[uids] = self.exposed_GP_bitmask[uids] | GP_mask
        self.has_immunity[uids] = True

        # Co-infections and overlapping infections count as one episode, which ends with the last of them
        self.num_current_infections[uids] -= 1
        episode_over = uids[self.num_current_infections[uids] == 0]
        self.num_recovered_infections[episode_over] += 1

    def initialize_immunity(self, min_age, max_age, exposures_per_year=1.5):
        """
        Give agents the infection history expected for their age; call after `sim.init()`

        Most adults have had several rotavirus infections in childhood, so a sim that starts
        with a naive population overestimates their susceptibility.

        Args:
            min_age (float): youngest age to initialize, in years
            max_age (float): oldest age to initialize, in years
            exposures_per_year (float): expected infections per year of life
        """
        ages = self.sim.people.age
        uids = ((ages >= min_age) & (ages < max_age)).uids
        self.has_immunity[uids] = True
        self.num_recovered_infections[uids] += np.round(exposures_per_year * ages[uids])
