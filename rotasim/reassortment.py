"""
Genetic reassortment between strains in co-infected hosts
"""

# Third-party imports
import numpy as np
import starsim as ss

# Local imports
from .rotavirus import Rotavirus

__all__ = ["RotaReassortmentConnector"]


class RotaReassortmentConnector(ss.Connector):
    """
    Reassortment: a host with several strains can pick up a new mix of their G and P genotypes

    Each step, each co-infected host reassorts with probability `reassortment_prob`, and is then
    infected with one G,P combination of the strains they carry, chosen at random from those they
    do not already have. All possible reassortants exist from the start, as Rotavirus diseases with
    no initial infections; a combination that isn't among the sim's diseases can't arise (see
    `use_preferred_partners` of `rs.Sim`).

    Args:
        reassortment_prob (float): daily probability of reassortment per co-infected host
    """

    def __init__(self, reassortment_prob=0.05, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(reassortment_prob=ss.bernoulli(ss.probperday(reassortment_prob)))
        self.rng_strain = ss.random(name="rng_strain")  # Which reassortant each host gets
        self.diseases = []

    def init_pre(self, sim):
        """Find the Rotavirus diseases, and each one's genotypes as positions"""
        super().init_pre(sim)
        self.diseases = [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]
        if not self.diseases:
            raise ValueError("RotaReassortmentConnector requires at least one Rotavirus disease")
        unique_G = sorted(set(d.G for d in self.diseases))
        unique_P = sorted(set(d.P for d in self.diseases))
        self.G_index = np.array([unique_G.index(d.G) for d in self.diseases])
        self.P_index = np.array([unique_P.index(d.P) for d in self.diseases])

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result("n_reassortments", label="Reassortment events", dtype=int, scale=True, summarize_by="sum"))

    def step(self):
        """Find the co-infected hosts, and infect those who reassort with a reassortant"""
        n_strains = sum(d.infected.values.astype(int) for d in self.diseases)
        coinfected = self.sim.people.auids[n_strains >= 2]
        hosts = self.pars.reassortment_prob.filter(coinfected)
        if not len(hosts):
            return

        # Each row is a strain, each column a host
        infected = np.array([d.infected[hosts] for d in self.diseases])
        has_G = np.zeros((self.G_index.max() + 1, len(hosts)), dtype=bool)
        has_P = np.zeros((self.P_index.max() + 1, len(hosts)), dtype=bool)
        np.logical_or.at(has_G, self.G_index, infected)
        np.logical_or.at(has_P, self.P_index, infected)
        candidates = has_G[self.G_index] & has_P[self.P_index] & ~infected

        # Pick one candidate at random for each host that has any
        n_candidates = candidates.sum(axis=0)
        pick = (self.rng_strain.rvs(hosts) * n_candidates).astype(int)
        chosen = (candidates.cumsum(axis=0) > pick).argmax(axis=0)
        for i in np.unique(chosen[n_candidates > 0]):
            self.diseases[i].set_prognoses(hosts[(chosen == i) & (n_candidates > 0)])

        self.results.n_reassortments[self.ti] = np.count_nonzero(n_candidates)
