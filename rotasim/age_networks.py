"""
Age-assortative contact network for rotavirus transmission
"""

# Third-party imports
import numpy as np
import starsim as ss

__all__ = ["AgeAssortativeNet"]


class AgeAssortativeNet(ss.DynamicNetwork):
    """
    A network where children and adults mix preferentially with their own group

    Everyone has `n_contacts` contacts per step on average. A share `assortativity` of children's
    contacts are with other children, and the rest are with adults, who fill the rest of their
    contacts with each other. Contacts are redrawn each step.

    Args:
        n_contacts (float): mean contacts per person per step
        assortativity (float): share of children's contacts that are with other children
        child_age_threshold (float): age in years below which people are children
        beta (float): transmission multiplier of each contact
    """

    def __init__(self, pars=None, **kwargs):
        super().__init__()
        self.define_pars(n_contacts=7, assortativity=0.5, child_age_threshold=5, beta=1.0)
        self.update_pars(pars, **kwargs)
        self.rng_pairs = ss.random(name="rng_pairs")

    def pick(self, uids, n):
        """Pick n agents from uids at random, with replacement"""
        return uids[(self.rng_pairs.rvs(n) * len(uids)).astype(int)]

    def add_pairs(self):
        """Make this step's contacts"""
        p = self.pars
        age = self.sim.people.age
        children = (age < p.child_age_threshold).uids
        adults = (age >= p.child_age_threshold).uids

        # Each contact within a group uses up two people's contacts, and a contact between groups one of each
        child_contacts = len(children) * p.n_contacts
        adult_contacts = len(adults) * p.n_contacts
        n_between = int(min(child_contacts * (1 - p.assortativity), adult_contacts))
        n_children = int((child_contacts - n_between) / 2)
        n_adults = int((adult_contacts - n_between) / 2)

        p1 = np.concatenate([self.pick(children, n_children), self.pick(adults, n_adults), self.pick(children, n_between)])
        p2 = np.concatenate([self.pick(children, n_children), self.pick(adults, n_adults), self.pick(adults, n_between)])
        keep = p1 != p2
        n = np.count_nonzero(keep)
        self.append(p1=p1[keep], p2=p2[keep], beta=np.full(n, float(p.beta)), dur=np.zeros(n))
