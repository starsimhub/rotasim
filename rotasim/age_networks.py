"""
Age-assortative contact networks for rotavirus transmission

Children and adults have equal total contact rates, but children preferentially contact
their own age group (e.g. 50% assortative, 50% cross-age mixing).
"""

import starsim as ss
import numpy as np

__all__ = ['AgeAssortativeNet']


class AgeAssortativeNet(ss.DynamicNetwork):
    """
    Age-assortative contact network with preferential within-group mixing

    Both young children (<5 years) and older (≥5 years) have the same total number
    of contacts, but 50% of young children's contacts are with their own age group and 50% are cross-age. The size of
    the older age group is typically much larger than the younger age group, and we are most concerned with ensuring
    that young children have sufficient contacts with their own age group to sustain transmission so the assortativity
    fraction applies only to young children's contacts. Older individuals are assigned the remaining contacts.

    Parameters:
        n_contacts: Mean total contacts per person per day (default: 7)
        assortativity: Fraction of contacts within same age group for young children (default: 0.5)
        child_age_threshold: Age in years defining children (default: 5)
    """

    def __init__(self, n_contacts=7, assortativity=0.5, child_age_threshold=5, beta=1.0, **kwargs):
        super().__init__(**kwargs)
        self.define_pars(
            n_contacts=n_contacts,
            assortativity=assortativity,  # Fraction of contacts within same age group
            child_age_threshold=child_age_threshold,
            beta=beta,  # Transmission probability per contact
        )

    def init_pre(self, sim):
        """Initialize network before simulation starts"""
        super().init_pre(sim)

    def step(self):
        """Contacts regenerated each timestep via add_pairs"""
        self.end_pairs()
        self.add_pairs()
        return

    def add_pairs(self, ti=None):
        """
        Create contacts based on age-assortative mixing

        Each person gets n_contacts total:
        - assortativity fraction with same age group
        - (1-assortativity) fraction with other age group
        """
        # Get current ages in years
        people = self.sim.people
        born = people.alive & (people.age > 0)

        child_uids = (people.age < self.pars.child_age_threshold & born).uids
        adult_uids = (people.age >= self.pars.child_age_threshold & born).uids

        n_children = len(child_uids)
        n_adults = len(adult_uids)

        contacts_p1 = []
        contacts_p2 = []

        # Calculate number of each type of contact
        # Each person gets n_contacts, split by assortativity
        total_child_contacts = int(n_children * self.pars.n_contacts)
        total_adult_contacts = int(n_adults * self.pars.n_contacts)
        total_cc_contacts = int(total_child_contacts * self.pars.assortativity)
        total_ac_contacts = int(total_child_contacts - total_cc_contacts)
        total_aa_contacts = total_adult_contacts - total_ac_contacts

        # Child-child contacts (assortative)
        if n_children > 1:
            for _ in range(round(total_cc_contacts/2)):
                c1, c2 = np.random.choice(child_uids, size=2, replace=True)
                if c1 != c2:  # Avoid self-loops
                    contacts_p1.append(c1)
                    contacts_p2.append(c2)

        # Adult-adult contacts (assortative)
        if n_adults > 1:
            for _ in range(round(total_aa_contacts/2)):
                a1, a2 = np.random.choice(adult_uids, size=2, replace=True)
                if a1 != a2:  # Avoid self-loops
                    contacts_p1.append(a1)
                    contacts_p2.append(a2)

        # Child-adult contacts (cross-age mixing)
        for _ in range(round(total_ac_contacts/2)):
            c = np.random.choice(child_uids)
            a = np.random.choice(adult_uids)
            contacts_p1.append(c)
            contacts_p2.append(a)


        # Append contacts using the Network.append method
        if len(contacts_p1) > 0:
            p1 = np.array(contacts_p1)
            p2 = np.array(contacts_p2)
            beta = np.ones(len(p1)) * self.pars.beta
            dur = np.ones(len(p1))  # Duration in timesteps
            self.append(p1=p1, p2=p2, beta=beta, dur=dur)

        return
