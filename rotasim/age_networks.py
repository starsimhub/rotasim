"""
Age-assortative contact networks for rotavirus transmission

Children and adults have equal total contact rates, but preferentially contact
their own age group (50% assortative, 50% cross-age mixing).
"""

import starsim as ss
import numpy as np

__all__ = ['AgeAssortativeNet']


class AgeAssortativeNet(ss.DynamicNetwork):
    """
    Age-assortative contact network with preferential within-group mixing

    Both children (<5 years) and adults (≥5 years) have the same total number
    of contacts, but 50% are with their own age group and 50% are cross-age.

    Parameters:
        n_contacts: Mean total contacts per person per day (default: 7)
        assortativity: Fraction of contacts within same age group (default: 0.5)
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

        if not born.any():
            return

        child_uids = (people.age < self.pars.child_age_threshold & born).uids
        adult_uids = (people.age >= self.pars.child_age_threshold & born).uids

        n_children = len(child_uids)
        n_adults = len(adult_uids)

        if n_children == 0 or n_adults == 0:
            # Fall back to random mixing if only one age group
            all_uids = born.uids
            n_contacts_total = int(len(all_uids) * self.pars.n_contacts / 2)
            p1 = np.random.choice(all_uids, size=n_contacts_total, replace=True)
            p2 = np.random.choice(all_uids, size=n_contacts_total, replace=True)
            # Filter out self-loops
            valid = p1 != p2
            p1 = p1[valid]
            p2 = p2[valid]
            if len(p1) > 0:
                beta = np.ones(len(p1)) * self.pars.beta
                dur = np.ones(len(p1))
                self.append(p1=p1, p2=p2, beta=beta, dur=dur)
            return

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

        # # count unique values in contacts_p1 and contacts_p2
        # unique_p1, p1_counts = np.unique(contacts_p1, return_counts=True)
        # unique_p2, p2_counts = np.unique(contacts_p2, return_counts=True)
        #
        # all_counts = dict.fromkeys(self.sim.people.uid[:], 0)
        # for uid, count in zip(unique_p1, p1_counts):
        #     all_counts[uid] += count
        # for uid, count in zip(unique_p2, p2_counts):
        #     all_counts[uid] += count


        # Append contacts using the Network.append method
        if len(contacts_p1) > 0:
            p1 = np.array(contacts_p1)
            p2 = np.array(contacts_p2)
            beta = np.ones(len(p1)) * self.pars.beta
            dur = np.ones(len(p1))  # Duration in timesteps
            self.append(p1=p1, p2=p2, beta=beta, dur=dur)

        return
