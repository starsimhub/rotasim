"""
Age-assortative contact networks for rotavirus transmission

Children and adults have equal total contact rates, but preferentially contact
their own age group (50% assortative, 50% cross-age mixing).
"""

import starsim as ss
import numpy as np

__all__ = ['AgeAssortativeNet']


class AgeAssortativeNet(ss.Network):
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
        self.child_mask = None
        self.adult_mask = None

    def step(self):
        """Contacts regenerated each timestep via add_pairs"""
        pass

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

        ages_years = people.age.values / 365.25

        # Define age groups
        self.child_mask = (ages_years < self.pars.child_age_threshold) & born
        self.adult_mask = (ages_years >= self.pars.child_age_threshold) & born

        child_uids = np.where(self.child_mask)[0]
        adult_uids = np.where(self.adult_mask)[0]

        n_children = len(child_uids)
        n_adults = len(adult_uids)

        # DEBUG: Print on first timestep
        if ti == 0:
            print(f"\n[AgeAssortativeNet DEBUG @ ti={ti}]")
            print(f"  Assortativity: {self.pars.assortativity:.2f}")
            print(f"  n_contacts: {self.pars.n_contacts}")
            print(f"  Population: {n_children} children, {n_adults} adults")
            print(f"  Child fraction: {n_children/(n_children+n_adults)*100:.1f}%")

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
        n_within = self.pars.assortativity  # Fraction within same age
        n_between = 1.0 - self.pars.assortativity  # Fraction across ages

        # Child-child contacts (assortative)
        if n_children > 1:
            n_cc = int(n_children * self.pars.n_contacts * n_within / 2)
            for _ in range(n_cc):
                c1, c2 = np.random.choice(child_uids, size=2, replace=True)
                if c1 != c2:  # Avoid self-loops
                    contacts_p1.append(c1)
                    contacts_p2.append(c2)

        # Adult-adult contacts (assortative)
        if n_adults > 1:
            n_aa = int(n_adults * self.pars.n_contacts * n_within / 2)
            for _ in range(n_aa):
                a1, a2 = np.random.choice(adult_uids, size=2, replace=True)
                if a1 != a2:  # Avoid self-loops
                    contacts_p1.append(a1)
                    contacts_p2.append(a2)

        # Child-adult contacts (cross-age mixing)
        n_ca = int((n_children + n_adults) * self.pars.n_contacts * n_between / 2)
        for _ in range(n_ca):
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

            # DEBUG: Count contact types on first timestep
            if ti == 0:
                n_cc = sum(1 for i, j in zip(p1, p2) if self.child_mask[i] and self.child_mask[j])
                n_aa = sum(1 for i, j in zip(p1, p2) if self.adult_mask[i] and self.adult_mask[j])
                n_ca = sum(1 for i, j in zip(p1, p2) if (self.child_mask[i] and self.adult_mask[j]) or (self.adult_mask[i] and self.child_mask[j]))
                total_contacts = len(p1)
                print(f"  Contacts created:")
                print(f"    Child-child: {n_cc} ({n_cc/total_contacts*100:.1f}%)")
                print(f"    Adult-adult: {n_aa} ({n_aa/total_contacts*100:.1f}%)")
                print(f"    Cross-age: {n_ca} ({n_ca/total_contacts*100:.1f}%)")
                print(f"    Total: {total_contacts}")

            self.append(p1=p1, p2=p2, beta=beta, dur=dur)

        return
