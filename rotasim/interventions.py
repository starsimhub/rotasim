"""
Rotavirus vaccination
"""

# Third-party imports
import numpy as np
import starsim as ss

# Local imports
from .rotavirus import Rotavirus

__all__ = ["RotaVaccination"]

default_effectiveness = {1: [0.7], 2: [0.6, 0.8], 3: [0.5, 0.7, 0.85]}  # By the number of doses


class RotaVaccination(ss.Intervention):
    """
    Routine rotavirus vaccination of infants, with several doses and protection specific to G and P

    Each agent in the age range is offered each dose once, `dose_interval` apart, and takes it up with
    probability `uptake_dist`. Their protection against a strain is the effectiveness of the last dose
    they took, times the efficacy for how well the strain matches the vaccine, and it wanes exponentially
    from that dose. It lowers `rel_sus` when it protects more than the agent's immunity does, and lasts
    after the program ends.

    Args:
        start_date (str/ss.date): when vaccination starts
        end_date (str/ss.date): when vaccination ends (default: never)
        pars (dict): parameters
        n_doses (int): number of doses in the schedule
        dose_interval (ss.Duration): time between doses
        G_antigens (list): G genotypes in the vaccine
        P_antigens (list): P genotypes in the vaccine
        dose_effectiveness (list): effectiveness after each dose (default: depends on n_doses)
        min_age (ss.Duration): youngest age to vaccinate
        max_age (ss.Duration): oldest age to vaccinate
        uptake_dist (float/ss.bernoulli): probability of taking up each dose offered
        waning_rate_dist (ss.Dist): mean duration of protection, drawn per person at each dose
        waning_delay (ss.Duration): time after a dose before protection starts to wane
        homotypic_efficacy (float): relative efficacy against a strain whose G and P are both in the vaccine
        partial_heterotypic_efficacy (float): relative efficacy against a strain whose G or P is in the vaccine
        complete_heterotypic_efficacy (float): relative efficacy against a strain with neither

    Examples:
        ```python
        # Two doses of a monovalent G1P8 vaccine
        vax = rs.RotaVaccination(start_date='2025-01-01')

        # Three doses of a pentavalent-like vaccine
        vax = rs.RotaVaccination(start_date='2025-01-01', n_doses=3, G_antigens=[1, 2, 3, 4], P_antigens=[8], uptake_dist=0.9)
        ```
    """

    def __init__(self, start_date, end_date=None, pars=None, **kwargs):
        super().__init__()
        self.define_pars(
            start_date=start_date,
            end_date=end_date,
            n_doses=2,
            dose_interval=ss.days(28),
            G_antigens=[1],
            P_antigens=[8],
            dose_effectiveness=None,
            min_age=ss.days(42),
            max_age=ss.days(365),
            uptake_dist=ss.bernoulli(0.8),
            waning_rate_dist=ss.lognormal(mean=ss.days(365)),
            waning_delay=ss.days(0),
            homotypic_efficacy=1.0,
            partial_heterotypic_efficacy=0.6,
            complete_heterotypic_efficacy=0.3,
        )
        self.update_pars(pars=pars, **kwargs)

        p = self.pars
        if p.n_doses < 1:
            raise ValueError(f"n_doses must be at least 1, got {p.n_doses}")
        if p.dose_effectiveness is None:
            p.dose_effectiveness = default_effectiveness.get(p.n_doses, np.linspace(0.4, 0.8, p.n_doses).tolist())
        if len(p.dose_effectiveness) != p.n_doses:
            raise ValueError(f"dose_effectiveness must have {p.n_doses} values, got {len(p.dose_effectiveness)}")
        for key in ["homotypic_efficacy", "partial_heterotypic_efficacy", "complete_heterotypic_efficacy"]:
            if not 0 <= p[key] <= 1:
                raise ValueError(f"{key} must be between 0 and 1, got {p[key]}")

        self.define_states(
            ss.IntArr("doses_offered", default=0),
            ss.IntArr("doses_received", default=0),
            ss.BoolArr("completed_schedule", default=False),
            ss.TimeArr("ti_next_dose"),
            ss.TimeArr("ti_last_dose"),
            ss.FloatArr("waning_rate", default=0.0),
        )

    def init_pre(self, sim):
        super().init_pre(sim)
        self.diseases = [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]
        if not self.diseases:
            raise RuntimeError("RotaVaccination requires at least one Rotavirus disease")
        self.match_efficacy = [self.match_efficacy_of(d) for d in self.diseases]

    def init_post(self):
        super().init_post()
        p = self.pars
        years = self.t.yearvec
        self.start_ti = np.searchsorted(years, ss.date(p.start_date).years)
        self.end_ti = np.inf if p.end_date is None else np.searchsorted(years, ss.date(p.end_date).years, side="right") - 1

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result("new_doses", label="Doses given", dtype=int, scale=True))

    def match_efficacy_of(self, disease):
        """The relative efficacy against a strain, from how well it matches the vaccine"""
        p = self.pars
        matches = (disease.G in p.G_antigens) + (disease.P in p.P_antigens)
        return [p.complete_heterotypic_efficacy, p.partial_heterotypic_efficacy, p.homotypic_efficacy][matches]

    def check_eligibility(self):
        """The agents due to be offered a dose now, by age, doses offered so far, and the time since the last offer"""
        p = self.pars
        age = self.sim.people.age
        in_range = (age >= p.min_age.years) & (age <= p.max_age.years)
        due = ~(self.ti_next_dose > self.ti) # Those never offered a dose have no time set
        return (in_range & (self.doses_offered < p.n_doses) & due).uids

    def step(self):
        """Update protection, then offer doses"""
        self.apply_protection()
        if self.start_ti <= self.ti <= self.end_ti:
            eligible = self.check_eligibility()
            self.doses_offered[eligible] += 1
            self.ti_next_dose[eligible] = self.ti + self.pars.dose_interval.days / self.t.dt.days
            self.vaccinate(self.pars.uptake_dist.filter(eligible))

    def vaccinate(self, uids):
        """Give the next dose to some agents"""
        self.doses_received[uids] += 1
        self.ti_last_dose[uids] = self.ti
        self.completed_schedule[uids] = self.doses_received[uids] >= self.pars.n_doses
        self.waning_rate[uids] = 1.0 / self.pars.waning_rate_dist.rvs(uids)
        self.results.new_doses[self.ti] = len(uids)

    def apply_protection(self):
        """Lower rel_sus, for each strain, of those for whom the vaccine protects more than their immunity does"""
        p = self.pars
        uids = (self.doses_received > 0).uids
        if not len(uids):
            return
        delay = p.waning_delay.days / self.t.dt.days
        steps_waning = np.maximum(self.ti - self.ti_last_dose[uids] - delay, 0)
        effectiveness = np.array(p.dose_effectiveness)[self.doses_received[uids] - 1]
        protection = effectiveness * np.exp(-self.waning_rate[uids] * steps_waning)
        for disease, efficacy in zip(self.diseases, self.match_efficacy):
            disease.rel_sus[uids] = np.minimum(disease.rel_sus[uids], 1 - protection * efficacy)

    def get_vaccination_summary(self):
        """How many agents the program has reached, as a dict"""
        n_doses = self.pars.n_doses
        doses = self.doses_received
        received = doses > 0
        return dict(
            total_agents=len(doses),
            doses_offered=np.count_nonzero(self.doses_offered > 0),
            received_any_dose=np.count_nonzero(received),
            completed_schedule=np.count_nonzero(self.completed_schedule),
            doses_by_number={k: np.count_nonzero(doses == k) for k in range(n_doses + 1)},
            mean_doses=doses[received].mean() if received.any() else 0,
        )
