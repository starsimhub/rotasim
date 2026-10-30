"""
Rotavirus disease class: one instance per strain
"""

# Third-party imports
import numpy as np
import starsim as ss

__all__ = ["Rotavirus"]


def init_prev_by_age(module, sim, uids):
    """The chance that each agent is infected at the start, to give the disease's init_prevalence with its init_age_dist"""
    ages = sim.people.age[uids]
    n_infected = int(len(uids) * module.pars.init_prevalence)
    p = np.zeros(len(uids))
    for age_min, age_max, share in module.pars.init_age_dist:
        in_group = (ages >= age_min) & (ages < age_max)
        p[in_group] = share * n_infected / max(in_group.sum(), 1)
    return np.minimum(p, 1)


class Rotavirus(ss.Infection):
    """
    One rotavirus strain, as a disease of its own

    Each strain is a G,P combination. People can carry several strains at once, and
    `RotaImmunityConnector` handles the protection between them.

    An infection has a symptomatic phase, then an asymptomatic phase with reduced shedding.
    People are susceptible again once they recover, with their protection set by the connector.

    Args:
        G (int): G genotype
        P (int): P genotype
        pars (dict): parameters
        init_prevalence (float): share of people infected at the start
        init_age_dist (list): how those infections are spread by age, as (minimum age, maximum age, share)
        init_prev (float/ss.Dist): who is infected at the start, instead of the two above
        beta (ss.Rate): transmission rate
        dur_inf (ss.Dist): total duration of infection
        dur_asymptomatic (ss.Dist): duration of the asymptomatic phase, the last part of the infection
        asymptomatic_shedding_rate (float): infectiousness during the asymptomatic phase, relative to the symptomatic phase
        waning_rate_dist (ss.Dist): mean duration of immunity after recovery, drawn per person; immunity then decays exponentially
        waning_delay (ss.dur): time after recovery before immunity starts to wane
        fitness (float): the strain's relative transmissibility (see `rs.Sim`)
    """

    immunity = None  # The RotaImmunityConnector, if any; it sets this when the sim is initialized

    def __init__(self, G, P, pars=None, **kwargs):
        super().__init__()
        self.G = G
        self.P = P

        self.define_pars(
            init_age_dist=[(0, 100, 1)],
            init_prevalence=0.04,
            init_prev=ss.bernoulli(p=init_prev_by_age),
            beta=ss.perday(0.16),
            dur_inf=ss.lognorm_ex(mean=ss.days(13)),
            dur_asymptomatic=ss.constant(ss.days(8)),
            asymptomatic_shedding_rate=0.1,
            waning_rate_dist=ss.normal(loc=ss.days(91), scale=ss.days(14)),
            waning_delay=ss.days(0),
            fitness=1,
        )

        # ss.Infection already provides susceptible, infected, rel_sus, rel_trans, ti_infected
        self.define_states(
            ss.BoolState("asymptomatic", label="Asymptomatic"),
            ss.TimeArr("ti_asymptomatic", label="Time of asymptomatic phase start"),
            ss.BoolState("recovered", label="Recovered"),
            ss.TimeArr("ti_recovered", label="Time of recovery"),
            ss.FloatArr("waning_rate", default=0.0, label="Rate of immunity waning per step"),
            ss.FloatArr("waned_immunity_efficacy", default=0.0, label="Immunity left from this strain, from 0 to 1"),
            ss.FloatArr("n_infections", default=0, label="Number of infections with this strain"),
        )

        kwargs.setdefault("name", f"G{G}P{P}")
        self.update_pars(pars=pars, **kwargs)

    def init_results(self):
        super().init_results()
        self.define_results(ss.Result("new_recovered", label="New recoveries", dtype=int, scale=True))

    def set_prognoses(self, uids, sources=None):
        """Infect some agents, and schedule the end of each phase of their infection"""
        super().set_prognoses(uids, sources)

        self.infected[uids] = True
        self.asymptomatic[uids] = False
        self.recovered[uids] = False
        self.ti_infected[uids] = self.ti
        self.n_infections[uids] += 1
        self.rel_trans[uids] = 1.0
        self.waned_immunity_efficacy[uids] = 1.0

        # An infection happens at the end of a step, so its course starts at the next one
        dur_inf = self.pars.dur_inf.rvs(uids)
        dur_asymptomatic = self.pars.dur_asymptomatic.rvs(uids)
        self.ti_asymptomatic[uids] = self.ti + 1 + np.maximum(dur_inf - dur_asymptomatic, 0)
        self.ti_recovered[uids] = self.ti + 1 + dur_inf

        if self.immunity is not None:
            self.immunity.record_infection(uids)

    def step_state(self):
        """Move people through the phases of infection, and wane immunity"""
        symptoms_end = (self.infected & ~self.asymptomatic & self.ti_asymptomatic.due()).uids
        self.asymptomatic[symptoms_end] = True
        self.rel_trans[symptoms_end] = self.pars.asymptomatic_shedding_rate

        recovering = (self.asymptomatic & self.ti_recovered.due()).uids
        self.infected[recovering] = False
        self.asymptomatic[recovering] = False
        self.recovered[recovering] = True
        self.susceptible[recovering] = True  # SIRS: protection after recovery comes from rel_sus, set by the connector
        self.waning_rate[recovering] = 1.0 / self.pars.waning_rate_dist.rvs(recovering)
        self.results.new_recovered[self.ti] = len(recovering)

        if self.immunity is not None:
            self.immunity.record_recovery(self, recovering)

        # Immunity holds for waning_delay after recovery, then decays exponentially
        recovered = self.recovered.uids
        delay = self.pars.waning_delay.days / self.t.dt.days
        steps_waning = np.maximum(self.ti - self.ti_recovered[recovered] - delay, 0)
        self.waned_immunity_efficacy[recovered] = np.exp(-self.waning_rate[recovered] * steps_waning)

    @property
    def strain(self):
        """The strain, as a (G,P) tuple"""
        return (self.G, self.P)
