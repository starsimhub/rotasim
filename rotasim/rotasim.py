"""
Sim class for multi-strain rotavirus simulations
"""

# Third-party imports
import starsim as ss

# Local imports
from .immunity import RotaImmunityConnector
from .reassortment import RotaReassortmentConnector
from .rotavirus import Rotavirus
from .utils import apply_scenario_overrides, generate_gp_reassortments, strain_pars, validate_scenario

__all__ = ["Sim"]


class Sim(ss.Sim):
    """
    A rotavirus simulation, set up from a scenario of initial strains

    All G,P combinations of the initial strains are created as diseases, so that reassortants
    can arise; those not in the scenario start with no infections. A strain's transmission rate
    is `base_beta` times its fitness, per day.

    By default, a sim has a daily timestep, runs for 10 years, has a random network, and has both
    `RotaImmunityConnector` and `RotaReassortmentConnector`; any of these can be replaced, using the
    usual `ss.Sim` arguments. Agents age even with no births or deaths, since immunity depends on age.

    Args:
        scenario (str/dict): the name of a built-in scenario (see `rs.list_scenarios()`), or a scenario (see `rs.validate_scenario()`)
        base_beta (float): transmission rate per day of a strain with a fitness of 1
        override_fitness (float/dict): fitness of all strains, or by strain ({(G,P): fitness})
        override_prevalence (float/dict): initial prevalence of all strains, or by strain ({(G,P): prevalence}); see `rs.apply_scenario_overrides()`
        override_strains (dict): strains to add or replace ({(G,P): {"fitness": x, "init_prevalence": y}})
        use_preferred_partners (bool): whether to create only the G,P combinations in `rs.PREFERRED_PARTNERS`
        kwargs (dict): passed to `ss.Sim`

    Examples:
        ```python
        sim = rs.Sim(scenario='simple', n_agents=5000).run()
        sim = rs.Sim(scenario='baseline', override_fitness={(1,8): 0.95, (2,4): 0.8})
        sim = rs.Sim(scenario=dict(strains={(1,8): dict(fitness=1.0, init_prevalence=0.015), (2,4): dict(fitness=0.8, init_prevalence=0.01)}))
        ```
    """

    def __init__(self, scenario="baseline", base_beta=0.16, override_fitness=None, override_prevalence=None, override_strains=None, use_preferred_partners=False, **kwargs):
        final_scenario = apply_scenario_overrides(
            validate_scenario(scenario),
            override_fitness=override_fitness,
            override_prevalence=override_prevalence,
            override_strains=override_strains,
        )

        kwargs.setdefault("dt", ss.days(1))
        kwargs.setdefault("use_aging", True)
        kwargs.setdefault("networks", "random")
        if "stop" not in kwargs and "dur" not in kwargs:
            kwargs["dur"] = ss.years(10)
        if "connectors" not in kwargs:
            kwargs["connectors"] = [RotaImmunityConnector(), RotaReassortmentConnector()]

        diseases = self.make_strains(final_scenario, base_beta, use_preferred_partners)
        super().__init__(diseases=diseases, **kwargs)

        self.scenario = scenario
        self.final_scenario = final_scenario
        self.base_beta = base_beta

    @staticmethod
    def make_strains(scenario, base_beta, use_preferred_partners):
        """Make a Rotavirus disease for each G,P combination of the scenario's strains"""
        diseases = []
        for strain in generate_gp_reassortments(list(scenario["strains"]), use_preferred_partners):
            pars = strain_pars(scenario, strain)
            G, P = strain
            diseases.append(Rotavirus(G=G, P=P, beta=ss.perday(base_beta * pars["fitness"]), **pars))
        return diseases

    @property
    def initial_strains(self):
        """The strains of the scenario, as (G,P) tuples"""
        return list(self.final_scenario["strains"])
