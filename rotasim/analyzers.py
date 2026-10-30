"""
Analyzers for Rotasim: strains, events, ages, infections, and individual agents
"""

# Third-party imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import starsim as ss

# Local imports
from .rotavirus import Rotavirus

__all__ = ["StrainStats", "EventStats", "AgeStats", "InfectedStrainStats", "PersonTimeByAge", "UidTracker"]


def rotavirus_diseases(sim):
    """The Rotavirus diseases of a sim"""
    return [d for d in sim.diseases.values() if isinstance(d, Rotavirus)]


def new_infections(analyzer):
    """The agents infected with any strain in the current step"""
    new = [(d.ti_infected == analyzer.ti).uids for d in rotavirus_diseases(analyzer.sim)]
    return ss.uids(np.unique(np.concatenate(new)))


def years_elapsed(analyzer):
    """The time since the start of the sim, in years"""
    yearvec = analyzer.t.yearvec
    return yearvec[analyzer.ti] - yearvec[0]


class StrainStats(ss.Analyzer):
    """
    Infections with each strain over time

    Gives results named after each strain, e.g. `G1P8_count` (people infected) and `G1P8_proportion`
    (the strain's share of all infections).

    Example:
        ```python
        sim = rs.Sim(scenario='simple', analyzers=rs.StrainStats()).run()
        sim.analyzers.strainstats.results.G1P8_proportion
        ```
    """

    def init_results(self):
        super().init_results()
        self.diseases = rotavirus_diseases(self.sim)
        for disease in self.diseases:
            self.define_results(
                ss.Result(f"{disease.name}_proportion", dtype=float, scale=False),
                ss.Result(f"{disease.name}_count", dtype=float, scale=True),
            )

    def step(self):
        counts = np.array([d.infected.sum() for d in self.diseases], dtype=float)
        total = counts.sum()
        for disease, count in zip(self.diseases, counts):
            self.results[f"{disease.name}_count"][self.ti] = count
            self.results[f"{disease.name}_proportion"][self.ti] = count / total if total else 0.0


class EventStats(ss.Analyzer):
    """
    Key events per step: `births`, `deaths`, `recoveries`, `new_infections`, `reassortments`, and
    the people currently infected with any strain (`infected_agents`) or several (`coinfected_agents`)
    """

    event_types = ["births", "deaths", "recoveries", "new_infections", "reassortments", "infected_agents", "coinfected_agents"]

    def init_results(self):
        super().init_results()
        self.diseases = rotavirus_diseases(self.sim)
        self.reassortment = next((c for c in self.sim.connectors.values() if "n_reassortments" in c.results), None)
        self.define_results(*[ss.Result(event, dtype=int, scale=True) for event in self.event_types])

    def step(self):
        sim, ti, res = self.sim, self.ti, self.results
        if "births" in sim.results:
            res.births[ti] = sim.results.births.new[ti]
        res.deaths[ti] = sim.results.new_deaths[ti]
        if self.reassortment is not None:
            res.reassortments[ti] = self.reassortment.results.n_reassortments[ti]
        res.recoveries[ti] = sum(d.results.new_recovered[ti] for d in self.diseases)
        res.new_infections[ti] = sum(d.results.new_infections[ti] for d in self.diseases)
        strains = sum(d.infected.values.astype(int) for d in self.diseases)
        res.infected_agents[ti] = np.count_nonzero(strains >= 1)
        res.coinfected_agents[ti] = np.count_nonzero(strains >= 2)


class AgeStats(ss.Analyzer):
    """People alive in each age group over time, labeled by age in months"""

    age_bins = [2 / 12, 4 / 12, 6 / 12, 1, 2, 3, 4, 5]  # The upper edge of each group but the last, in years
    age_labels = ["0-2", "2-4", "4-6", "6-12", "12-24", "24-36", "36-48", "48-60", "60+"]

    def init_results(self):
        super().init_results()
        self.define_results(*[ss.Result(label, dtype=int, scale=True) for label in self.age_labels])

    def step(self):
        groups = np.digitize(self.sim.people.age.values, self.age_bins)
        counts = np.bincount(groups, minlength=len(self.age_labels))
        for label, count in zip(self.age_labels, counts):
            self.results[label][self.ti] = count


class InfectedStrainStats(ss.Analyzer):
    """
    A log of every infection, with `to_df()` giving one row for each

    Its columns are `id`, `Strain`, `CollectionTime` (in years from the start of the sim), `Age` (a group,
    as in `AgeStats`), `age_months_precise`, `PopulationSize`, `n_infections` (the agent's infection number
    with this strain), and `severity` (the probability that the infection is severe).

    Args:
        use_infection_based_severity (bool): whether severity depends on the infection number, rather than being constant
        constant_severity (float): severity if it is constant
    """

    severity_rates = np.array([0.051, 0.0644, 0.0432, 0.0378])  # By infection number, the last for the fourth and later

    def __init__(self, use_infection_based_severity=True, constant_severity=0.05, **kwargs):
        super().__init__(**kwargs)
        self.use_infection_based_severity = use_infection_based_severity
        self.constant_severity = constant_severity
        self.events = []

    def init_results(self):
        super().init_results()
        self.diseases = rotavirus_diseases(self.sim)

    def step(self):
        people = self.sim.people
        for disease in self.diseases:
            uids = (disease.ti_infected == self.ti).uids
            if len(uids):
                ages = people.age[uids]
                n_infections = disease.n_infections[uids].astype(int)
                severity = self.constant_severity
                if self.use_infection_based_severity:
                    severity = self.severity_rates[np.minimum(n_infections, 4) - 1]
                self.events.append(pd.DataFrame(dict(
                    id=uids,
                    Strain=f"G{disease.G}P{disease.P}A1B1",
                    CollectionTime=years_elapsed(self),
                    Age=np.array(AgeStats.age_labels)[np.digitize(ages, AgeStats.age_bins)],
                    age_months_precise=ages * 12.0,
                    PopulationSize=len(people),
                    n_infections=n_infections,
                    severity=severity,
                )))

    def to_df(self):
        """The infections, as a dataframe with one row for each"""
        return pd.concat(self.events, ignore_index=True) if self.events else pd.DataFrame()


class PersonTimeByAge(ss.Analyzer):
    """
    Person-time per age bin, for the denominators of incidence rates

    Adds up the people alive in each bin at each step, which stays right as the population changes,
    unlike multiplying a count at one time by the length of the window.

    Args:
        bins_months (dict): the bins, as label: (lowest age in months, highest age in months), with the highest not included
        calibration_window (tuple): start and stop, in years from the start of the sim (default: the whole sim)

    Example:
        ```python
        sim = rs.Sim(scenario='simple', analyzers=rs.PersonTimeByAge(calibration_window=(5, 10))).run()
        sim.analyzers.persontimebyage.person_months
        ```
    """

    default_bins_months = {"<6 m": (0.0, 6.0), "6-11 m": (6.0, 12.0), "12-23 m": (12.0, 24.0), "24-35 m": (24.0, 36.0)}

    def __init__(self, bins_months=None, calibration_window=None, **kwargs):
        super().__init__(**kwargs)
        self.bins_months = dict(bins_months or self.default_bins_months)
        self.calibration_window = calibration_window
        self.person_years = dict.fromkeys(self.bins_months, 0.0)

    def step(self):
        if self.ti == 0:
            return
        window = self.calibration_window
        if window is None or window[0] <= years_elapsed(self) < window[1]:
            ages = self.sim.people.age.values * 12.0
            for label, (low, high) in self.bins_months.items():
                self.person_years[label] += np.count_nonzero((ages >= low) & (ages < high)) * self.t.dt_year

    @property
    def person_months(self):
        """Person-months per age bin"""
        return {label: years * 12.0 for label, years in self.person_years.items()}


class UidTracker(ss.Analyzer):
    """
    Values of disease states over time, for chosen agents and every strain

    Args:
        uids (int/list): the agents to track
        track_fields (str/list): the states to track (default: 'rel_sus')

    Example:
        ```python
        sim = rs.Sim(scenario='simple', analyzers=rs.UidTracker(uids=[0, 10, 20])).run()
        tracker = sim.analyzers.uidtracker
        tracker.get_field_matrix('G1P8', 'rel_sus') # Shape: (timepoints, uids)
        ```
    """

    def __init__(self, uids=None, track_fields="rel_sus", **kwargs):
        super().__init__(**kwargs)
        self.track_uids = ss.uids(np.atleast_1d([] if uids is None else uids).astype(int))
        self.track_fields = [track_fields] if isinstance(track_fields, str) else list(track_fields)

    def init_results(self):
        super().init_results()
        self.diseases = rotavirus_diseases(self.sim)
        shape = (self.t.npts, len(self.track_uids))
        self.data = {d.name: {field: np.full(shape, np.nan) for field in self.track_fields} for d in self.diseases}

    def step(self):
        for disease in self.diseases:
            for field in self.track_fields:
                self.data[disease.name][field][self.ti] = disease[field].raw[self.track_uids]

    def get_field_matrix(self, strain_name, field_name):
        """The values of a state for a strain, as an array of shape (timepoints, uids)"""
        return self.data[strain_name][field_name]

    def get_uid_ages(self):
        """The current ages of the tracked agents, in years"""
        return self.sim.people.age.raw[self.track_uids]

    def to_df(self):
        """The values as a dataframe, with columns named like `G1P8_rel_sus_10` for agent 10"""
        columns = {}
        for strain, fields in self.data.items():
            for field, matrix in fields.items():
                columns.update({f"{strain}_{field}_{uid}": matrix[:, i] for i, uid in enumerate(self.track_uids)})
        return pd.DataFrame(columns, index=self.timevec)

    def plot_field_heatmap(self, strain_name, field_name="rel_sus", cmap="viridis_r", **kwargs):
        """
        Plot a state as a heatmap, with one row per agent

        Args:
            strain_name (str): the strain, e.g. 'G1P8'
            field_name (str): the state
            cmap (str): the colormap
            kwargs (dict): passed to `plt.subplots()`

        Returns:
            The figure and axes
        """
        fig, ax = plt.subplots(**kwargs)
        image = ax.imshow(self.get_field_matrix(strain_name, field_name).T, aspect="auto", cmap=cmap, origin="lower", interpolation="nearest")
        ax.set_yticks(np.arange(len(self.track_uids)), [f"{uid}" for uid in self.track_uids])
        ax.set_xlabel("Time step")
        ax.set_ylabel("Agent")
        ax.set_title(f"{field_name} for {strain_name}")
        fig.colorbar(image, ax=ax, label=field_name)
        return fig, ax
