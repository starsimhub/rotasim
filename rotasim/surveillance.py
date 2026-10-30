"""
Observation models: what a study or surveillance system would see of the simulated infections

These analyzers only observe, so they have no effect on transmission. Each one turns infections into
symptomatic cases with a `SymptomModel`, and gives its outputs as a dict from `results_dict()`.
"""

# Third-party imports
import numpy as np
import sciris as sc
import starsim as ss

# Local imports
from .analyzers import new_infections, rotavirus_diseases, years_elapsed

__all__ = ["SymptomModel", "MALEDTargets", "MALEDCohort", "Surveillance"]

maled_labels = ["<6 m", "6-11 m", "12-23 m", "24-35 m"]
maled_edges = np.array([0.0, 6.0, 12.0, 24.0, 36.0])  # Age bins in months
monthly_interval = 30.4375  # Days between scheduled stool samples, under one year of age
quarterly_interval = 91.3125  # Days between scheduled stool samples, from one year of age


def rate_per_100(cases, person_months):
    """Cases per 100 person-months, by age bin"""
    return {b: cases[b] / person_months[b] * 100.0 if person_months[b] > 0 else 0.0 for b in maled_labels}


class SymptomModel:
    """
    The probability that an infection is symptomatic

    Args:
        symptom_model (str): one of:

            - 'infection_number': by infection order, from `p_symp_1`, `p_symp_2`, and `p_symp_3plus`
            - 'age_only': logistic in age a in months, capped at 60, with `beta0 + beta1*(a-12) + beta2*(a-12)^2`
            - 'age_and_infection': as 'age_only', plus `beta3*order`, with order capped at 5
            - 'age_binned': by age bin, from `p_symp_age_0_6`, `p_symp_age_6_11`, and `p_symp_age_12plus`
    """

    models = ["infection_number", "age_only", "age_and_infection", "age_binned"]
    keys = ["symptom_model", "p_symp_1", "p_symp_2", "p_symp_3plus", "beta0", "beta1", "beta2", "beta3", "p_symp_age_0_6", "p_symp_age_6_11", "p_symp_age_12plus"]

    def __init__(self, symptom_model="infection_number", p_symp_1=1.0, p_symp_2=1.0, p_symp_3plus=1.0, beta0=0.0, beta1=0.0, beta2=0.0, beta3=0.0, p_symp_age_0_6=0.5, p_symp_age_6_11=0.5, p_symp_age_12plus=0.5):
        if symptom_model not in self.models:
            raise ValueError(f"symptom_model must be one of {sc.strjoin(self.models)}, not {symptom_model!r}")
        self.model = symptom_model
        self.p_order = np.array([p_symp_1, p_symp_2, p_symp_3plus])
        self.betas = (beta0, beta1, beta2, beta3)
        self.p_age = (p_symp_age_0_6, p_symp_age_6_11, p_symp_age_12plus)

    @classmethod
    def pop_from(cls, kwargs):
        """Make a symptom model, taking its arguments out of an analyzer's keyword arguments"""
        return cls(**{key: kwargs.pop(key) for key in cls.keys if key in kwargs})

    def prob(self, age_months, order):
        """The probability of symptoms, given age in months and infection order (1 for a first infection)"""
        age_months = np.asarray(age_months, dtype=float)
        order = np.asarray(order, dtype=float)
        if self.model == "infection_number":
            return self.p_order[np.minimum(order, 3).astype(int) - 1]
        elif self.model == "age_binned":
            return np.select([age_months < 6, age_months < 12], self.p_age[:2], default=self.p_age[2])

        beta0, beta1, beta2, beta3 = self.betas
        centered = np.minimum(age_months, 60.0) - 12.0
        logit = beta0 + beta1 * centered + beta2 * centered**2
        if self.model == "age_and_infection":
            logit = logit + beta3 * np.minimum(order, 5.0)
        return 1.0 / (1.0 + np.exp(-np.clip(logit, -30, 30)))


class MALEDTargets(ss.Analyzer):
    """
    MAL-ED calibration targets, with memory use that doesn't grow with the number of infections

    Keeps only case counts by age bin, person-time by age bin, each agent's age at first detection, and
    prevalence. An infection is detected if it is symptomatic and reported, or otherwise with probability
    `p_asymp_detect`.

    Args:
        calibration_window (tuple): start and stop, in years from the start of the sim
        beta0, beta1, beta2 (float): parameters of the 'age_only' symptom model (see `SymptomModel`)
        reporting_rate (float): probability that a symptomatic infection is reported
        constant_severity (float): probability that a symptomatic infection is severe enough to be reported
        p_asymp_detect (float): probability of detecting an infection that is not a reported case
        censor_at_months (float): age above which infections are not counted
    """

    def __init__(self, calibration_window, beta0, beta1, beta2, reporting_rate=1.0, constant_severity=1.0, p_asymp_detect=0.4, censor_at_months=36.0, **kwargs):
        super().__init__(**kwargs)
        self.window = calibration_window
        self.symptoms = SymptomModel("age_only", beta0=beta0, beta1=beta1, beta2=beta2)
        self.p_reported = reporting_rate * constant_severity
        self.p_asymp = p_asymp_detect
        self.censor = censor_at_months
        self.rng_symp = ss.random(name="rng_symp")
        self.rng_report = ss.random(name="rng_report")
        self.rng_detect = ss.random(name="rng_detect")
        self.cases = np.zeros(len(maled_labels), dtype=int)
        self.person_years = np.zeros(len(maled_labels))
        self.prev = []  # Prevalence in the window, by step
        self.define_states(ss.FloatArr("first_detected", default=np.nan))  # Age in months at first detection

    def step(self):
        people = self.sim.people
        if self.ti == 0 or not self.window[0] <= years_elapsed(self) < self.window[1]:
            return

        ages = people.age.values * 12.0
        counts, _ = np.histogram(ages, maled_edges)
        self.person_years += counts * self.t.dt_year
        infected = sum(d.infected.values.astype(int) for d in rotavirus_diseases(self.sim))
        self.prev.append(np.count_nonzero(infected) / len(ages))

        uids = new_infections(self)
        uids = uids[people.age[uids] * 12.0 <= self.censor]
        if len(uids):
            age_months = people.age[uids] * 12.0
            symptomatic = self.rng_symp.rvs(uids) < self.symptoms.prob(age_months, np.ones(len(uids)))
            reported = symptomatic & (self.rng_report.rvs(uids) < self.p_reported)
            self.cases += np.histogram(age_months[reported], maled_edges)[0]
            detected = reported | (self.rng_detect.rvs(uids) < self.p_asymp)
            first = uids[detected]
            first = first[np.isnan(self.first_detected[first])]
            self.first_detected[first] = people.age[first] * 12.0

    def results_dict(self):
        """Incidence by age bin, quartiles of age at first detection, and prevalence"""
        cases = dict(zip(maled_labels, self.cases.tolist()))
        person_months = dict(zip(maled_labels, (self.person_years * 12.0).tolist()))
        ages = self.first_detected.raw[~np.isnan(self.first_detected.raw)]
        q25, median, q75 = np.quantile(ages, [0.25, 0.5, 0.75]) if len(ages) else [np.nan] * 3
        half = len(self.prev) // 2
        prev_drift = np.mean(self.prev[half:]) - np.mean(self.prev[:half]) if half else np.nan
        return dict(
            ir=rate_per_100(cases, person_months),
            cases=cases,
            person_months=person_months,
            fi_q25=q25,
            fi_median=median,
            fi_q75=q75,
            fi_n=len(ages),
            prev_mean=np.mean(self.prev) if self.prev else np.nan,
            prev_drift=prev_drift,
        )


class MALEDCohort(ss.Analyzer):
    """
    The MAL-ED birth cohort study, as observed from the sim

    Newborns are enrolled during `enroll_window`, and each is followed to an age drawn from `censoring_ages`.
    Symptoms depend on the true number of infections so far. A symptomatic infection is detected if a
    stool is collected and tests positive, at `symp_collection * eia_sensitivity`, which doesn't depend on
    age. An asymptomatic one is detected if a scheduled stool falls in its shedding period, which is less
    likely from one year of age, when sampling moves from monthly to quarterly.

    Args:
        censoring_ages (array): each child's age at study exit in the data, in months
        enroll_window (tuple): start and stop of enrollment, in years from the start of the sim
        symp_collection (float): probability that a stool is collected from a symptomatic infection
        eia_sensitivity (float): sensitivity of the assay
        shed_days (float): days of shedding, for scheduled sampling
        log_events (bool): whether to keep every infection of the cohort, in `events`
        kwargs (dict): arguments for `SymptomModel`, and for the analyzer
    """

    def __init__(self, censoring_ages, enroll_window=(5.0, 7.5), symp_collection=0.80, eia_sensitivity=0.85, shed_days=13.0, log_events=False, **kwargs):
        self.symptoms = SymptomModel.pop_from(kwargs)
        super().__init__(**kwargs)
        self.enroll_window = enroll_window
        self.censoring_ages = np.asarray(censoring_ages, dtype=float)
        self.p_detect_symp = symp_collection * eia_sensitivity
        self.eia = eia_sensitivity
        self.shed = shed_days
        self.log_events = log_events
        self.events = []
        self.rng_exit = ss.random(name="rng_exit")
        self.rng_symp = ss.random(name="rng_symp")
        self.rng_detect = ss.random(name="rng_detect")
        self.person_years = np.zeros(len(maled_labels))
        self.cases_all = np.zeros(len(maled_labels), dtype=int)
        self.cases_symp = np.zeros(len(maled_labels), dtype=int)
        self.define_states(
            ss.BoolArr("enrolled", default=False),
            ss.FloatArr("exit_age", default=np.nan),  # Age at study exit, in months
            ss.FloatArr("last_age", default=0.0),  # Oldest age seen in the study, in months
            ss.IntArr("n_inf", default=0),  # True number of infections so far
            ss.IntArr("n_det", default=0),  # Number of infections detected
            ss.FloatArr("first_infected", default=np.nan),  # Age at first infection, in months
            ss.FloatArr("first_detected", default=np.nan),  # Age at first detection, in months
        )

    def init_results(self):
        super().init_results()
        # Agents primed as neonates by an intervention (e.g. NeonatalPriming) count their next infection one higher
        primed = next((iv for iv in self.sim.interventions.values() if hasattr(iv, "order_primed_uids")), None)
        self.order_primed = primed.order_primed_uids if primed is not None else set()

    def p_scheduled(self, age_months):
        """The probability that a scheduled stool falls during an asymptomatic infection"""
        interval = np.where(age_months < 12, monthly_interval, quarterly_interval)
        return np.minimum(self.shed / interval, 1.0)

    def enroll(self):
        """Enroll the agents born this step"""
        people = self.sim.people
        newborns = (people.age <= self.t.dt_year).uids
        newborns = newborns[~self.enrolled[newborns]]
        if len(newborns):
            self.enrolled[newborns] = True
            picks = (self.rng_exit.rvs(newborns) * len(self.censoring_ages)).astype(int)
            self.exit_age[newborns] = self.censoring_ages[picks]

    def step(self):
        if self.ti == 0:
            return
        if self.enroll_window[0] <= years_elapsed(self) < self.enroll_window[1]:
            self.enroll()

        people = self.sim.people
        cohort = self.enrolled.uids
        age_months = people.age[cohort] * 12.0
        following = age_months <= self.exit_age[cohort]
        self.person_years += np.histogram(age_months[following], maled_edges)[0] * self.t.dt_year
        self.last_age[cohort[following]] = age_months[following]

        uids = new_infections(self)
        uids = uids[self.enrolled[uids]]
        uids = uids[people.age[uids] * 12.0 <= self.exit_age[uids]]
        if not len(uids):
            return

        self.n_inf[uids] += 1
        age_months = people.age[uids] * 12.0
        primed = np.isin(uids, list(self.order_primed)) if self.order_primed else np.zeros(len(uids), dtype=bool)
        order = self.n_inf[uids] + primed
        first = np.isnan(self.first_infected[uids])
        self.first_infected[uids[first]] = age_months[first]

        symptomatic = self.rng_symp.rvs(uids) < self.symptoms.prob(age_months, order)
        p_detect = np.where(symptomatic, self.p_detect_symp, self.p_scheduled(age_months) * self.eia)
        detected = self.rng_detect.rvs(uids) < p_detect

        if self.log_events:
            self.events += [dict(uid=int(u), age_m=a, order=int(o), symp=bool(s), detected=bool(d)) for u, a, o, s, d in zip(uids, age_months, order, symptomatic, detected)]

        found = uids[detected]
        self.n_det[found] += 1
        self.cases_all += np.histogram(age_months[detected], maled_edges)[0]
        self.cases_symp += np.histogram(age_months[detected & symptomatic], maled_edges)[0]
        first = np.isnan(self.first_detected[found])
        self.first_detected[found[first]] = age_months[detected][first]

    def results_dict(self):
        """Incidence by age bin, the time to first detection for a survival analysis, and the share of children detected"""
        cohort = np.flatnonzero(self.enrolled.raw)
        det = self.first_detected.raw[cohort]
        first_infected = self.first_infected.raw[cohort]
        observed = ~np.isnan(det)
        ever_infected = ~np.isnan(first_infected)
        n_det = self.n_det.raw[cohort]
        time = np.where(observed, det, np.minimum(self.last_age.raw[cohort], self.exit_age.raw[cohort]))
        person_months = dict(zip(maled_labels, (self.person_years * 12.0).tolist()))
        return dict(
            n_enrolled=len(cohort),
            km_time=time.tolist(),
            km_observed=observed.astype(int).tolist(),
            ir_symp=rate_per_100(dict(zip(maled_labels, self.cases_symp)), person_months),
            ir_all=rate_per_100(dict(zip(maled_labels, self.cases_all)), person_months),
            frac_ever_detected=observed.mean() if len(cohort) else np.nan,
            frac_ever_infected=ever_infected.mean() if len(cohort) else np.nan,
            repeat_detected_frac=(n_det[observed] >= 2).mean() if observed.any() else 0.0,
            true_first_median=np.median(first_infected[ever_infected]) if ever_infected.any() else np.nan,
        )


class Surveillance(ss.Analyzer):
    """
    Cross-sectional surveillance (e.g. UK or Belgian sentinel data): the age distribution of symptomatic cases

    By default the last age bin is open, so it counts cases of any older age. With `cap_age_m`, cases
    at or above that age are left out, and the bins are the intervals between the edges. This avoids applying
    the symptom model to adults, whom real surveillance mostly doesn't capture.

    If the sim has a vaccine intervention with `dose_ages` and `dose_n` (a dict of doses by UID), it also
    gives direct vaccine effectiveness, comparing the fully vaccinated with those who have had no doses.

    Args:
        bin_edges_m (tuple): left edges of the age bins, in months
        cap_age_m (float): the age from which cases are left out, in months
        window (tuple): start and stop of observation, in years from the start of the sim
        kwargs (dict): arguments for `SymptomModel`, and for the analyzer
    """

    def __init__(self, bin_edges_m=(0.0, 6.0, 12.0, 24.0, 36.0), cap_age_m=None, window=(5.0, 10.0), **kwargs):
        self.symptoms = SymptomModel.pop_from(kwargs)
        super().__init__(**kwargs)
        self.edges = np.asarray(bin_edges_m, dtype=float)
        self.cap = cap_age_m
        self.nbins = len(self.edges) - 1 if self.cap is not None else len(self.edges)
        self.window = window
        self.rng_symp = ss.random(name="rng_symp")
        self.cases = np.zeros(self.nbins, dtype=int)
        self.person_years = np.zeros(self.nbins)
        self.cases_vax = np.zeros(self.nbins, dtype=int)
        self.cases_unvax = np.zeros(self.nbins, dtype=int)
        self.py_vax = np.zeros(self.nbins)
        self.py_unvax = np.zeros(self.nbins)
        self.first_inf_ages = []  # Age at first infection of those born during the sim, in months
        self.sero_bands = dict(sero_2y=(24.0, 36.0), sero_5to6y=(60.0, 84.0))  # Age bands for seroprevalence, in months
        self.sero_pos = dict.fromkeys(self.sero_bands, 0)
        self.sero_total = dict.fromkeys(self.sero_bands, 0)

    def init_results(self):
        super().init_results()
        self.immunity = rotavirus_diseases(self.sim)[0].immunity
        self.vaccine = next((iv for iv in self.sim.interventions.values() if hasattr(iv, "dose_n")), None)
        self.min_vax_age = 0.0
        if self.vaccine is not None:
            self.min_vax_age = max(self.vaccine.dose_ages) * 12.0
            self.vax_doses = getattr(self.vaccine, "n_doses_target", len(self.vaccine.dose_ages))

    def bin_of(self, age_months):
        """The age bin of each age, in months"""
        return np.clip(np.digitize(age_months, self.edges) - 1, 0, self.nbins - 1)

    def vaccine_status(self, uids):
        """Whether each agent is fully vaccinated, and whether they have had no doses"""
        doses = self.vaccine.dose_n
        full = np.fromiter((u for u, n in doses.items() if n >= self.vax_doses), dtype=int)
        return np.isin(uids, full), ~np.isin(uids, np.fromiter(doses, dtype=int))

    def observe_population(self):
        """Add to the person-time, and to the seroprevalence counts, which include any ages above the cap"""
        people = self.sim.people
        uids = people.auids
        age_months = people.age.values * 12.0
        seropositive = self.immunity.num_recovered_infections.values >= 1
        for band, (low, high) in self.sero_bands.items():
            in_band = (age_months >= low) & (age_months < high)
            self.sero_pos[band] += np.count_nonzero(seropositive & in_band)
            self.sero_total[band] += np.count_nonzero(in_band)

        if self.cap is not None:
            keep = age_months < self.cap
            uids, age_months = uids[keep], age_months[keep]
        bins = self.bin_of(age_months)
        self.person_years += np.bincount(bins, minlength=self.nbins) * self.t.dt_year
        if self.vaccine is not None:
            full, none = self.vaccine_status(uids)
            eligible = age_months >= self.min_vax_age
            self.py_vax += np.bincount(bins[eligible & full], minlength=self.nbins) * self.t.dt_year
            self.py_unvax += np.bincount(bins[eligible & none], minlength=self.nbins) * self.t.dt_year

    def observe_infections(self):
        """Count this step's infections as symptomatic cases, and note first infections"""
        people = self.sim.people
        uids = new_infections(self)
        if not len(uids):
            return
        age_months = people.age[uids] * 12.0
        order = self.immunity.num_recovered_infections[uids] + 1.0

        born_in_sim = people.age[uids] <= years_elapsed(self)
        self.first_inf_ages += age_months[(order < 1.5) & born_in_sim].tolist()

        symptomatic = self.rng_symp.rvs(uids) < self.symptoms.prob(age_months, order)
        cases, ages = uids[symptomatic], age_months[symptomatic]
        if self.cap is not None:
            keep = ages < self.cap
            cases, ages = cases[keep], ages[keep]
        bins = self.bin_of(ages)
        self.cases += np.bincount(bins, minlength=self.nbins)
        if self.vaccine is not None:
            full, none = self.vaccine_status(cases)
            eligible = ages >= self.min_vax_age
            self.cases_vax += np.bincount(bins[eligible & full], minlength=self.nbins)
            self.cases_unvax += np.bincount(bins[eligible & none], minlength=self.nbins)

    def step(self):
        if self.ti > 0 and self.window[0] <= years_elapsed(self) < self.window[1]:
            self.observe_population()
            self.observe_infections()

    def results_dict(self):
        """Cases, incidence per 100 child-years, and population by age bin, plus seroprevalence and direct effectiveness"""
        total = self.cases.sum()
        py = self.person_years
        out = dict(
            case_counts=self.cases.tolist(),
            case_proportions=(self.cases / total if total else np.zeros(self.nbins)).tolist(),
            total_cases=int(total),
            person_years=py.tolist(),
            ir_per_100cy=np.divide(self.cases * 100.0, py, out=np.zeros(self.nbins), where=py > 0).tolist(),
            pop_fraction=(py / py.sum() if py.sum() else np.zeros(self.nbins)).tolist(),
            first_inf_median_months=np.median(self.first_inf_ages) if self.first_inf_ages else np.nan,
            n_first_infs=len(self.first_inf_ages),
        )
        for band in self.sero_bands:
            out[band] = self.sero_pos[band] / self.sero_total[band] if self.sero_total[band] else np.nan
        if self.vaccine is not None:
            rate_vax = np.divide(self.cases_vax, self.py_vax, out=np.full(self.nbins, np.nan), where=self.py_vax > 0)
            rate_unvax = np.divide(self.cases_unvax, self.py_unvax, out=np.full(self.nbins, np.nan), where=self.py_unvax > 0)
            out.update(cases_vax=self.cases_vax.tolist(), cases_unvax=self.cases_unvax.tolist(), py_vax=self.py_vax.tolist(), py_unvax=self.py_unvax.tolist(), ve_direct=(1.0 - rate_vax / rate_unvax).tolist(), min_vax_age_m=self.min_vax_age)
        return out
