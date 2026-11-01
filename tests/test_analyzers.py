"""
Test analyzers and observation models
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import rotasim as rs

pars = dict(n_agents=2000, start="2000-01-01", stop="2003-01-01", verbose=0)


def make_demographics():
    return [ss.Births(birth_rate=ss.peryear(30)), ss.Deaths(death_rate=ss.peryear(10))]


@sc.timer()
def test_analyzers():
    sc.heading("Testing analyzers")
    analyzers = [rs.StrainStats(), rs.EventStats(), rs.AgeStats(), rs.InfectedStrainStats(), rs.PersonTimeByAge(calibration_window=(1, 2)), rs.UidTracker(uids=[0, 5], track_fields=["rel_sus", "infected"])]
    sim = rs.Sim(scenario="simple", analyzers=analyzers, demographics=make_demographics(), **pars).run()

    res = sim.analyzers.strainstats.results
    total = sum(res[f"{name}_count"] for name in sim.diseases.keys())
    assert np.array_equal(total, sum(d.results.infected for d in sim.diseases.values()))  # Counts match the diseases
    assert np.nanmax(sum(res[f"{name}_proportion"] for name in sim.diseases.keys())) < 1.001  # Proportions add up to 1

    events = sim.analyzers.eventstats.results
    assert events.births.sum() == sim.results.births.birth.sum()
    assert events.new_infections.sum() == sum(d.results.infection.sum() for d in sim.diseases.values())
    assert events.reassortments.sum() > 0  # With two strains circulating, reassortants arise
    assert (events.infected_agents >= events.coinfected_agents).all()

    ages = sim.analyzers.agestats.results
    assert sum(ages[label][-1] for label in rs.AgeStats.age_labels) == sim.people.n_alive  # Everyone is in one age group

    infections = sim.analyzers.infectedstrainstats.to_df()
    seeds = sum(int(d.pars.init_prevalence * pars["n_agents"]) for d in sim.diseases.values())
    assert len(infections) > seeds  # Seeds and transmission are both logged
    assert (infections.Strain == "G1P8A1B1").any() and infections.CollectionTime.max() > 2
    assert infections.severity.isin(rs.InfectedStrainStats.severity_rates).all()

    person_months = sim.analyzers.persontimebyage.person_months
    assert person_months["<6 m"] > 0
    expected = np.count_nonzero(sim.people.age.values < 0.5) * 12  # About that many people for a year
    assert 0.5 * expected < person_months["<6 m"] < 1.5 * expected

    tracker = sim.analyzers.uidtracker
    assert tracker.get_field_matrix("G1P8", "rel_sus").shape == (sim.t.npts, 2)
    assert tracker.to_df().shape == (sim.t.npts, 2 * 2 * len(sim.diseases))
    assert len(tracker.get_uid_ages()) == 2
    tracker.plot_field_heatmap("G1P8")


@sc.timer()
def test_symptoms():
    sc.heading("Testing symptom models")
    by_order = rs.SymptomModel(p_symp_1=0.6, p_symp_2=0.4, p_symp_3plus=0.2)
    assert np.allclose(by_order.prob([10, 10, 10, 10], [1, 2, 3, 7]), [0.6, 0.4, 0.2, 0.2])  # By infection order

    by_age = rs.SymptomModel("age_binned", p_symp_age_0_6=0.1, p_symp_age_6_11=0.5, p_symp_age_12plus=0.9)
    assert np.allclose(by_age.prob([3, 8, 20], [1, 1, 1]), [0.1, 0.5, 0.9])  # By age bin

    logistic = rs.SymptomModel("age_only", beta0=0)
    assert np.allclose(logistic.prob([12, 40], [1, 2]), 0.5)  # A logistic of 0
    with_order = rs.SymptomModel("age_and_infection", beta3=-1)
    assert with_order.prob([12], [1]) > with_order.prob([12], [3])  # Later infections are milder

    with pytest.raises(ValueError):
        rs.SymptomModel("nonexistent")


@sc.timer()
def test_observers():
    sc.heading("Testing observation models")
    targets = rs.MALEDTargets(calibration_window=(1, 3), beta0=0, beta1=-0.05, beta2=0)
    cohort = rs.MALEDCohort(censoring_ages=[24, 30, 36], enroll_window=(1, 2), p_symp_1=0.6, p_symp_2=0.4, p_symp_3plus=0.2, log_events=True)
    surveillance = rs.Surveillance(window=(1, 3), cap_age_m=36, p_symp_1=0.6, p_symp_2=0.4, p_symp_3plus=0.2)
    immunity = rs.RotaImmunityConnector(use_fixed_susceptibility=True)
    kwargs = sc.mergedicts(pars, dict(n_agents=5000))
    sim = rs.Sim(scenario="single", analyzers=[targets, cohort, surveillance], connectors=immunity, networks=ss.RandomNet(n_contacts=7), demographics=make_demographics(), **kwargs).run()

    out = sim.analyzers.maledtargets.results_dict()
    assert sum(out["cases"].values()) > 0 and out["ir"]["12-23 m"] > 0  # Cases, as a rate
    assert 0 < out["prev_mean"] < 1 and out["fi_n"] > 0
    assert 0 < out["fi_q25"] <= out["fi_median"] <= out["fi_q75"] <= 36

    out = sim.analyzers.maledcohort.results_dict()
    assert out["n_enrolled"] > 50  # About a year of births
    assert 0 < out["frac_ever_detected"] <= out["frac_ever_infected"] <= 1  # Not every infection is detected
    assert len(out["km_time"]) == out["n_enrolled"] and max(out["km_time"]) <= 36  # Followed to their exit age at most
    assert sum(out["ir_symp"].values()) <= sum(out["ir_all"].values())  # Detected cases include those without symptoms
    events = sim.analyzers.maledcohort.events
    assert len(events) > 0 and max(e["age_m"] for e in events) <= 36

    out = sim.analyzers.surveillance.results_dict()
    assert out["total_cases"] > 0 and len(out["case_counts"]) == 4  # Four bins with a cap
    assert np.isclose(sum(out["case_proportions"]), 1) and np.isclose(sum(out["pop_fraction"]), 1)
    assert 0 < out["sero_2y"] <= 1 and out["n_first_infs"] > 0


if __name__ == "__main__":
    T = sc.timer()
    test_analyzers()
    test_symptoms()
    test_observers()
    T.toc()
