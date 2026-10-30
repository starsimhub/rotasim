"""
Test vaccination and the age-assortative network
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import rotasim as rs

pars = dict(n_agents=5000, start="2020-01-01", stop="2022-01-01", verbose=0)
demographics = [ss.Births(birth_rate=ss.peryear(30)), ss.Deaths(death_rate=ss.peryear(10))]


@sc.timer()
def test_vaccination():
    sc.heading("Testing vaccination")
    vax = rs.RotaVaccination(start_date="2020-03-01", uptake_dist=0.8)
    sim = rs.Sim(scenario="simple", override_prevalence=0, interventions=vax, demographics=sc.dcp(demographics), people=ss.People(pars["n_agents"], age_data=ss.uniform(0, 2)), **pars).run()
    vax = sim.interventions.rotavaccination
    summary = vax.get_vaccination_summary()
    assert summary["received_any_dose"] > 100  # Reaches infants
    assert 0.9 < summary["received_any_dose"] / summary["doses_offered"] < 1  # Refusing a dose doesn't stop the next being offered...
    uptake = vax.results.new_doses.sum() / vax.doses_offered.sum()
    assert 0.75 < uptake < 0.85  # ...and each dose is offered once, at the uptake given
    assert summary["doses_by_number"][2] > summary["doses_by_number"][1]  # Most reach two doses
    assert vax.results.new_doses[:50].sum() == 0  # Not before the start
    assert sim.people.age[vax.doses_received > 0].min() >= 0.1  # Not before the minimum age

    vaccinated = (vax.doses_received == 2).uids
    assert not np.isnan(sim.diseases.G1P8.rel_sus.values).any()  # Nobody's susceptibility is undefined
    assert sim.diseases.G1P8.rel_sus[vaccinated].mean() < sim.diseases.G2P4.rel_sus[vaccinated].mean()  # The vaccine strain is protected against most
    assert sim.diseases.G1P4.rel_sus[vaccinated].mean() < sim.diseases.G2P4.rel_sus[vaccinated].mean()  # Partial matches next
    assert sim.diseases.G1P8.rel_sus[(vax.doses_received == 0).uids].min() == 1  # And the unvaccinated are not protected

    ended = rs.RotaVaccination(start_date="2020-03-01", end_date="2020-06-01", n_doses=1, name="ended")
    sim = rs.Sim(scenario="single", interventions=ended, demographics=sc.dcp(demographics), **pars).run()
    doses = sim.interventions.ended.results.new_doses
    assert doses.sum() > 0 and doses[200:].sum() == 0  # Stops at the end date

    with pytest.raises(ValueError):
        rs.RotaVaccination(start_date="2020-01-01", n_doses=2, dose_effectiveness=[0.5])


@sc.timer()
def test_network():
    sc.heading("Testing the age-assortative network")
    net = rs.AgeAssortativeNet(n_contacts=8, assortativity=0.7, child_age_threshold=5)
    sim = rs.Sim(scenario="single", networks=net, **pars).init()
    sim.run_one_step()
    net = sim.networks.ageassortativenet
    n_agents = sim.people.n_alive
    assert np.isclose(2 * len(net) / n_agents, 8, rtol=0.05)  # Everyone has about n_contacts, counting both ends of each

    age = sim.people.age.raw
    p1_child = age[net.p1] < 5
    p2_child = age[net.p2] < 5
    n_children = (sim.people.age < 5).sum()
    assert np.isclose((p1_child.astype(int) + p2_child.astype(int)).sum() / n_children, 8, rtol=0.05)  # Children have n_contacts too
    assert np.isclose(2 * (p1_child & p2_child).sum() / (p1_child.astype(int) + p2_child.astype(int)).sum(), 0.7, rtol=0.05)  # With assortativity among them
    assert not np.any(net.p1 == net.p2)  # No contacts with oneself

    sim.run_one_step()
    assert np.isclose(2 * len(sim.networks.ageassortativenet) / n_agents, 8, rtol=0.05)  # Replaced each step, so they don't accumulate
    sim = rs.Sim(scenario="single", networks=rs.AgeAssortativeNet(), **pars).run()
    assert sim.diseases.G1P8.results.cum_infections[-1] > 0  # It transmits


if __name__ == "__main__":
    T = sc.timer()
    test_vaccination()
    test_network()
    T.toc()
