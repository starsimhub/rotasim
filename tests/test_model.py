"""
Test scenarios, sims, immunity, and reassortment
"""
import numpy as np
import pytest
import sciris as sc
import starsim as ss
import rotasim as rs

pars = dict(n_agents=1000, start="2000-01-01", stop="2001-01-01", verbose=0)


def make_sim(scenario="simple", **kwargs):
    """A sim with no initial infections, to be set up by hand"""
    return rs.Sim(scenario=scenario, override_prevalence=0, **sc.mergedicts(pars, kwargs)).init()


def recover(disease, uids):
    """Take agents through a whole infection"""
    disease.set_prognoses(uids)
    disease.immunity.record_recovery(disease, uids)
    disease.infected[uids] = False
    disease.recovered[uids] = True
    disease.susceptible[uids] = True


@sc.timer()
def test_scenarios():
    sc.heading("Testing scenarios")
    assert "baseline" in rs.list_scenarios()  # Built-in scenarios have descriptions
    for name in rs.list_scenarios():
        rs.Sim(scenario=name, **pars).init()  # Every built-in scenario can be set up

    assert rs.generate_gp_reassortments([(1, 8), (2, 4)]) == [(1, 4), (1, 8), (2, 4), (2, 8)]  # Every combination of G and P
    preferred = rs.generate_gp_reassortments([(1, 8), (2, 4), (4, 6)], use_preferred_partners=True)
    assert (1, 4) not in preferred and (2, 8) in preferred  # Filtered by preferred partners...
    assert (4, 6) in preferred  # ...but an initial strain is always kept

    sim = rs.Sim(scenario="simple", base_beta=0.2, override_fitness={(1, 8): 0.5, (1, 4): 0.1}, override_strains={(3, 8): dict(fitness=0.7, init_prevalence=0.001)}, **pars).init()
    assert len(sim.diseases) == 6  # Three G and two P
    assert np.isclose(sim.diseases.G1P8.pars.beta.value, 0.1)  # base_beta times fitness
    assert np.isclose(sim.diseases.G1P4.pars.beta.value, 0.02)  # A reassortant's fitness can be set too
    assert sim.diseases.G2P8.pars.init_prevalence == 0  # Reassortants start with no infections

    default_fitness = rs.Sim(scenario="realistic_competition", **pars).init()
    assert default_fitness.diseases.G1P4.pars.fitness == 0.05  # The scenario's default fitness

    for bad in [dict(), dict(strains={}), dict(strains={(1, 8, 2): {}}), dict(strains={(0, 8): {}}), "nonexistent", 5]:
        with pytest.raises(ValueError):
            rs.Sim(scenario=bad)


@sc.timer()
def test_sim():
    sc.heading("Testing a full sim")
    sim = rs.Sim(scenario="simple", demographics=[ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(10))], rand_seed=1, **pars).run()
    g1p8 = sim.diseases.G1P8
    assert g1p8.results.infection.cum[-1] > 0  # It spreads
    assert sim.results.births.birth.sum() > 0 and sim.results.died.sum() > 0  # With demographics
    assert sim.people.age.mean() > 0  # Agents age
    assert np.nanmax(g1p8.rel_sus.values) <= 1 and np.nanmin(g1p8.rel_sus.values) >= 0  # Susceptibility stays a fraction

    repeat = rs.Sim(scenario="simple", demographics=[ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(10))], rand_seed=1, **pars).run()
    assert np.array_equal(g1p8.results.infection, repeat.diseases.G1P8.results.infection)  # The same seed gives the same result

    msim = ss.parallel(rs.Sim(scenario="single", **pars), n_runs=2, workers=1)
    assert len(msim.sims) == 2  # Runs with Starsim's multi-run tools

    default = rs.Sim(n_agents=100, verbose=0)
    assert default.scenario == "baseline" and default.pars.dt.days == 1  # The default sim is daily
    plain = ss.Sim(diseases=rs.Rotavirus(G=1, P=8), networks="random", dt=ss.days(1), start="2000-01-01", stop="2000-03-01", n_agents=500, verbose=0).run()
    assert plain.diseases.G1P8.results.infection.cum[-1] > 0  # The disease works in a plain sim, with no connectors


@sc.timer()
def test_natural_history():
    sc.heading("Testing the course of an infection")
    sim = make_sim("single", base_beta=0, start="2000-01-01", stop="2000-04-01")  # With no transmission, so nobody is reinfected
    disease = sim.diseases.G1P8
    uids = ss.uids([0, 1, 2])
    disease.set_prognoses(uids)
    assert disease.infected[uids].all() and not disease.susceptible[uids].any()  # Infected, so not susceptible
    assert np.all(disease.rel_trans[uids] == 1)  # Fully infectious
    while sim.ti < 3:
        sim.run_one_step()
    assert disease.rel_trans[uids].max() == 1  # Still symptomatic at day 3
    while sim.ti < 9:
        sim.run_one_step()
    assert np.allclose(disease.rel_trans[uids], 0.1)  # Shedding less when asymptomatic
    assert disease.infected[uids].all()  # But still infected
    while sim.ti < 16:
        sim.run_one_step()
    assert not disease.infected[uids].any() and disease.recovered[uids].all()  # Recovered
    assert disease.susceptible[uids].all()  # Susceptible again, with reduced rel_sus
    assert disease.rel_sus[uids].max() < 0.1  # Well protected soon after recovery
    while sim.ti < 90:
        sim.run_one_step()
    assert 0.3 < disease.waned_immunity_efficacy[uids].mean() < 0.6  # Waned: about 13 weeks is the mean duration, and it has been 11


@sc.timer()
def test_cross_immunity():
    sc.heading("Testing protection between strains")
    sim = make_sim()
    immunity = sim.connectors.rotaimmunityconnector
    diseases = sim.diseases
    uid = ss.uids([0])
    recover(diseases.G1P8, uid)
    immunity.step()
    baseline = 1 - np.exp(-0.1)  # After one episode
    assert np.isclose(diseases.G1P8.rel_sus[uid][0], 0)  # Homotypic
    assert np.isclose(diseases.G1P4.rel_sus[uid][0], 0.5)  # Shares G
    assert np.isclose(diseases.G2P8.rel_sus[uid][0], 0.5)  # Shares P
    assert np.isclose(diseases.G2P4.rel_sus[uid][0], 1 - 0.3 * baseline)  # Complete heterotypic: unaffected by the immunity to other strains
    assert np.all(diseases.G2P4.rel_sus[ss.uids([1, 2])] == 1)  # Others are naive

    both = ss.uids([3])
    diseases.G1P8.set_prognoses(both)
    diseases.G2P4.set_prognoses(both)
    assert immunity.num_current_infections[both][0] == 2
    immunity.record_recovery(diseases.G1P8, both)
    assert immunity.num_recovered_infections[both][0] == 0  # One episode isn't over until its last strain is
    immunity.record_recovery(diseases.G2P4, both)
    assert immunity.num_recovered_infections[both][0] == 1  # A co-infection is one episode

    fixed = make_sim(connectors=rs.RotaImmunityConnector(use_fixed_susceptibility=True, sus_after_1=0.7, sus_after_2=0.5, sus_after_3plus=0.3))
    immunity = fixed.connectors.rotaimmunityconnector
    immunity.num_recovered_infections[ss.uids([0, 1, 2, 3])] = [1, 2, 3, 4]
    assert np.allclose(immunity.baseline_protection()[:5], [0.3, 0.5, 0.7, 0.7, 0])  # Protection by the number of episodes

    immunity.initialize_immunity(min_age=5, max_age=200, exposures_per_year=1)
    older = (fixed.people.age >= 10).uids
    assert immunity.num_recovered_infections[older].min() >= 10  # A history that depends on age


@sc.timer()
def test_maternal():
    sc.heading("Testing maternal immunity")
    assert np.allclose(rs.erlang_survival([0, 1], 1, 1), [1, np.exp(-1)])  # One stage is exponential
    assert rs.erlang_survival(0.5, 6, 1) > np.exp(-0.5)  # More stages hold on for longer...
    assert rs.erlang_survival(2.0, 6, 1) < np.exp(-2.0)  # ...and then fall faster

    newborn_pars = dict(people=ss.People(pars["n_agents"], age_data=ss.uniform(0, 2)))
    for model in ["erlang", "titer"]:
        sim = make_sim(connectors=rs.RotaImmunityConnector(maternal_immunity_efficacy=0.9, maternal_immunity_model=model), **newborn_pars)
        immunity = sim.connectors.rotaimmunityconnector
        protection = immunity.maternal_protection()
        ages = sim.people.age.values
        assert protection.max() <= 0.9  # Limited by the efficacy
        assert protection[ages < 0.1].mean() > 3 * protection[ages > 1].mean()  # It wanes with age
        immunity.step()
        assert np.allclose(sim.diseases.G1P8.rel_sus.values, 1 - protection)  # It protects against every strain

    assert len(np.unique(immunity.maternal_titer_z.values)) > 1  # Each infant has a titer of their own
    before = immunity.maternal_titer_z.values.copy()
    immunity.step()
    assert np.array_equal(before, immunity.maternal_titer_z.values)  # It stays the same over time


@sc.timer()
def test_reassortment():
    sc.heading("Testing reassortment")
    sim = make_sim(connectors=[rs.RotaImmunityConnector(), rs.RotaReassortmentConnector(reassortment_prob=1.0)])
    connector = sim.connectors.rotareassortmentconnector
    diseases = sim.diseases
    uids = ss.uids(np.arange(100))
    diseases.G1P8.set_prognoses(uids)
    diseases.G2P4.set_prognoses(uids)
    diseases.G1P8.set_prognoses(ss.uids([200]))  # Not co-infected, so no reassortment
    connector.step()
    n_G1P4, n_G2P8 = diseases.G1P4.infected.sum(), diseases.G2P8.infected.sum()
    assert n_G1P4 + n_G2P8 == 100  # Each host gets one reassortant
    assert n_G1P4 > 25 and n_G2P8 > 25  # Chosen at random
    assert not diseases.G1P4.infected[ss.uids([200])][0]

    connector.step()
    assert diseases.G1P4.infected.sum() + diseases.G2P8.infected.sum() == 200  # A second step gives each host the other one
    connector.step()
    assert diseases.G1P4.infected.sum() + diseases.G2P8.infected.sum() == 200  # Then there are no new combinations

    preferred = make_sim(use_preferred_partners=True, connectors=[rs.RotaImmunityConnector(), rs.RotaReassortmentConnector(reassortment_prob=1.0)])
    assert "G1P4" not in preferred.diseases  # Not a preferred pairing, so it can't arise
    preferred.diseases.G1P8.set_prognoses(uids)
    preferred.diseases.G2P4.set_prognoses(uids)
    preferred.connectors.rotareassortmentconnector.step()
    assert preferred.diseases.G2P8.infected.sum() == 100


if __name__ == "__main__":
    T = sc.timer()
    test_scenarios()
    test_sim()
    test_natural_history()
    test_cross_immunity()
    test_maternal()
    test_reassortment()
    T.toc()
