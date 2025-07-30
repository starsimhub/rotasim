import rotasim as rs
import sciris as sc
import pytest


def test_strainstats():
    # Test running sims
    sc.heading("Test StrainStats analyzer")
    sim = rs.Sim(N=10_000, timelimit=4, analyzers=rs.StrainStats())
    events = sim.run()

    strains = sim.connectors.rota.pars.segment_combinations

    # Proportion at all time steps should sum to 1

    for i in range(len(sim.results.strainstats.timevec)):
        prop_sum = 0
        for strain in strains:
            prop_sum += sim.results.strainstats[f"{strain} proportion"][i]
        assert pytest.approx(prop_sum, 0.001) == 1

    # df = sim.analyzers.strainstats.to_df()
    # plots = sim.results.strainstats.plot()
    print(sim.analyzers.strainstats.get_strain_proportions("(1, 8, 1, 1)"))

    return


test_strainstats()
