"""
Check results against baseline values.

NB: the two tests could be combined into one, but are left separate for clarity.
"""
import numpy as np
import sciris as sc
import rotasim as rs
import starsim as ss

N = 2_000
timelimit = 10
verbose = False


def test_default(make=False):
    sc.heading("Testing default parameters")
    filename = "test_events_default.json"

    # Generate new baseline
    if make:
        with sc.timer() as T:
            rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose)
            rota.run()
            events = rota.connectors["rota"].event_dict
        sc.savejson(filename, events)

    # Check old baseline
    else:
        with sc.timer() as T:
            rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose)
            rota.run()
            events = rota.connectors["rota"].event_dict
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, "Events do not match for default simulation"
        print(f"Defaults matched:\n{events}")

    return


def test_alt(make=False):
    sc.heading("Testing alternate parameters")
    filename = "test_events_alt.json"

    rota_inputs = dict(
        reassortment_rate=0.2,
        fitness_hypothesis=2,
        omega=365 / 50,
        initial_immunity=True,
        experiment_number=2,
    )

    intervention_inputs = dict(
        ve_i_to_ve_s_ratio=0.5,
    )

    # Generate new baseline
    if make:
        rota = rs.Rota(**rota_inputs)
        rotaVaxProg = rs.RotaVaxProg(**intervention_inputs)
        sim = rs.Sim(
            n_agents=N,
            to_csv=False,
            timelimit=timelimit,
            interventions=rotaVaxProg,
            connectors=rota,
            rand_seed=rota_inputs["experiment_number"],
            verbose=verbose,
        )
        sim.run()
        events = sim.connectors["rota"].event_dict
        sc.savejson(filename, events)

    # Check old baseline
    else:
        rota = rs.Rota(**rota_inputs)
        rotaVaxProg = rs.RotaVaxProg(**intervention_inputs)
        sim = rs.Sim(
            n_agents=N,
            to_csv=False,
            timelimit=timelimit,
            interventions=rotaVaxProg,
            connectors=rota,
            rand_seed=rota_inputs["experiment_number"],
            verbose=verbose,
        )
        sim.run()
        events = sim.connectors["rota"].event_dict
        saved = sc.objdict(sc.loadjson(filename))
        assert events == saved, (
            "Events do not match for alternate parameters simulation"
        )
        print(f"Alternate parameters matched:\n{events}")

    return

def test_vx_intervention():
    """
    Test the RotaVax intervention with default parameters.
    * Should run without errors.
    * Vx strain should be whatever the dominant strain is at start of vx campaign
    """
    interventions = []
    rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose, interventions=interventions)
    rota.run()

    # verify that a RotaVaxProg intervention was created
    assert len(rota.interventions) > 0, "No interventions were created"
    assert isinstance(rota.interventions[0], rs.RotaVaxProg), "First intervention is not a RotaVaxProg"


def test_vx_scheduling():
    """
    Test the timing of the RotaVax intervention. Only run it on a specific year
    """
    interventions = [rs.RotaVaxProg(start_date="2001-1-1", end_date="2001-12-31")]
    rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose, interventions=interventions)
    rota.run(until=2002)

    # there should be no vaccinations in 2000 and 2002
    assert np.sum(rota.results.rotavaxprog['new_vaccinated_first_dose'][0:365]) == 0, "Unexpected vaccinations in 2000"
    assert np.sum(rota.results.rotavaxprog['new_vaccinated_first_dose'][365*2:]) == 0, "Unexpected vaccinations in 2002"

    # there should be vaccinations in 2001
    assert np.sum(rota.results.rotavaxprog['new_vaccinated_first_dose'][365:365*2]) > 0, "No vaccinations in 2001"


def test_vx_multivalent():
    """
    Test multi-strain vaccine.
    """
    interventions = [rs.RotaVaxProg(start_date="2001-01-01", end_date="2002-01-01", vx_strains=['G1', 'P1', 'G2', 'P2'])]
    rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose, interventions=interventions)
    rota.run(until=2002)

    # Check that the vaccine was administered for all strains
    assert rota.interventions.rotavaxprog.product.is_match("11"), "Vaccine product does not match expected multi-strain vaccine"
    assert rota.interventions.rotavaxprog.product.is_match("12"), "Vaccine product does not match expected multi-strain vaccine"
    assert rota.interventions.rotavaxprog.product.is_match("21"), "Vaccine product does not match expected multi-strain vaccine"
    assert rota.interventions.rotavaxprog.product.is_match("22"), "Vaccine product does not match expected multi-strain vaccine"


def test_vx_waning():
    """
    Test waning and waning delay of the RotaVax intervention.
    """
    interventions = [rs.RotaVaxProg(start_date="2001-01-01", waning_delay=ss.dur(100, unit="days")) ,]
    rota = rs.Sim(N=N, timelimit=timelimit, verbose=verbose, interventions=interventions)
    rota.run(until=2001.25)

    # All vaccinations so for should not have waned yet
    assert np.all(rota.people.rotavaxprog.waned_effectiveness[rota.people.rotavaxprog.vaccinated] ), "Not all vaccinated have waned effectiveness of 1"

    rota.run(until=2002)
    vaccinated = rota.people.rotavaxprog.vaccinated
    assert np.sum(rota.people.rotavaxprog.waned_effectiveness[vaccinated]) < np.sum(vaccinated), "No vaccine waning detected after 1 year"


if __name__ == "__main__":
    make = False  # Set to True to regenerate results
    benchmark = False  # Set to True to redo the performance results
    test_default(make=make)
    test_alt(make=make)
    test_vx_intervention()
    test_vx_scheduling()
    test_vx_multivalent()
    test_vx_waning()