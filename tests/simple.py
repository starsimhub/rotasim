import sciris as sc
import rotasim as rs

with sc.timer():
    sim = rs.Sim(
        verbose=True,
        to_csv=False,
        n_agents=50000,
        timelimit=10,
        rota_kwargs={"vaccination_time": 5, "time_to_equilibrium": 2},
    )
    sim.run()

    events = sim.connectors["rota"].event_dict
    print(events)
