import sciris as sc
import rotasim as rs

with sc.timer():
    sim = rs.Sim(verbose=True)
    sim.run()

    events = sim.connectors['rota'].event_dict
    print(events)