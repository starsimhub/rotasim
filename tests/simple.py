import sciris as sc
import rotasim as rs

with sc.timer():
    rota = rs.Rota()
    sim = rs.Sim(connectors=rota)
    sim.run()

    events = sim.connectors['rota'].event_dict
    print(events)