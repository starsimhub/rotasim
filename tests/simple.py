import sciris as sc
import rotasim as rs

with sc.timer():
    rota = rs.Rota()
    sim = rs.Sim(connectors=rota)
    events = sim.run()
    print(events)