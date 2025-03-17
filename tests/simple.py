import sciris as sc
import rotasim as rs

with sc.timer():
    sim = rs.RotaSim()
    events = sim.run()
    print(events)