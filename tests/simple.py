import sciris as sc
import rotasim as rs

with sc.timer():
    sim = rs.Sim()
    events = sim.run()
    print(events)