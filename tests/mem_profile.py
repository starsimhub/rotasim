import sciris as sc
import rotasim as rs

def main():
    with sc.timer():
        sim = rs.Sim(n_agents=5e3, verbose=True)
        sim.run()

sc.mprofile(main)
