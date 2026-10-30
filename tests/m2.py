import sciris as sc
import starsim as ss

def main():
    sim = ss.Sim(n_agents=20e3, diseases='sis', networks='random')
    sim.run()

sc.mprofile(main)
