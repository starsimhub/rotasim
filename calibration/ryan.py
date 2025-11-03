import starsim as ss
import rotasim as rs
import numpy as np

# Create a simple simulation
sim = rs.Sim(
    n_agents=1000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=True,
    scenario='single',
    base_beta=0.05,  # Low beta to minimize infections for cleaner test
    override_prevalence=0.0,  # Start with no infections
    analyzers=[],  # No analyzers needed
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

sim.init()
print(f"Initial age (uid==0): {sim.people.age[0]}")
sim.run()
print(f"End age (uid==0): {sim.people.age[0]}")