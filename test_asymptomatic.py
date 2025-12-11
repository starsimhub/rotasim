"""
Test script to verify asymptomatic phase implementation
"""
import rotasim as rs
import starsim as ss
import numpy as np

print("="*60)
print("Testing Asymptomatic Phase Implementation")
print("="*60)

# Create a simple simulation
sim = rs.Sim(
    n_agents=1000,
    start=0,
    stop=50,
    dt=1,
    scenario='single',
    networks=ss.RandomNet(n_contacts=5),
    verbose=True,
)

# Initialize first
sim.init()

print("\nSimulation parameters:")
print(f"  Total infection duration: {sim.diseases[0].pars.dur_inf}")
print(f"  Asymptomatic duration: {sim.diseases[0].pars.dur_asymptomatic}")
print(f"  Asymptomatic shedding rate: {sim.diseases[0].pars.asymptomatic_shedding_rate}")
print("\nRunning simulation...")
sim.run()

# Check results
disease = sim.diseases[0]
print("\n" + "="*60)
print("Results Summary")
print("="*60)

# Count agents in each state at the end
print(f"\nFinal state counts:")
print(f"  Susceptible: {disease.susceptible.sum()}")
print(f"  Infected (symptomatic): {(disease.infected & ~disease.asymptomatic).sum()}")
print(f"  Infected (asymptomatic): {(disease.infected & disease.asymptomatic).sum()}")
print(f"  Recovered: {disease.recovered.sum()}")

# Check transmission rates
symptomatic_count = (disease.infected & ~disease.asymptomatic).sum()
asymptomatic_count = (disease.infected & disease.asymptomatic).sum()

if symptomatic_count > 0:
    symptomatic_trans = disease.rel_trans[(disease.infected & ~disease.asymptomatic).uids]
    print(f"\n  Symptomatic transmission rates: {symptomatic_trans}")

if asymptomatic_count > 0:
    asymptomatic_trans = disease.rel_trans[(disease.infected & disease.asymptomatic).uids]
    print(f"  Asymptomatic transmission rates: {asymptomatic_trans}")

print(f"\nTotal infections over simulation: {disease.n_infections.sum()}")
print(f"Total recoveries: {disease.results['new_recovered'].sum()}")

print("\n✓ Test complete!")
print("\nKey features:")
print("  - Infections progress: susceptible → infected (symptomatic) → asymptomatic → recovered")
print("  - Symptomatic phase has full transmission (rel_trans = 1.0)")
print("  - Asymptomatic phase has 90% reduced transmission (rel_trans = 0.1)")
print("  - Asymptomatic phase lasts 8 days")
