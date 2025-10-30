"""
Test if init_post is being called
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Testing init_post Execution")
print("="*60)

# Very simple simulation - just 20 agents, 15 days (enough for recovery)
sim = rs.Sim(
    n_agents=20,
    start='2003-01-01',
    stop='2003-01-16',  # 15 days - enough for infections to recover
    verbose=False,
    scenario='baseline',
    base_beta=0.40,
    override_prevalence=0.10,  # 10% = 2 agents initially infected
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning simulation...")
sim.run()

print("\n" + "="*60)
print("After sim.run():")
print("="*60)

# Get immunity connector
immunity = sim.get_connector_by_type("RotaImmunityConnector")

# Check disease n_infections AFTER run
print(f"DEBUG: Type of sim.diseases: {type(sim.diseases)}")
print(f"DEBUG: sim.diseases content: {sim.diseases}")
print(f"DEBUG: sim.diseases.values(): {list(sim.diseases.values())}")

total_disease_infections = 0
for disease in sim.diseases.values():
    print(f"DEBUG: Checking disease {disease.name}, has n_infections: {hasattr(disease, 'n_infections')}")
    if hasattr(disease, 'n_infections'):
        # Only sum for ALIVE agents
        alive_n_inf = disease.n_infections[sim.people.alive].sum()
        all_n_inf = disease.n_infections.sum()
        total_disease_infections += alive_n_inf
        print(f"{disease.name}: n_infections (alive) = {alive_n_inf:.0f}, n_infections (all) = {all_n_inf:.0f}")

# Check immunity connector
recovered_counts_alive = immunity.num_recovered_infections[sim.people.alive].sum()
recovered_counts_all = immunity.num_recovered_infections.sum()
print(f"\nImmunity connector:")
print(f"  num_recovered_infections (alive) = {recovered_counts_alive:.0f}")
print(f"  num_recovered_infections (all) = {recovered_counts_all:.0f}")

# Check for mismatch
if total_disease_infections == 0 and recovered_counts_all > 0:
    print("\n" + "="*60)
    print("PROBLEM FOUND:")
    print("="*60)
    print("Disease n_infections = 0 but immunity num_recovered_infections > 0")
    print("This means record_recovery is being called but n_infections never incremented")
    print("\nPossible causes:")
    print("1. init_post is not being called")
    print("2. init_post is called but initially_infected.uids is empty")
    print("3. n_infections increment in init_post is not working")
