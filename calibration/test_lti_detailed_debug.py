"""
Detailed debug script to track LTI assignment step by step
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Detailed Long-Term Immunity Debug")
print("="*60)

# Very simple simulation - just 100 agents, 10 days
sim = rs.Sim(
    n_agents=100,
    start='2003-01-01',
    stop='2003-01-11',  # Just 10 days
    verbose=False,
    scenario='baseline',
    base_beta=0.40,
    override_prevalence=0.02,  # 2% = 2 agents
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning 10-day simulation with 100 agents...")
sim.run()

print("\n" + "="*60)
print("After 10 days:")
print("="*60)

# Get immunity connector
immunity = sim.get_connector_by_type("RotaImmunityConnector")

# Check basic stats
print(f"\nPopulation: {len(sim.people)} agents, {sim.people.alive.sum()} alive")
print(f"Long-term immune: {immunity.long_term_immune.sum()} agents")
print(f"Has any immunity: {immunity.has_immunity.sum()} agents")

# Check num_recovered_infections
recovered_counts = immunity.num_recovered_infections[sim.people.alive]
print(f"\nRecovered infection counts (alive agents):")
print(f"  0 infections: {(recovered_counts == 0).sum()}")
print(f"  1 infection:  {(recovered_counts == 1).sum()}")
print(f"  2 infections: {(recovered_counts == 2).sum()}")
print(f"  3+ infections: {(recovered_counts >= 3).sum()}")

# Check disease n_infections separately
print(f"\nDisease.n_infections counts:")
for disease in sim.diseases:
    if hasattr(disease, 'n_infections'):
        disease_infections = disease.n_infections[sim.people.alive]
        print(f"  {disease.name}: total={disease_infections.sum():.0f}, max={disease_infections.max():.0f}")

# Show which agents have LTI and their infection history
lti_agents = immunity.long_term_immune[sim.people.alive].nonzero()[0]
if len(lti_agents) > 0:
    print(f"\n" + "="*60)
    print(f"Long-term immune agents (showing first 10):")
    print("="*60)
    for i, uid in enumerate(lti_agents[:10]):
        recovered = immunity.num_recovered_infections[uid]
        print(f"  Agent {uid}: {recovered:.0f} recovered infections")

        # Check each disease
        for disease in sim.diseases:
            if hasattr(disease, 'n_infections'):
                n_inf = disease.n_infections[uid]
                if n_inf > 0:
                    print(f"    {disease.name}: {n_inf:.0f} infections")

# Check if there's a mismatch between disease.n_infections and immunity.num_recovered_infections
print(f"\n" + "="*60)
print("Comparing infection counters:")
print("="*60)

for uid in range(min(10, len(sim.people))):
    if sim.people.alive[uid]:
        recovered = immunity.num_recovered_infections[uid]
        total_disease_infections = sum(
            disease.n_infections[uid]
            for disease in sim.diseases
            if hasattr(disease, 'n_infections')
        )

        if recovered > 0 or total_disease_infections > 0:
            match = "✓" if recovered == total_disease_infections else "✗ MISMATCH"
            print(f"Agent {uid}: recovered={recovered:.0f}, disease.n_infections={total_disease_infections:.0f} {match}")
