"""
Step-by-step debug to understand LTI assignment issue
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Step-by-Step Long-Term Immunity Debug")
print("="*60)

# Tiny simulation - 20 agents, just 3 days
sim = rs.Sim(
    n_agents=20,
    start='2003-01-01',
    stop='2003-01-04',  # Just 3 days
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

print("\nRunning 3-day simulation with 20 agents...")
print("Initial prevalence: 10% (2 agents)")
sim.run()

print("\n" + "="*60)
print("After 3 days:")
print("="*60)

# Get immunity connector
immunity = sim.get_connector_by_type("RotaImmunityConnector")

# Count states
lti_count = immunity.long_term_immune.sum()
has_immunity_count = immunity.has_immunity.sum()

print(f"Long-term immune: {lti_count} agents")
print(f"Has any immunity: {has_immunity_count} agents")
print()

# Check each disease
total_infected = 0
total_recovered = 0
for disease in sim.diseases.values():
    if hasattr(disease, 'infected'):
        infected = disease.infected.sum()
        recovered = disease.recovered.sum()
        n_infections_total = disease.n_infections[sim.people.alive].sum()
        total_infected += infected
        total_recovered += recovered
        if infected > 0 or recovered > 0 or n_infections_total > 0:
            print(f"{disease.name}: {infected} infected, {recovered} recovered, n_infections={n_infections_total:.0f}")

print()

# Check num_recovered_infections
recovered_counts = immunity.num_recovered_infections[sim.people.alive]
print(f"num_recovered_infections: min={recovered_counts.min():.0f}, max={recovered_counts.max():.0f}, sum={recovered_counts.sum():.0f}")
print()

# Check susceptibility for each disease
for disease in sim.diseases.values():
    if hasattr(disease, 'rel_sus'):
        rel_sus = disease.rel_sus[sim.people.alive]
        print(f"{disease.name} rel_sus: mean={rel_sus.mean():.3f}, min={rel_sus.min():.3f}, max={rel_sus.max():.3f}")
        zero_sus = (rel_sus == 0.0).sum()
        full_sus = (rel_sus == 1.0).sum()
        print(f"  Zero susceptibility: {zero_sus} agents")
        print(f"  Full susceptibility: {full_sus} agents")

print("="*60)
print("Final Analysis:")
print("="*60)

# Check for mismatches
immunity = sim.get_connector_by_type("RotaImmunityConnector")
for uid in range(min(20, len(sim.people))):
    if sim.people.alive[uid]:
        recovered = immunity.num_recovered_infections[uid]
        total_disease_infections = sum(
            disease.n_infections[uid]
            for disease in sim.diseases.values()
            if hasattr(disease, 'n_infections')
        )

        if recovered > 0 or total_disease_infections > 0:
            match = "✓" if recovered == total_disease_infections else "✗ MISMATCH"
            print(f"Agent {uid}: recovered={recovered:.0f}, disease.n_infections={total_disease_infections:.0f} {match}")
