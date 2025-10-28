"""
Debug script to understand why long-term immunity is preventing all infections
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Testing Long-Term Immunity Implementation")
print("="*60)

# Simple simulation with long-term immunity
sim = rs.Sim(
    n_agents=1000,
    start='2003-01-01',
    stop='2005-01-01',  # Just 2 years
    verbose=True,
    scenario='baseline',
    base_beta=0.40,
    override_prevalence=0.002,
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning simulation...")
sim.run()

print("\n" + "="*60)
print("Results:")
print("="*60)

# Check immunity connector
immunity = sim.get_connector_by_type("RotaImmunityConnector")
if immunity:
    print(f"\nLong-term immune agents: {immunity.long_term_immune.sum()}")
    print(f"Total agents: {len(sim.people)}")
    print(f"Alive agents: {sim.people.alive.sum()}")
    
    # Check how many people have long-term immunity
    lti_alive = immunity.long_term_immune[sim.people.alive].sum()
    print(f"Long-term immune (alive): {lti_alive}")
    
    # Check ages of LTI agents
    if hasattr(immunity, 'age_at_lti'):
        lti_ages = immunity.age_at_lti[immunity.long_term_immune & sim.people.alive]
        if len(lti_ages) > 0:
            print(f"Ages when LTI developed: mean={lti_ages.mean():.1f}, range=[{lti_ages.min():.1f}, {lti_ages.max():.1f}]")

# Check infections across all diseases
total_infections = 0
for disease in sim.diseases:
    if hasattr(disease, 'n_infections'):
        infections = disease.n_infections[sim.people.alive].sum()
        total_infections += infections
        print(f"\n{disease.name} infections (alive agents): {infections:.0f}")
        
print(f"\nTotal infections recorded: {total_infections:.0f}")

# Check if anyone is currently infected
currently_infected = sum(d.infected.sum() for d in sim.diseases if hasattr(d, 'infected'))
print(f"Currently infected: {currently_infected}")

# Check susceptibility
print("\n" + "="*60)
print("Susceptibility Analysis:")
print("="*60)

for disease in sim.diseases:
    if hasattr(disease, 'rel_sus'):
        sus = disease.rel_sus[sim.people.alive]
        print(f"\n{disease.name} relative susceptibility:")
        print(f"  Mean: {sus.mean():.4f}")
        print(f"  Min: {sus.min():.4f}")
        print(f"  Max: {sus.max():.4f}")
        print(f"  Zero susceptibility: {(sus == 0).sum()} agents")
        print(f"  Full susceptibility: {(sus == 1).sum()} agents")

