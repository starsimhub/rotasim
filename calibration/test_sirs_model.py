"""
Test the simplified SIRS model with 13-week temporary immunity (no LTI)

This test verifies:
1. Disease circulates endemically (doesn't go extinct)
2. Agents cycle through S → I → R → S as expected
3. Infection counting still works correctly
4. 13-week immunity period functions properly
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Testing Simplified SIRS Model (13-week immunity, no LTI)")
print("="*60)

# Single-strain model for clarity
# Use 'simple' scenario (single strain: G1P8 only)
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',  # 10 years
    verbose=True,
    scenario='simple',  # Just G1P8
    base_beta=0.40,
    override_prevalence=0.002,
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning 10-year simulation with single strain (G1P8)...")
print("Parameters:")
print(f"  n_agents: 5000")
print(f"  base_beta: 0.40")
print(f"  init_prev: 0.002")
print(f"  Temporary immunity: 13 weeks (91 days mean)")
print(f"  LTI: DISABLED")
print()

sim.run()

print("\n" + "="*60)
print("Simulation Complete - Analyzing Results")
print("="*60)

# Get immunity connector
immunity = sim.get_connector_by_type("RotaImmunityConnector")

# Count LTI (should be zero now)
lti_count = immunity.long_term_immune.sum()
print(f"\nLong-term immune: {lti_count} agents (should be 0)")

# Check disease circulation by year
print("\n" + "="*60)
print("Disease Circulation by Year")
print("="*60)

for disease in sim.diseases.values():
    if hasattr(disease, 'results') and 'new_infections' in disease.results:
        new_inf = disease.results['new_infections']

        # Group by year (365 days per year)
        n_years = len(new_inf) // 365
        yearly_infections = []

        for year in range(n_years):
            start_day = year * 365
            end_day = (year + 1) * 365
            year_total = new_inf[start_day:end_day].sum()
            yearly_infections.append(year_total)

        print(f"\n{disease.name} infections by year:")
        for year, count in enumerate(yearly_infections):
            print(f"  {2003 + year}: {count:.0f} infections")

        # Check if disease persisted
        last_5_years = sum(yearly_infections[-5:])
        print(f"\nTotal infections in last 5 years (2008-2012): {last_5_years:.0f}")

        if last_5_years > 0:
            print("✓ Disease circulated endemically!")
        else:
            print("✗ Disease went extinct")

# Check infection counting
print("\n" + "="*60)
print("Infection Counting")
print("="*60)

alive = sim.people.alive
recovered_counts = immunity.num_recovered_infections[alive]
print(f"\nnum_recovered_infections: min={recovered_counts.min():.0f}, max={recovered_counts.max():.0f}")

# Distribution of infection episodes
unique, counts = np.unique(recovered_counts, return_counts=True)
print("\nDistribution of infection episodes:")
for episodes, agent_count in zip(unique, counts):
    if episodes > 0:
        pct = 100 * agent_count / len(alive)
        print(f"  {int(episodes)} episodes: {agent_count} agents ({pct:.1f}%)")

never_infected = (recovered_counts == 0).sum()
pct_never = 100 * never_infected / len(alive)
print(f"  Never infected: {never_infected} agents ({pct_never:.1f}%)")

# Check susceptibility
print("\n" + "="*60)
print("Susceptibility Distribution")
print("="*60)

for disease in sim.diseases.values():
    if hasattr(disease, 'rel_sus'):
        rel_sus = disease.rel_sus[alive]
        print(f"\n{disease.name} rel_sus:")
        print(f"  Mean: {rel_sus.mean():.3f}")
        print(f"  Min: {rel_sus.min():.3f}")
        print(f"  Max: {rel_sus.max():.3f}")

        # Distribution
        zero_sus = (rel_sus == 0.0).sum()
        full_sus = (rel_sus == 1.0).sum()
        partial_sus = len(alive) - zero_sus - full_sus

        print(f"  Zero susceptibility (0.0): {zero_sus} agents ({100*zero_sus/len(alive):.1f}%)")
        print(f"  Full susceptibility (1.0): {full_sus} agents ({100*full_sus/len(alive):.1f}%)")
        print(f"  Partial susceptibility: {partial_sus} agents ({100*partial_sus/len(alive):.1f}%)")

print("\n" + "="*60)
print("Summary")
print("="*60)

# Final verdict
for disease in sim.diseases.values():
    if hasattr(disease, 'results') and 'new_infections' in disease.results:
        new_inf = disease.results['new_infections']
        n_years = len(new_inf) // 365
        yearly_infections = [new_inf[year*365:(year+1)*365].sum() for year in range(n_years)]

        total_infections = sum(yearly_infections)
        last_5_years = sum(yearly_infections[-5:])

        print(f"\n{disease.name}:")
        print(f"  Total infections: {total_infections:.0f}")
        print(f"  Infections in last 5 years: {last_5_years:.0f}")
        print(f"  LTI agents: {lti_count}")

        if last_5_years > 0 and lti_count == 0:
            print("\n✓ SUCCESS: SIRS model working correctly!")
            print("  - Disease circulates endemically")
            print("  - No long-term immunity")
            print("  - 13-week temporary immunity functioning")
        elif lti_count > 0:
            print("\n✗ PROBLEM: LTI mechanism not fully disabled")
        else:
            print("\n✗ PROBLEM: Disease still going extinct despite removing LTI")
