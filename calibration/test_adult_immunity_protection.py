"""
Test to verify adults with baseline_immunity are actually protected from infection
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("TESTING ADULT BASELINE IMMUNITY PROTECTION")
print("="*80)

# Create simulation with high adult baseline immunity
adult_immunity_value = 0.95  # 95% protection for adults

print(f"\n1. Creating simulation with adult_baseline_immunity = {adult_immunity_value:.2f}...")

sim = rs.Sim(
    n_agents=5000,
    start='2010-01-01',
    stop='2012-01-01',  # 2 years
    verbose=False,
    scenario='single',
    base_beta=0.25,  # Moderate transmission
    override_prevalence=0.005,  # 0.5% initial prevalence
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            adult_baseline_immunity=adult_immunity_value,
            adult_age_threshold=5.0,
            homotypic_immunity_efficacy=0.5,
            partial_heterotypic_immunity_efficacy=0.3,
            complete_heterotypic_immunity_efficacy=0.2,
        )
    ],
)

sim.init()

# Identify adults BEFORE running simulation
adult_age_threshold = 5.0
adults_mask = sim.people.age.values >= adult_age_threshold
adult_uids = np.where(adults_mask)[0]

children_mask = sim.people.age.values < adult_age_threshold
child_uids = np.where(children_mask)[0]

print(f"   Total agents: {len(sim.people)}")
print(f"   Adults (≥{adult_age_threshold} years): {len(adult_uids)}")
print(f"   Children (<{adult_age_threshold} years): {len(child_uids)}")

# Check baseline_immunity values
baseline_immunity_adults = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids]
baseline_immunity_children = sim.connectors.rotaimmunityconnector.baseline_immunity[child_uids]

print(f"\n2. Checking baseline_immunity values...")
print(f"   Adults with baseline_immunity > 0: {(baseline_immunity_adults > 0).sum()}")
print(f"   Mean adult baseline_immunity: {baseline_immunity_adults.mean():.3f}")
print(f"   Children with baseline_immunity > 0: {(baseline_immunity_children > 0).sum()}")
print(f"   Mean child baseline_immunity: {baseline_immunity_children.mean():.3f}")

# Check initial rel_sus for adults vs children
# Get the first (and only) disease
disease = list(sim.diseases.values())[0]
initial_rel_sus_adults = disease.rel_sus[adult_uids].mean()
initial_rel_sus_children = disease.rel_sus[child_uids].mean()

print(f"\n3. Checking initial relative susceptibility (rel_sus)...")
print(f"   Mean adult rel_sus: {initial_rel_sus_adults:.3f} (expected: {1-adult_immunity_value:.3f})")
print(f"   Mean child rel_sus: {initial_rel_sus_children:.3f} (expected: 1.0)")

# Store initial UIDs for tracking
adult_uids_initial = set(adult_uids)
child_uids_initial = set(child_uids)

print(f"\n4. Running simulation (2 years)...")
sim.run()
print("   ✓ Complete")

# Get infection data
print(f"\n5. Analyzing infections...")
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()
total_infections = len(df)
unique_infected = df['id'].nunique()

print(f"   Total infections: {total_infections}")
print(f"   Unique agents infected: {unique_infected}")

# Categorize infections by initial age group
adult_infections = df[df['id'].isin(adult_uids_initial)]
child_infections = df[df['id'].isin(child_uids_initial)]

unique_adults_infected = adult_infections['id'].nunique()
unique_children_infected = child_infections['id'].nunique()

print(f"\n6. Infection breakdown by initial age group:")
print(f"   Adults infected: {unique_adults_infected} / {len(adult_uids_initial)} ({unique_adults_infected/len(adult_uids_initial)*100:.2f}%)")
print(f"   Children infected: {unique_children_infected} / {len(child_uids_initial)} ({unique_children_infected/len(child_uids_initial)*100:.2f}%)")

# Calculate infection rates per 100 person-years
years = 2.0
adult_rate = (unique_adults_infected / len(adult_uids_initial)) / years * 100
child_rate = (unique_children_infected / len(child_uids_initial)) / years * 100

print(f"\n7. Infection rates (per 100 person-years):")
print(f"   Adult rate: {adult_rate:.2f} per 100 person-years")
print(f"   Child rate: {child_rate:.2f} per 100 person-years")
print(f"   Ratio (child/adult): {child_rate/adult_rate if adult_rate > 0 else 'inf'}x")

# Check final rel_sus for a sample of adults and children
final_adults_alive = [uid for uid in adult_uids_initial if uid < len(sim.people)][:100]
final_children_alive = [uid for uid in child_uids_initial if uid < len(sim.people)][:100]

if len(final_adults_alive) > 0:
    final_rel_sus_adults = disease.rel_sus[final_adults_alive].mean()
    print(f"\n8. Final relative susceptibility (sample):")
    print(f"   Mean adult rel_sus: {final_rel_sus_adults:.3f}")
else:
    print(f"\n8. No adults available to check final rel_sus")

# Expected protection
expected_protection = adult_immunity_value
expected_reduction = 1 / (1 - expected_protection)  # e.g., 0.95 immunity → 20x reduction

print(f"\n" + "="*80)
print("RESULTS SUMMARY")
print("="*80)

# Verify protection is working
actual_reduction = child_rate / adult_rate if adult_rate > 0 else float('inf')

print(f"\nExpected protection: {expected_protection*100:.0f}%")
print(f"Expected infection rate reduction: {expected_reduction:.1f}x")
print(f"Actual infection rate reduction: {actual_reduction:.1f}x")

# Success criteria
if actual_reduction >= expected_reduction * 0.5:  # Allow 50% tolerance
    print(f"\n✓ SUCCESS: Adult baseline immunity is providing protection!")
    print(f"  - Adults are {actual_reduction:.1f}x less likely to be infected than children")
    print(f"  - This is consistent with {expected_protection*100:.0f}% baseline immunity")
elif adult_rate == 0:
    print(f"\n✓ EXCELLENT: No adults were infected!")
    print(f"  - Complete protection from baseline immunity")
elif actual_reduction >= 2.0:
    print(f"\n✓ PARTIAL SUCCESS: Adults are {actual_reduction:.1f}x less likely to be infected")
    print(f"  - Protection is working, but less than expected {expected_reduction:.1f}x")
    print(f"  - This may be due to stochastic variation or other factors")
else:
    print(f"\n✗ WARNING: Adult protection may not be working properly")
    print(f"  - Expected {expected_reduction:.1f}x reduction, got {actual_reduction:.1f}x")
    print(f"  - Adult infection rate: {adult_rate:.2f} per 100 person-years")
    print(f"  - Child infection rate: {child_rate:.2f} per 100 person-years")

    # Diagnose why protection isn't working
    print(f"\n  Diagnostic checks:")
    print(f"    - Initial adult rel_sus: {initial_rel_sus_adults:.3f} (expected ~{1-adult_immunity_value:.3f})")
    print(f"    - Final adult rel_sus: {final_rel_sus_adults:.3f}" if len(final_adults_alive) > 0 else "    - Final rel_sus: N/A")
    print(f"    - Adult baseline_immunity values: mean={baseline_immunity_adults.mean():.3f}")

print("="*80)
