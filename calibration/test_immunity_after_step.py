"""
Test to check if adult immunity works correctly after step() is called
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("TESTING ADULT IMMUNITY AFTER STEP() CALLS")
print("="*80)

adult_immunity_value = 0.95
adult_age_threshold = 5.0

print(f"\n1. Creating simulation with adult_baseline_immunity = {adult_immunity_value:.2f}...")

sim = rs.Sim(
    n_agents=5000,
    start='2010-01-01',
    stop='2010-02-01',  # 1 month only
    verbose=False,
    scenario='single',
    base_beta=0.25,
    override_prevalence=0.005,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            adult_baseline_immunity=adult_immunity_value,
            adult_age_threshold=adult_age_threshold,
            homotypic_immunity_efficacy=0.5,
            partial_heterotypic_immunity_efficacy=0.3,
            complete_heterotypic_immunity_efficacy=0.2,
        )
    ],
)

sim.init()

# Get disease reference
disease = list(sim.diseases.values())[0]

# Identify adults and children
adults_mask = sim.people.age.values >= adult_age_threshold
adult_uids = np.where(adults_mask)[0]
children_mask = sim.people.age.values < adult_age_threshold
child_uids = np.where(children_mask)[0]

print(f"   Total agents: {len(sim.people)}")
print(f"   Adults (≥{adult_age_threshold} years): {len(adult_uids)}")
print(f"   Children (<{adult_age_threshold} years): {len(child_uids)}")

# Check BEFORE any steps
print(f"\n2. Checking state IMMEDIATELY after init() (before any steps)...")
baseline_immunity_adults = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids]
print(f"   Adults with baseline_immunity > 0: {(baseline_immunity_adults > 0).sum()}")
print(f"   Mean adult baseline_immunity: {baseline_immunity_adults.mean():.3f}")

rel_sus_adults_before = disease.rel_sus[adult_uids].mean()
rel_sus_children_before = disease.rel_sus[child_uids].mean()
print(f"   Mean adult rel_sus: {rel_sus_adults_before:.3f} (expected: {1-adult_immunity_value:.3f})")
print(f"   Mean child rel_sus: {rel_sus_children_before:.3f} (expected: 1.0)")

# Now run for 1 month
print(f"\n3. Running simulation for 1 month...")
sim.run()
print("   ✓ Complete")

# Check AFTER running for 1 month
print(f"\n4. Checking state AFTER running for 1 month...")

# Re-identify adults (some may have aged or died)
adults_mask_after = sim.people.age.values >= adult_age_threshold
adult_uids_after = np.where(adults_mask_after)[0]
children_mask_after = sim.people.age.values < adult_age_threshold
child_uids_after = np.where(children_mask_after)[0]

print(f"   Adults alive: {len(adult_uids_after)}")
print(f"   Children alive: {len(child_uids_after)}")

baseline_immunity_adults_after = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids_after]
print(f"   Adults with baseline_immunity > 0: {(baseline_immunity_adults_after > 0).sum()}")
print(f"   Mean adult baseline_immunity: {baseline_immunity_adults_after.mean():.3f}")

rel_sus_adults_after = disease.rel_sus[adult_uids_after].mean()
rel_sus_children_after = disease.rel_sus[child_uids_after].mean()
print(f"   Mean adult rel_sus: {rel_sus_adults_after:.3f} (expected: {1-adult_immunity_value:.3f})")
print(f"   Mean child rel_sus: {rel_sus_children_after:.3f} (expected: 1.0)")

# Get infection data
print(f"\n5. Checking infections during the 1-month period...")
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

# Categorize infections by age group (use initial classification)
adult_uids_initial_set = set(adult_uids)
child_uids_initial_set = set(child_uids)

adult_infections = df[df['id'].isin(adult_uids_initial_set)]
child_infections = df[df['id'].isin(child_uids_initial_set)]

unique_adults_infected = adult_infections['id'].nunique()
unique_children_infected = child_infections['id'].nunique()

print(f"\n   Infections by initial age group:")
print(f"   Adults infected: {unique_adults_infected} / {len(adult_uids)} ({unique_adults_infected/len(adult_uids)*100:.2f}%)")
print(f"   Children infected: {unique_children_infected} / {len(child_uids)} ({unique_children_infected/len(child_uids)*100:.2f}%)")

# Calculate infection rates (per month, then annualize)
months = 1.0
adult_rate_monthly = (unique_adults_infected / len(adult_uids)) * 100
child_rate_monthly = (unique_children_infected / len(child_uids)) * 100
adult_rate_annual = adult_rate_monthly * 12
child_rate_annual = child_rate_monthly * 12

print(f"\n6. Infection rates:")
print(f"   Adult rate: {adult_rate_annual:.2f} per 100 person-years")
print(f"   Child rate: {child_rate_annual:.2f} per 100 person-years")
if adult_rate_annual > 0:
    ratio = child_rate_annual / adult_rate_annual
    print(f"   Ratio (child/adult): {ratio:.1f}x")
else:
    print(f"   Ratio (child/adult): infinite (no adult infections!)")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

# Expected protection
expected_protection = adult_immunity_value
expected_reduction = 1 / (1 - expected_protection)  # 0.95 → 20x

print(f"\nExpected protection: {expected_protection*100:.0f}%")
print(f"Expected infection rate reduction: {expected_reduction:.1f}x")

if adult_rate_annual > 0:
    actual_reduction = child_rate_annual / adult_rate_annual
    print(f"Actual infection rate reduction: {actual_reduction:.1f}x")
else:
    actual_reduction = float('inf')
    print(f"Actual infection rate reduction: infinite (perfect protection!)")

# Diagnosis
print(f"\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if rel_sus_adults_after < 0.1:  # Close to expected 0.05
    print(f"\n✓ rel_sus is correctly set after simulation starts!")
    print(f"  - Adult rel_sus after 1 month: {rel_sus_adults_after:.3f}")
    print(f"  - Expected value: {1-adult_immunity_value:.3f}")
    print(f"  - Protection is being applied correctly during simulation")

    if actual_reduction >= expected_reduction * 0.5:  # Within 50% of expected
        print(f"\n✓ Immunity is working as expected!")
        print(f"  - Adults are {actual_reduction:.1f}x less likely to be infected")
        print(f"  - This matches expected {expected_reduction:.1f}x reduction")
    elif adult_rate_annual == 0:
        print(f"\n✓ Perfect protection observed!")
        print(f"  - No adult infections in 1 month")
        print(f"  - This suggests very strong protection")
    else:
        print(f"\n⚠ Protection is weaker than expected")
        print(f"  - Expected {expected_reduction:.1f}x reduction")
        print(f"  - Got {actual_reduction:.1f}x reduction")
        print(f"  - This may be due to stochastic variation or other factors")

elif rel_sus_adults_before > 0.9 and rel_sus_adults_after > 0.9:
    print(f"\n✗ PROBLEM: rel_sus is NOT being updated!")
    print(f"  - Adult rel_sus before: {rel_sus_adults_before:.3f}")
    print(f"  - Adult rel_sus after: {rel_sus_adults_after:.3f}")
    print(f"  - Expected: {1-adult_immunity_value:.3f}")
    print(f"  - The step() function is not updating rel_sus correctly")

elif rel_sus_adults_before > 0.9 and rel_sus_adults_after < 0.1:
    print(f"\n✓ GOOD NEWS: rel_sus gets updated during run()!")
    print(f"  - Adult rel_sus before run: {rel_sus_adults_before:.3f}")
    print(f"  - Adult rel_sus after run: {rel_sus_adults_after:.3f}")
    print(f"  - Expected: {1-adult_immunity_value:.3f}")
    print(f"  - step() is correctly updating rel_sus")
    print(f"\n  However, initial infections may occur before protection kicks in...")

print("\n" + "="*80)
