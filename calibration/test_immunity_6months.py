"""
Test adult immunity over 6 months to check final rel_sus values
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("TESTING ADULT IMMUNITY OVER 6 MONTHS")
print("="*80)

adult_immunity_value = 0.95
adult_age_threshold = 5.0

print(f"\n1. Creating simulation with adult_baseline_immunity = {adult_immunity_value:.2f}...")
print(f"   Running for 6 months (2010-01-01 to 2010-07-01)")

sim = rs.Sim(
    n_agents=5000,
    start='2010-01-01',
    stop='2010-07-01',  # 6 months
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

# Identify initial adults and children
adults_mask_init = sim.people.age.values >= adult_age_threshold
adult_uids_init = np.where(adults_mask_init)[0]
children_mask_init = sim.people.age.values < adult_age_threshold
child_uids_init = np.where(children_mask_init)[0]

print(f"   Initial population:")
print(f"     Adults (≥{adult_age_threshold} years): {len(adult_uids_init)}")
print(f"     Children (<{adult_age_threshold} years): {len(child_uids_init)}")

# Check initial state
baseline_immunity_adults_init = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids_init]
rel_sus_adults_init = disease.rel_sus[adult_uids_init].mean()
rel_sus_children_init = disease.rel_sus[child_uids_init].mean()

print(f"\n2. Initial state (immediately after init):")
print(f"   Adult baseline_immunity: {baseline_immunity_adults_init.mean():.3f}")
print(f"   Adult rel_sus: {rel_sus_adults_init:.3f}")
print(f"   Child rel_sus: {rel_sus_children_init:.3f}")

# Run simulation
print(f"\n3. Running simulation...")
sim.run()
print("   ✓ Complete")

# Check final state
print(f"\n4. Final state (after 6 months):")

# Re-identify adults and children at end
adults_mask_final = sim.people.age.values >= adult_age_threshold
adult_uids_final = np.where(adults_mask_final)[0]
children_mask_final = sim.people.age.values < adult_age_threshold
child_uids_final = np.where(children_mask_final)[0]

print(f"   Final population:")
print(f"     Adults alive: {len(adult_uids_final)}")
print(f"     Children alive: {len(child_uids_final)}")

# Get final baseline immunity and rel_sus
baseline_immunity_adults_final = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids_final]
baseline_immunity_children_final = sim.connectors.rotaimmunityconnector.baseline_immunity[child_uids_final]

rel_sus_adults_final = disease.rel_sus[adult_uids_final].mean()
rel_sus_children_final = disease.rel_sus[child_uids_final].mean()

print(f"\n   Adult statistics:")
print(f"     Mean baseline_immunity: {baseline_immunity_adults_final.mean():.3f}")
print(f"     Adults with baseline_immunity > 0: {(baseline_immunity_adults_final > 0).sum()}")
print(f"     Mean rel_sus: {rel_sus_adults_final:.3f}")
print(f"     Expected rel_sus: {1-adult_immunity_value:.3f}")

print(f"\n   Child statistics:")
print(f"     Mean baseline_immunity: {baseline_immunity_children_final.mean():.3f}")
print(f"     Children with baseline_immunity > 0: {(baseline_immunity_children_final > 0).sum()}")
print(f"     Mean rel_sus: {rel_sus_children_final:.3f}")

# Get infection data
print(f"\n5. Infection statistics over 6 months:")
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

# Categorize by initial age group
adult_uids_init_set = set(adult_uids_init)
child_uids_init_set = set(child_uids_init)

adult_infections = df[df['id'].isin(adult_uids_init_set)]
child_infections = df[df['id'].isin(child_uids_init_set)]

unique_adults_infected = adult_infections['id'].nunique()
unique_children_infected = child_infections['id'].nunique()

print(f"\n   By initial age group:")
print(f"     Adults infected: {unique_adults_infected} / {len(adult_uids_init)} ({unique_adults_infected/len(adult_uids_init)*100:.2f}%)")
print(f"     Children infected: {unique_children_infected} / {len(child_uids_init)} ({unique_children_infected/len(child_uids_init)*100:.2f}%)")

# Calculate infection rates (annualized)
months = 6.0
adult_rate_annual = (unique_adults_infected / len(adult_uids_init)) * (12/months) * 100
child_rate_annual = (unique_children_infected / len(child_uids_init)) * (12/months) * 100

print(f"\n   Annualized infection rates:")
print(f"     Adult rate: {adult_rate_annual:.2f} per 100 person-years")
print(f"     Child rate: {child_rate_annual:.2f} per 100 person-years")
if adult_rate_annual > 0:
    ratio = child_rate_annual / adult_rate_annual
    print(f"     Ratio (child/adult): {ratio:.1f}x")

# Analysis
print("\n" + "="*80)
print("ANALYSIS")
print("="*80)

expected_reduction = 1 / (1 - adult_immunity_value)  # 0.95 → 20x

print(f"\nExpected protection: {adult_immunity_value*100:.0f}%")
print(f"Expected infection rate reduction: {expected_reduction:.1f}x")

if adult_rate_annual > 0:
    actual_reduction = child_rate_annual / adult_rate_annual
    print(f"Actual infection rate reduction: {actual_reduction:.1f}x")

    if actual_reduction >= expected_reduction * 0.5:
        print(f"\n✓ Protection is working reasonably well!")
    else:
        print(f"\n⚠ Protection is weaker than expected")
else:
    print(f"Actual infection rate reduction: infinite (no adult infections!)")
    print(f"\n✓ Perfect protection!")

print(f"\n{'='*80}")
print("FINAL REL_SUS VALUES")
print("="*80)
print(f"\nAdults:")
print(f"  Mean rel_sus: {rel_sus_adults_final:.4f}")
print(f"  Expected:     {1-adult_immunity_value:.4f}")
print(f"  Difference:   {abs(rel_sus_adults_final - (1-adult_immunity_value)):.4f}")

print(f"\nChildren:")
print(f"  Mean rel_sus: {rel_sus_children_final:.4f}")

# Check if rel_sus is reasonable
if rel_sus_adults_final < 0.15:  # Within 3x of expected 0.05
    print(f"\n✓ Adult rel_sus is close to expected value")
    print(f"  This suggests the immunity system is working properly")
else:
    print(f"\n⚠ Adult rel_sus is higher than expected")
    print(f"  This suggests possible issues with immunity application")

print("="*80)
