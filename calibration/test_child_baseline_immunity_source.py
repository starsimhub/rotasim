"""
Diagnostic test to identify when and how children acquire baseline_immunity
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("DIAGNOSTIC: TRACKING WHEN CHILDREN GET BASELINE IMMUNITY")
print("="*80)

adult_immunity_value = 0.95
adult_age_threshold = 5.0

print(f"\nSetup:")
print(f"  adult_baseline_immunity = {adult_immunity_value}")
print(f"  adult_age_threshold = {adult_age_threshold} years")

# Create simulation
sim = rs.Sim(
    n_agents=5000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    base_beta=0.25,
    override_prevalence=0.005,
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            adult_baseline_immunity=adult_immunity_value,
            adult_age_threshold=adult_age_threshold,
        )
    ],
)

sim.init()

# Checkpoint 1: After init, before any initialize_immunity calls
print(f"\n{'='*80}")
print("CHECKPOINT 1: After sim.init() (before initialize_immunity calls)")
print("="*80)

adults_mask = sim.people.age.values >= adult_age_threshold
adult_uids = np.where(adults_mask)[0]
children_mask = sim.people.age.values < adult_age_threshold
child_uids = np.where(children_mask)[0]

print(f"\nPopulation:")
print(f"  Adults (≥{adult_age_threshold} y): {len(adult_uids)}")
print(f"  Children (<{adult_age_threshold} y): {len(child_uids)}")

baseline_adults_1 = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids]
baseline_children_1 = sim.connectors.rotaimmunityconnector.baseline_immunity[child_uids]

print(f"\nBaseline immunity:")
print(f"  Adults with baseline_immunity > 0: {(baseline_adults_1 > 0).sum()} / {len(adult_uids)}")
print(f"  Mean adult baseline_immunity: {baseline_adults_1.mean():.3f}")
print(f"  Children with baseline_immunity > 0: {(baseline_children_1 > 0).sum()} / {len(child_uids)}")
print(f"  Mean child baseline_immunity: {baseline_children_1.mean():.3f}")

# Show age distribution of children
print(f"\nChildren age distribution:")
age_3_4 = ((sim.people.age.values >= 3) & (sim.people.age.values < adult_age_threshold)).sum()
age_0_3 = (sim.people.age.values < 3).sum()
print(f"  Age 0-3 years: {age_0_3}")
print(f"  Age 3-{adult_age_threshold} years: {age_3_4}")

# Checkpoint 2: After first initialize_immunity call (18-125 years)
print(f"\n{'='*80}")
print("CHECKPOINT 2: After initialize_immunity(min_age=18, max_age=125, ...)")
print("="*80)

sim.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=18,
    max_age=125,
    min_exposures=2,
    max_exposures=3
)

baseline_adults_2 = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids]
baseline_children_2 = sim.connectors.rotaimmunityconnector.baseline_immunity[child_uids]

print(f"\nBaseline immunity:")
print(f"  Adults with baseline_immunity > 0: {(baseline_adults_2 > 0).sum()} / {len(adult_uids)}")
print(f"  Mean adult baseline_immunity: {baseline_adults_2.mean():.3f}")
print(f"  Children with baseline_immunity > 0: {(baseline_children_2 > 0).sum()} / {len(child_uids)}")
print(f"  Mean child baseline_immunity: {baseline_children_2.mean():.3f}")

if (baseline_children_2 > 0).sum() > (baseline_children_1 > 0).sum():
    print(f"\n  ⚠ WARNING: {(baseline_children_2 > 0).sum() - (baseline_children_1 > 0).sum()} additional children gained baseline_immunity!")
else:
    print(f"\n  ✓ No children gained baseline_immunity from this call")

# Checkpoint 3: After second initialize_immunity call (3-18 years) - THIS IS THE SUSPECTED BUG
print(f"\n{'='*80}")
print("CHECKPOINT 3: After initialize_immunity(min_age=3, max_age=18, ...)")
print("="*80)
print("THIS IS THE SUSPECTED BUG SOURCE!\n")

sim.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=3,
    max_age=18,
    min_exposures=1,
    max_exposures=3
)

baseline_adults_3 = sim.connectors.rotaimmunityconnector.baseline_immunity[adult_uids]
baseline_children_3 = sim.connectors.rotaimmunityconnector.baseline_immunity[child_uids]

print(f"\nBaseline immunity:")
print(f"  Adults with baseline_immunity > 0: {(baseline_adults_3 > 0).sum()} / {len(adult_uids)}")
print(f"  Mean adult baseline_immunity: {baseline_adults_3.mean():.3f}")
print(f"  Children with baseline_immunity > 0: {(baseline_children_3 > 0).sum()} / {len(child_uids)}")
print(f"  Mean child baseline_immunity: {baseline_children_3.mean():.3f}")

children_gained = (baseline_children_3 > 0).sum() - (baseline_children_2 > 0).sum()
if children_gained > 0:
    print(f"\n  ⚠ BUG CONFIRMED: {children_gained} children gained baseline_immunity from this call!")

    # Identify which children
    children_with_immunity = child_uids[baseline_children_3 > 0]
    ages_with_immunity = sim.people.age.values[children_with_immunity]

    print(f"\n  Age breakdown of children with baseline_immunity:")
    print(f"    Age 0-3 years: {(ages_with_immunity < 3).sum()}")
    print(f"    Age 3-{adult_age_threshold} years: {(ages_with_immunity >= 3).sum()}")
    print(f"\n  This confirms the bug: initialize_immunity(min_age=3, ...) is giving")
    print(f"  baseline_immunity to children aged 3-{adult_age_threshold}, even though")
    print(f"  they are below the adult_age_threshold!")
else:
    print(f"\n  ✓ No children gained baseline_immunity from this call")

# Final analysis
print(f"\n{'='*80}")
print("ROOT CAUSE ANALYSIS")
print("="*80)

print(f"""
The bug is in calibrate_uk.py lines 283-285:

  Line 283: initialize_immunity(min_age=18, max_age=125, min_exposures=2, max_exposures=3)
  Line 284: initialize_immunity(min_age=3, max_age=18, min_exposures=1, max_exposures=3)
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            THIS IS THE PROBLEM!

The second call initializes immunity for ages 3-18 years.
This includes children aged 3-{adult_age_threshold} years, who are below adult_age_threshold={adult_age_threshold}.

In immunity.py line 463, initialize_immunity() sets:
    self.baseline_immunity[eligible_uids] = self.pars.adult_baseline_immunity

So any agent aged 3-18 gets baseline_immunity = {adult_immunity_value}, including children!

SOLUTION:
Change line 284 in calibrate_uk.py from:
    initialize_immunity(min_age=3, max_age=18, ...)
To:
    initialize_immunity(min_age={adult_age_threshold}, max_age=18, ...)

This ensures only agents ≥{adult_age_threshold} years get baseline_immunity,
consistent with the adult_age_threshold parameter.
""")

print("="*80)
