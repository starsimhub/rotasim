"""
Verify that baseline_immunity is immediately applied to rel_sus after initialize_immunity()
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("VERIFICATION: rel_sus is updated after initialize_immunity()")
print("="*80)

# Use typical calibration parameters
sim = rs.Sim(
    n_agents=5000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    base_beta=0.25,
    override_prevalence=0.002,
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            adult_baseline_immunity=0.95,
            adult_age_threshold=5.0,
            homotypic_immunity_efficacy=0.5,
        )
    ],
)

sim.init()

# Initialize immunity (mimics calibrate_uk.py)
sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=18, max_age=125, min_exposures=2, max_exposures=3)
sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=3, max_age=18, min_exposures=1, max_exposures=3)

disease = list(sim.diseases.values())[0]

# Check immunity by age group
age_groups = [
    (0, 1, "0-1 years"),
    (1, 2, "1-2 years"),
    (2, 3, "2-3 years"),
    (3, 5, "3-5 years"),
    (5, 18, "5-18 years"),
    (18, 125, "18+ years"),
]

print("\nImmunity values AFTER initialize_immunity() calls:")
print(f"{'Age Group':<15} {'N':<8} {'baseline_imm':<15} {'rel_sus':<15} {'Status'}")
print("-" * 70)

all_correct = True

for min_age, max_age, label in age_groups:
    mask = (sim.people.age.values >= min_age) & (sim.people.age.values < max_age)
    uids = np.where(mask)[0]

    if len(uids) > 0:
        baseline_imm = sim.connectors.rotaimmunityconnector.baseline_immunity[uids].mean()
        rel_sus = disease.rel_sus[uids].mean()

        # Check if rel_sus matches baseline_immunity
        expected_rel_sus = 1 - baseline_imm
        match = abs(rel_sus - expected_rel_sus) < 0.01

        status = "✓" if match else "✗ BUG!"
        if not match:
            all_correct = False
            status += f" (expected {expected_rel_sus:.3f})"

        print(f"{label:<15} {len(uids):<8} {baseline_imm:<15.3f} {rel_sus:<15.3f} {status}")
    else:
        print(f"{label:<15} {0:<8} {'N/A':<15} {'N/A':<15}")

print("\n" + "="*80)
print("VERIFICATION RESULT")
print("="*80)

if all_correct:
    print("\n✓✓✓ SUCCESS! rel_sus is correctly updated after initialize_immunity()")
    print("\nExpected behavior:")
    print("  - Ages 0-3: baseline_imm=0.0, rel_sus=1.0 (fully susceptible)")
    print("  - Ages 3-5: baseline_imm≈0.5-0.95, rel_sus≈0.05-0.5 (partially protected)")
    print("  - Ages 5+: baseline_imm≈0.73-0.95, rel_sus≈0.05-0.27 (well protected)")
    print("\n✓ Immunity is now properly applied BEFORE the simulation starts!")
    print("✓ This should fix the '100% of cases in people over 5' bug!")
else:
    print("\n✗✗✗ FAILURE! rel_sus does NOT match baseline_immunity!")
    print("\nThe fix did not work. baseline_immunity is set but rel_sus is not updated.")

print("="*80)
