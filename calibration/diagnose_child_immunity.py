"""
Diagnostic: Check immunity values for children vs adults
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("DIAGNOSTIC: Checking immunity by age group")
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

# Mimic calibrate_uk.py initialization
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

print("\nImmunity values AFTER initialization:")
print(f"{'Age Group':<15} {'N':<8} {'baseline_imm':<15} {'rel_sus':<15}")
print("-" * 60)

for min_age, max_age, label in age_groups:
    mask = (sim.people.age.values >= min_age) & (sim.people.age.values < max_age)
    uids = np.where(mask)[0]

    if len(uids) > 0:
        baseline_imm = sim.connectors.rotaimmunityconnector.baseline_immunity[uids].mean()
        rel_sus = disease.rel_sus[uids].mean()
        print(f"{label:<15} {len(uids):<8} {baseline_imm:<15.3f} {rel_sus:<15.3f}")
    else:
        print(f"{label:<15} {0:<8} {'N/A':<15} {'N/A':<15}")

# Run for 1 timestep to see if step() changes things
print("\n" + "="*80)
print("After running 1 timestep:")
print("="*80)

sim.step()

print(f"\n{'Age Group':<15} {'N':<8} {'baseline_imm':<15} {'rel_sus':<15}")
print("-" * 60)

for min_age, max_age, label in age_groups:
    mask = (sim.people.age.values >= min_age) & (sim.people.age.values < max_age)
    uids = np.where(mask)[0]

    if len(uids) > 0:
        baseline_imm = sim.connectors.rotaimmunityconnector.baseline_immunity[uids].mean()
        rel_sus = disease.rel_sus[uids].mean()
        print(f"{label:<15} {len(uids):<8} {baseline_imm:<15.3f} {rel_sus:<15.3f}")
    else:
        print(f"{label:<15} {0:<8} {'N/A':<15} {'N/A':<15}")

print("\n" + "="*80)
print("ANALYSIS")
print("="*80)
print("\nExpected values:")
print("  Ages 0-3: baseline_immunity = 0.0, rel_sus = 1.0 (fully susceptible)")
print("  Ages 3-5: baseline_immunity = 0.5-0.95, rel_sus = 0.05-0.5")
print("  Ages 5+: baseline_immunity = 0.95, rel_sus = 0.05")
print("\nIf children <5 have high baseline_immunity or low rel_sus, that's the bug!")
print("="*80)
