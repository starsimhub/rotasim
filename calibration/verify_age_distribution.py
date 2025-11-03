"""
Standalone verification of age distribution in calibration results.
Does NOT import from calibrate_uk.py to avoid triggering calibration.
"""
import numpy as np
import starsim as ss
import rotasim as rs
import pandas as pd

# Best parameters from uk_calibration_corrected_aging.txt
best_params = {
    'reporting_rate': 0.0001539909745373953,
    'homotypic_immunity_efficacy': 0.16127362929023997,
    'partial_heterotypic_immunity_efficacy': 0.006089726965997355,
    'complete_heterotypic_immunity_efficacy': 0.12178083594800583,
    'base_beta': 0.060285715739135576,
    'maternal_immunity_efficacy': 0.0,
    'adult_baseline_immunity': 0.9660509093091926
}

print("="*70)
print("VERIFICATION: Age Distribution from Calibration")
print("="*70)
print(f"\nUsing best parameters:")
for k, v in best_params.items():
    print(f"  {k}: {v:.6f}")

# Create simulation
print("\n" + "="*70)
print("Step 1: Creating simulation")
print("="*70)

sim = rs.Sim(
    n_agents=10000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=True,
    scenario='single',
    base_beta=best_params['base_beta'],
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=rs.AgeAssortativeNet(n_contacts=7, assortativity=0.5),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Initialize the simulation (creates people object)
sim.init()
print(f"✓ Simulation initialized with {len(sim.people)} agents")

# Initialize UK age distribution (copied from calibrate_uk.py to avoid import issues)
print("\n" + "="*70)
print("Step 2: Initializing UK age distribution")
print("="*70)

target_age_dist = {
    '0-4': 0.0126,
    '5-9': 0.0127,
    '10-14': 0.0366,
    '15+': 0.9381
}

n_agents = len(sim.people)
age_groups = []
age_counts = []

for age_range, proportion in target_age_dist.items():
    count = int(n_agents * proportion)
    age_counts.append(count)
    age_groups.append(age_range)

# Adjust for rounding
total_assigned = sum(age_counts)
if total_assigned < n_agents:
    age_counts[-1] += (n_agents - total_assigned)

# Assign ages
current_idx = 0
for age_range, count in zip(age_groups, age_counts):
    if age_range == '0-4':
        ages = np.random.uniform(0, 5, count)
    elif age_range == '5-9':
        ages = np.random.uniform(5, 10, count)
    elif age_range == '10-14':
        ages = np.random.uniform(10, 15, count)
    else:  # 15+
        # UK population: ~20% are 15-30, ~30% are 30-50, ~30% are 50-70, ~20% are 70+
        ages = []
        for _ in range(count):
            rand = np.random.random()
            if rand < 0.2:
                ages.append(np.random.uniform(15, 30))
            elif rand < 0.5:
                ages.append(np.random.uniform(30, 50))
            elif rand < 0.8:
                ages.append(np.random.uniform(50, 70))
            else:
                ages.append(np.random.uniform(70, 90))
        ages = np.array(ages)

    sim.people.age.raw[current_idx:current_idx+count] = ages
    current_idx += count

ages = sim.people.age.raw
print(f"✓ Ages initialized")
print(f"  <1y:   {((ages < 1).sum() / len(ages)) * 100:.2f}%  ({(ages < 1).sum()} agents)")
print(f"  1-2y:  {(((ages >= 1) & (ages < 2)).sum() / len(ages)) * 100:.2f}%  ({((ages >= 1) & (ages < 2)).sum()} agents)")
print(f"  2-5y:  {(((ages >= 2) & (ages < 5)).sum() / len(ages)) * 100:.2f}%  ({((ages >= 2) & (ages < 5)).sum()} agents)")
print(f"  ≥5y:   {((ages >= 5).sum() / len(ages)) * 100:.2f}%  ({(ages >= 5).sum()} agents)")

# Initialize adult baseline immunity
print("\n" + "="*70)
print("Step 3: Initializing adult baseline immunity")
print("="*70)

adult_baseline_immunity = best_params['adult_baseline_immunity']
adult_mask = sim.people.age.raw >= 18

print(f"✓ Setting {adult_baseline_immunity:.1%} immunity for {adult_mask.sum()} adults (age ≥18)")

# Set baseline immunity for adults
for uid in np.where(adult_mask)[0]:
    # Set immunity dict with a very early recovery time (year 0) to indicate baseline immunity
    sim.people.rota.immunity[uid] = {'G1P8': 0.0}

# Seed initial infections
print("\n" + "="*70)
print("Step 4: Seeding initial infections")
print("="*70)

overall_prevalence = 0.002
n_infections = int(n_agents * overall_prevalence)

# Age-specific infection probabilities (higher for young children)
age_infection_prob = np.ones(n_agents)
age_infection_prob[sim.people.age.raw < 5] = 10.0  # 10x more likely for <5y
age_infection_prob[sim.people.age.raw >= 18] = 0.1  # 10x less likely for adults

age_infection_prob = age_infection_prob / age_infection_prob.sum()
infected_uids = np.random.choice(n_agents, size=n_infections, replace=False, p=age_infection_prob)

for uid in infected_uids:
    sim.people.rota.infected[uid] = True
    sim.people.rota.ti_infected[uid] = sim.ti

print(f"✓ Seeded {n_infections} initial infections")

# Run simulation
print("\n" + "="*70)
print("Step 5: Running simulation (2003-2013)")
print("="*70)

sim.run()

print("✓ Simulation complete")

# Analyze results
print("\n" + "="*70)
print("Step 6: Analyzing infection age distribution")
print("="*70)

# Get analyzer results
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

if analyzer is None:
    print("ERROR: Could not find InfectedStrainStats analyzer")
    exit(1)

df = analyzer.to_df()

# Focus on calibration period (years 5-9, which is 2008-2012)
calibration_df = df[(df['CollectionTime'] >= 5) & (df['CollectionTime'] < 10)].copy()

print(f"\nTotal infections in calibration period (2008-2012): {len(calibration_df)}")
print(f"Unique agents infected: {calibration_df['id'].nunique()}")

# Look at the raw Age field from analyzer
print("\n" + "-"*70)
print("Raw Age values from analyzer:")
print("-"*70)
age_value_counts = calibration_df['Age'].value_counts().sort_index()
for age_val, count in age_value_counts.items():
    pct = (count / len(calibration_df)) * 100
    print(f"  {age_val:8s}: {count:5d} infections ({pct:5.1f}%)")

# Map to age categories used in calibration
calibration_df['AgeCat'] = 'Unknown'
calibration_df.loc[calibration_df['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
calibration_df.loc[calibration_df['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
calibration_df.loc[calibration_df['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
calibration_df.loc[calibration_df['Age'] == '60+', 'AgeCat'] = '>=5 y'

print("\n" + "-"*70)
print("Infection age categories (all infections):")
print("-"*70)
age_cat_counts = calibration_df['AgeCat'].value_counts()
total = len(calibration_df)
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y', 'Unknown']:
    count = age_cat_counts.get(age_cat, 0)
    pct = (count / total) * 100
    print(f"  {age_cat:8s}: {count:5d} infections ({pct:5.1f}%)")

# Filter for symptomatic infections (first 3 per person)
calibration_df = calibration_df.sort_values(['id', 'CollectionTime'])
calibration_df['infection_number'] = calibration_df.groupby('id').cumcount() + 1
symptomatic = calibration_df[calibration_df['infection_number'] <= 3]

print("\n" + "-"*70)
print("Infection age categories (SYMPTOMATIC ONLY - first 3 per person):")
print("-"*70)
print(f"Total symptomatic infections: {len(symptomatic)}")

symp_age_counts = symptomatic['AgeCat'].value_counts()
symp_total = len(symptomatic) if len(symptomatic) > 0 else 1
for age_cat in ['<1 y', '1-2 y', '2-5 y', '>=5 y', 'Unknown']:
    count = symp_age_counts.get(age_cat, 0)
    pct = (count / symp_total) * 100
    print(f"  {age_cat:8s}: {count:5d} infections ({pct:5.1f}%)")

print("\n" + "-"*70)
print("TARGET age distribution (from calibration data):")
print("-"*70)
print("  <1 y    :          (13.8%)")
print("  1-2 y   :          (27.7%)")
print("  2-5 y   :          (46.9%)")
print("  >=5 y   :          (11.7%)")

print("\n" + "="*70)
print("VERIFICATION COMPLETE")
print("="*70)

# Summary
print("\nKEY FINDINGS:")
if symp_age_counts.get('<1 y', 0) / symp_total > 0.9:
    print("  ⚠ WARNING: >90% of symptomatic infections are in <1 year olds")
    print("  This confirms the calibration result - NOT a data processing artifact")
    print("  The model is genuinely concentrating infections in infants")
else:
    print("  ✓ Age distribution appears reasonable")
    print("  The 100% result in calibration may be a data processing issue")
