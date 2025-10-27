"""
Diagnostic script to understand why cases are concentrated in adults
despite correct population age distribution
"""
import sys
sys.path.insert(0, '.')

import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np
import pandas as pd

print("="*70)
print("Diagnosing Age-Specific Transmission")
print("="*70)

# Create UK simulation with best-fit parameters from calibration
sim = rs.Sim(
    n_agents=5000,
    start='2008-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='baseline',
    base_beta=0.16,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Apply best-fit parameters
best_pars = {
    'reporting_rate': 0.000105,
    'homotypic_immunity_efficacy': 0.369,
    'partial_heterotypic_immunity_efficacy': 0.041,
    'complete_heterotypic_immunity_efficacy': 0.276,
    'rel_beta': 2.668,
    'maternal_immunity_efficacy': 0.0
}

print("\nApplying best-fit parameters:")
for par, val in best_pars.items():
    print(f"  {par}: {val:.6f}")

# Update immunity parameters
for conn_name, conn in sim.connectors.items():
    if 'Immunity' in conn_name:
        if hasattr(conn, 'pars'):
            for par, val in best_pars.items():
                if par in conn.pars:
                    conn.pars[par] = val

print("\nRunning simulation...")
sim.run()

print("\n" + "="*70)
print("Population Demographics at End")
print("="*70)

ages = sim.people.age.values
alive = sim.people.alive
ages_alive = ages[alive]

age_bins = [(0, 1), (1, 2), (2, 5), (5, 125)]
age_labels = ['<1 y', '1-2 y', '2-5 y', '>=5 y']

print(f"\n{'Age':<10} {'Population':<12} {'%':<10}")
print("-"*32)
for (low, high), label in zip(age_bins, age_labels):
    count = np.sum((ages_alive >= low) & (ages_alive < high))
    prop = count / len(ages_alive) * 100
    print(f"{label:<10} {count:<12} {prop:>6.2f}%")

print("\n" + "="*70)
print("Checking rel_beta Implementation")
print("="*70)

# Check if rel_beta is being applied
print(f"\nBase beta: {sim.pars.base_beta}")
print(f"Best-fit rel_beta: {best_pars['rel_beta']}")

# Try to find where rel_beta is used
print("\nSearching for rel_beta in connectors...")
for conn_name, conn in sim.connectors.items():
    if hasattr(conn, 'pars'):
        if 'rel_beta' in conn.pars:
            print(f"  {conn_name}: rel_beta = {conn.pars.rel_beta}")

print("\n" + "="*70)
print("Infection Events by Age")
print("="*70)

# Load infection data if available
try:
    # Look for the most recent CSV file
    import glob
    csv_files = glob.glob('../results/rota_strains_infected_all_*.csv')
    if csv_files:
        latest_file = max(csv_files, key=lambda x: x.split('_')[-1])
        print(f"\nLoading infection data from: {latest_file}")

        dat = pd.read_csv(latest_file)

        # Filter to simulation period (years 1-6 for 5 years)
        dat_period = dat[(dat['CollectionTime'] >= 1) & (dat['CollectionTime'] < 6)]

        print(f"\nTotal infections in period: {len(dat_period)}")

        # Map age categories
        dat_period['AgeCat'] = 'Other'
        dat_period.loc[dat_period['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
        dat_period.loc[dat_period['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
        dat_period.loc[dat_period['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
        dat_period.loc[dat_period['Age'] == '60+', 'AgeCat'] = '>=5 y'

        # Count infections by age
        age_counts = dat_period['AgeCat'].value_counts()
        total = len(dat_period)

        print(f"\n{'Age':<10} {'Infections':<12} {'%':<10}")
        print("-"*32)
        for label in age_labels:
            count = age_counts.get(label, 0)
            prop = count / total * 100 if total > 0 else 0
            print(f"{label:<10} {count:<12} {prop:>6.2f}%")

        # Count first infections only
        dat_period_sorted = dat_period.sort_values(['id', 'CollectionTime'])
        dat_period_sorted['infection_number'] = dat_period_sorted.groupby('id').cumcount() + 1

        first_infections = dat_period_sorted[dat_period_sorted['infection_number'] == 1]

        print(f"\n\nFirst Infections Only (n={len(first_infections)}):")
        print(f"{'Age':<10} {'First Infections':<17} {'%':<10}")
        print("-"*37)

        age_counts_first = first_infections['AgeCat'].value_counts()
        total_first = len(first_infections)

        for label in age_labels:
            count = age_counts_first.get(label, 0)
            prop = count / total_first * 100 if total_first > 0 else 0
            print(f"{label:<10} {count:<17} {prop:>6.2f}%")

        # Count infections by infection number
        print("\n" + "="*70)
        print("Infections by Infection Number")
        print("="*70)

        inf_num_counts = dat_period_sorted['infection_number'].value_counts().sort_index()
        print(f"\n{'Infection #':<15} {'Count':<10} {'%':<10}")
        print("-"*35)
        for inf_num in sorted(inf_num_counts.index[:10]):  # First 10
            count = inf_num_counts[inf_num]
            prop = count / len(dat_period_sorted) * 100
            print(f"{inf_num:<15} {count:<10} {prop:>6.2f}%")

        if len(inf_num_counts) > 10:
            print(f"{'11+':<15} {inf_num_counts[11:].sum():<10} {inf_num_counts[11:].sum()/len(dat_period_sorted)*100:>6.2f}%")

    else:
        print("\nNo CSV infection files found in ../results/")

except Exception as e:
    print(f"\nError loading infection data: {e}")

print("\n" + "="*70)
print("Summary")
print("="*70)
print("\nKEY QUESTION: Why are 97% of REPORTED cases in adults when:")
print("  1. Population is 94.6% adults (correct)")
print("  2. Infections should be higher in young children (via rel_beta)")
print("  3. Target shows 88% of cases in children <5 years")
print("\nPOSSIBLE ISSUES:")
print("  1. rel_beta may not be properly affecting transmission rates")
print("  2. Immunity buildup may be too slow (children get reinfected)")
print("  3. Reporting threshold (first 3 infections) may be incorrect")
print("  4. base_beta may be too high, swamping rel_beta effect")
