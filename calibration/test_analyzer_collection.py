"""
Test if InfectedStrainStats analyzer is collecting infection events during calibration-like runs

This test mimics the exact calibration setup to diagnose why all trials return identical GOF.
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

print("="*60)
print("Testing InfectedStrainStats Collection During Calibration-Like Run")
print("="*60)

# Use exact same setup as calibration
sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',  # 5 year burn-in before 2008
    stop='2013-01-01',   # End in 2012
    verbose=True,
    scenario='baseline',
    base_beta=0.40,
    override_prevalence=0.002,
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

print("\nRunning 10-year simulation (5-year burn-in + 5-year follow-up)...")
print("Parameters:")
print(f"  n_agents: 5000")
print(f"  base_beta: 0.40")
print(f"  init_prev: 0.002")
print(f"  burn-in: 2003-2007")
print(f"  follow-up: 2008-2012")
print()

sim.run()

print("\n" + "="*60)
print("Simulation Complete - Checking Analyzer Results")
print("="*60)

# Get the analyzer (using same approach as calibration.py line 316-318)
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

if analyzer is None:
    print("ERROR: InfectedStrainStats analyzer not found!")
    print(f"Available analyzers: {[type(a).__name__ for a in sim.analyzers.values()]}")
else:
    print(f"\nAnalyzer found: {analyzer}")

    # Get data using the to_df() method (same as calibration.py line 325)
    df = analyzer.to_df()

    if df is None:
        print("\nERROR: analyzer.to_df() returned None!")
    else:
        print(f"\nInfected strain stats DataFrame shape: {df.shape}")
        print(f"Total infection events collected: {len(df)}")

        if len(df) > 0:
            print("\nFirst 10 infection events:")
            print(df.head(10))

            print("\nInfections by year:")
            # Convert CollectionTime to years since start
            df['Year'] = df['CollectionTime'] / 365.0 + 2003
            yearly_counts = df.groupby(df['Year'].astype(int)).size()
            print(yearly_counts)

            print("\nInfections in follow-up period (2008-2012):")
            follow_up = df[(df['Year'] >= 2008) & (df['Year'] < 2013)]
            print(f"Total infections in follow-up: {len(follow_up)}")

            if len(follow_up) > 0:
                print("\nStrain distribution in follow-up:")
                strain_counts = follow_up['Strain'].value_counts()
                print(strain_counts)
            else:
                print("WARNING: Zero infections during follow-up period!")
        else:
            print("\nWARNING: DataFrame exists but is EMPTY!")

# Also check disease-level infection counts
print("\n" + "="*60)
print("Disease-Level Infection Counts")
print("="*60)

total_infections = 0
for disease in sim.diseases.values():
    if hasattr(disease, 'n_infections'):
        n_inf = disease.n_infections[sim.people.alive].sum()
        total_infections += n_inf
        print(f"{disease.name}: {n_inf:.0f} total infections")

print(f"\nTotal infections across all strains: {total_infections:.0f}")

# Check new_infections results
print("\n" + "="*60)
print("Checking new_infections Results Arrays")
print("="*60)

for disease in sim.diseases.values():
    if hasattr(disease, 'results') and 'new_infections' in disease.results:
        new_inf = disease.results['new_infections']
        total_new = new_inf.sum()
        print(f"{disease.name}: {total_new:.0f} new infections recorded in results")
        if total_new > 0:
            # Show when infections occurred
            nonzero_days = np.where(new_inf > 0)[0]
            print(f"  Infections on {len(nonzero_days)} different days")
            print(f"  First infection: day {nonzero_days[0] if len(nonzero_days) > 0 else 'N/A'}")
            print(f"  Last infection: day {nonzero_days[-1] if len(nonzero_days) > 0 else 'N/A'}")

print("\n" + "="*60)
print("Summary")
print("="*60)

if analyzer:
    df = analyzer.to_df()
    if df is not None:
        print(f"Analyzer collected: {len(df)} infection events")
        print(f"Disease n_infections total: {total_infections:.0f}")

        if len(df) == 0 and total_infections > 0:
            print("\n⚠ PROBLEM: Infections occurring but analyzer not collecting them!")
        elif len(df) > 0:
            follow_up = df[(df['CollectionTime'] / 365.0 + 2003 >= 2008) &
                           (df['CollectionTime'] / 365.0 + 2003 < 2013)]
            if len(follow_up) == 0:
                print("\n⚠ PROBLEM: Analyzer collecting infections but NONE during follow-up period!")
            else:
                print(f"\n✓ Analyzer working: {len(follow_up)} infections in follow-up period")
    else:
        print("\n⚠ PROBLEM: analyzer.to_df() returned None!")
else:
    print("\n⚠ PROBLEM: Analyzer not found!")
