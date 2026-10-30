"""
Test to understand how incidence is being calculated
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import pandas as pd
import numpy as np

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*60)
print("Testing Incidence Calculation")
print("="*60)

# Small quick simulation
sim = rs.Sim(
    n_agents=2000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
    demographics=[
        ss.Births(birth_rate=ss.peryear(25)),
        ss.Deaths(death_rate=ss.peryear(10)),
    ],
)

print("\nRunning simulation...")
sim.run()

# Get infection data
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        print(f"\nTotal infection events: {len(df)}")

        # Filter to years 1-9
        initial8 = df[(df['CollectionTime'] < 9) & (df['CollectionTime'] > 1)]
        print(f"Infection events in years 1-9: {len(initial8)}")

        if len(initial8) > 0:
            print(f"\nUnique agents infected: {initial8['id'].nunique()}")
            print(f"Total infection events: {len(initial8)}")
            print(f"Avg infections per agent: {len(initial8) / initial8['id'].nunique():.2f}")

            # Check by year
            print("\nInfections by year:")
            for year in range(1, 10):
                year_data = initial8[(initial8['CollectionTime'] >= year) & (initial8['CollectionTime'] < year+1)]
                unique_agents = year_data['id'].nunique()
                total_events = len(year_data)
                print(f"  Year {year}: {unique_agents} unique agents, {total_events} total events")

            # Add age categories like process_incidence does
            initial8 = initial8.copy()
            initial8['AgeCat'] = np.nan
            initial8.loc[initial8['Age'].isin(['0-2', '2-4', '4-6', '6-12']), 'AgeCat'] = '<1 y'
            initial8.loc[initial8['Age'].isin(['12-24']), 'AgeCat'] = '1-2 y'
            initial8.loc[initial8['Age'].isin(['24-36', '36-48', '48-60']), 'AgeCat'] = '2-5 y'
            initial8.loc[initial8['Age'] == '60+', 'AgeCat'] = '>=5 y'

            # Calculate like process_incidence does
            print("\n" + "="*60)
            print("How process_incidence calculates:")
            print("="*60)

            # This is what line 87 does: groups by age and time, counts UNIQUE agents
            cases_summary = initial8.groupby(['AgeCat', 'CollectionTime']).agg(
                Cases_age=('id', 'nunique')
            ).reset_index()

            print("\nSample of cases_summary (first 10 rows):")
            print(cases_summary.head(10))

            # Problem: CollectionTime is continuous (e.g., 1.234, 1.567), not integer years
            # So each timestep gets its own row!
            print(f"\nTotal rows in cases_summary: {len(cases_summary)}")
            print(f"This means we're counting unique agents per TIMESTEP, not per YEAR")

            # What it should be doing: count unique agents per YEAR
            initial8['Year'] = np.floor(initial8['CollectionTime']).astype(int)
            cases_by_year = initial8.groupby(['AgeCat', 'Year']).agg(
                Cases_age=('id', 'nunique')
            ).reset_index()

            print("\n" + "="*60)
            print("If we count unique agents per YEAR instead:")
            print("="*60)
            print(cases_by_year)

            # Calculate average over years
            avg_by_age = cases_by_year.groupby('AgeCat').agg(avg_cases=('Cases_age', 'mean')).reset_index()
            print("\nAverage cases per year by age:")
            print(avg_by_age)

print("\n✓ Analysis complete!")
