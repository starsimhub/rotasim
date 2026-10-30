"""
Quick test to check if transmission occurs throughout simulation
"""
import sciris as sc
import starsim as ss
import rotasim as rs
import numpy as np

# Create analyzer
analyzer = rs.InfectedStrainStats(use_infection_based_severity=True)

# Create sim
thisdir = sc.thispath(__file__)
people = ss.People(n_agents=10000, age_data=thisdir / 'uk_age_data.csv')
sim = rs.Sim(
    n_agents=10000,
    start='2003-01-01',
    stop='2013-01-01',
    verbose=False,
    scenario='single',
    people=people,
    analyzers=[analyzer],
    networks=ss.RandomNet(n_contacts=7),
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

# Init and run
sim.init()

print(f'After init: disease beta = {list(sim.diseases.values())[0].pars.beta}')

# Initialize adult immunity
sim.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=18, max_age=125, min_exposures=5, max_exposures=15
)

print('Running simulation...')
sim.run()

# Check infections
import pandas as pd
print(f'\nAnalyzer infection_events keys: {analyzer.infection_events.keys()}')
print(f'Number of ids: {len(analyzer.infection_events.get("id", []))}')
print(f'Number of strains: {len(analyzer.infection_events.get("Strain", []))}')

df = pd.DataFrame(analyzer.infection_events)  # Access raw data directly
print(f'\nTotal infections in dataframe: {len(df)}')
if len(df) > 0:
    print(f'Time range: {df["CollectionTime"].min():.2f} - {df["CollectionTime"].max():.2f} years')
    print(f'\nInfections by year:')
    df['year'] = np.floor(df['CollectionTime']).astype(int)
    year_counts = df.groupby('year').size()
    print(year_counts)
    print(f'\nYears 5-10 (calibration period): {year_counts[year_counts.index.isin(range(5,10))].sum()} infections')
else:
    print('NO INFECTIONS!')
