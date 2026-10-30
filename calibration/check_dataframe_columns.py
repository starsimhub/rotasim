"""
Check what columns are in the InfectedStrainStats analyzer output
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import rotasim as rs
import starsim as ss

print("Creating quick simulation to check dataframe columns...")
sim = rs.Sim(
    n_agents=1000,
    start='2010-01-01',
    stop='2010-06-01',
    verbose=False,
    scenario='single',
    analyzers=[rs.InfectedStrainStats()],
)

sim.run()

# Get analyzer
analyzer = None
for a in sim.analyzers.values():
    if type(a).__name__ == 'InfectedStrainStats':
        analyzer = a
        break

df = analyzer.to_df()

print(f"\nDataFrame shape: {df.shape}")
print(f"\nColumn names:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print(f"\nFirst few rows:")
print(df.head())

print(f"\nData types:")
print(df.dtypes)
