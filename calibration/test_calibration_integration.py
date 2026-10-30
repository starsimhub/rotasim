"""
Quick test to verify the calibration integration works
"""
import rotasim as rs
import sciris as sc
import sys

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

# Create a minimal test sim
# process_incidence expects data from years 1-9, so run for 10 years
sim = rs.Sim(
    n_agents=1000,
    start='2000-01-01',
    stop='2010-01-01',  # 10 years for testing
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
)

print('Running sim...')
sim.run()
print('Sim completed')

# Test getting data from analyzer
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        print(f'\nGot dataframe with {len(df)} infection events')
        print(f'Columns: {list(df.columns)}')

        if len(df) > 0:
            print('\nFirst few rows:')
            print(df.head())

            # Test process_incidence.process_model
            print('\nTesting process_incidence.process_model()...')
            result_df = process_incidence.process_model(df)
            print(f'Processed dataframe shape: {result_df.shape}')
            print(f'Columns: {list(result_df.columns)}')
            print('\nProcessed results:')
            print(result_df)
        else:
            print('WARNING: No infection events recorded')
