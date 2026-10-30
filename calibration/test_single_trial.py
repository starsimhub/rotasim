"""
Test a single calibration trial to verify the setup works
"""
import sciris as sc
import rotasim as rs

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("Creating base sim...")
sim = rs.Sim(
    n_agents=1000,
    start='2000-01-01',
    stop='2010-01-01',  # 10 years
    verbose=False,
    scenario='baseline',
    analyzers=[rs.InfectedStrainStats()],
)

print("Loading calibration data...")
data = process_incidence.process_data()
print(f"Calibration data shape: {data.shape}")
print(data)

print("\nRunning test simulation...")
sim.run()

print("\nExtracting results...")
# Test getting data from analyzer
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        print(f'Got dataframe with {len(df)} infection events')

        if len(df) > 0:
            print(f'Time range: {df["CollectionTime"].min():.2f} - {df["CollectionTime"].max():.2f} years')
            print(f'Strains: {df["Strain"].unique()}')

            # Test process_incidence.process_model
            print('\nProcessing with process_incidence.process_model()...')
            result_df = process_incidence.process_model(df, verbose=True)
            print(f'Processed dataframe shape: {result_df.shape}')
            if len(result_df) > 0:
                print(result_df)
            else:
                print("WARNING: Processed dataframe is empty!")
        else:
            print('WARNING: No infection events recorded')

print("\n✓ Test completed successfully!")
