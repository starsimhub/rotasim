"""
Quick test to see if the model produces any infections with the new parameters
"""
import rotasim as rs
import starsim as ss

print("Testing if model produces infections with 180-day immunity delay...")

sim = rs.Sim(
    n_agents=5000,
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

print(f"\nRunning simulation...")
sim.run()

# Check infections
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        print(f"\nTotal infections: {len(df)}")

        if len(df) > 0:
            print(f"Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")

            # Check by year
            print("\nInfections by year:")
            for year in range(0, 11):
                year_data = df[(df['CollectionTime'] >= year) & (df['CollectionTime'] < year+1)]
                print(f"  Year {year}: {len(year_data)} infections")

            # Check years 1-9 specifically
            years_1_9 = df[(df['CollectionTime'] > 1) & (df['CollectionTime'] < 9)]
            print(f"\nInfections in years 1-9: {len(years_1_9)}")

            if len(years_1_9) == 0:
                print("⚠ No infections in calibration window (years 1-9)!")
                print("   Model is dying out too quickly.")
        else:
            print("⚠ NO INFECTIONS AT ALL!")
            print("   Model died out immediately.")

print("\n✓ Test complete")
