"""
Test if we can achieve endemic circulation by increasing birth rate significantly.

The issue: Disease dies out because immunity lasts longer than susceptible replenishment.
Solution: Increase birth rate to create faster flow of new susceptibles.

Current: 25/1000/year = 0.068 births/day in 5000 pop = 0.34 susceptibles/day
Needed: Much higher to sustain ~1-2 infections/day
"""
import rotasim as rs
import starsim as ss

print("="*60)
print("Testing Endemic Circulation with Increased Birth Rate")
print("="*60)

# Test different birth rates
birth_rates = [50, 100, 150, 200]

for birth_rate in birth_rates:
    print(f"\n{'='*60}")
    print(f"Testing birth_rate = {birth_rate} per 1000 per year")
    print(f"{'='*60}")

    sim = rs.Sim(
        n_agents=5000,
        start='2000-01-01',
        stop='2010-01-01',
        verbose=False,
        scenario='baseline',
        analyzers=[rs.InfectedStrainStats()],
        demographics=[
            ss.Births(birth_rate=ss.peryear(birth_rate)),
            ss.Deaths(death_rate=ss.peryear(10)),
        ],
    )

    print(f"Running simulation...")
    sim.run()

    # Check infections
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            df = analyzer.to_df()
            print(f"\nTotal infections: {len(df)}")

            if len(df) > 0:
                print(f"Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")

                # Check years 1-9 (calibration window)
                years_1_9 = df[(df['CollectionTime'] >= 1) & (df['CollectionTime'] < 9)]
                print(f"Infections in years 1-9: {len(years_1_9)}")

                if len(years_1_9) > 100:
                    print(f"✓ SUCCESS! Model sustains endemic circulation")
                    print(f"  Average: {len(years_1_9)/8:.1f} infections per year")

                    # Show yearly breakdown
                    print("\nInfections by year:")
                    for year in range(0, 10):
                        year_data = df[(df['CollectionTime'] >= year) & (df['CollectionTime'] < year+1)]
                        print(f"  Year {year}: {len(year_data)} infections")
                    break
                elif len(years_1_9) > 0:
                    print(f"⚠ Partial success - some circulation but may be too low")
                else:
                    print(f"✗ FAILED - Disease still dies out")
            else:
                print(f"✗ FAILED - No infections at all")

print("\n" + "="*60)
print("✓ Test complete")
print("="*60)
