"""
Test calibration setup using parameter values from tests/simple.py that work
"""
import rotasim as rs
import starsim as ss

print("="*60)
print("Testing Endemic Circulation with simple.py Parameters")
print("="*60)
print("\nParameters from tests/simple.py:")
print("  base_beta: 0.16")
print("  override_prevalence: 0.002")
print("  n_contacts: 7")
print("  birth_rate: 70 per 1000/year")
print("  death_rate: 20 per 1000/year")
print("="*60)

sim = rs.Sim(
    n_agents=5000,
    start='2000-01-01',
    stop='2010-01-01',
    verbose=False,
    scenario='baseline',
    base_beta=0.16,  # From simple.py
    override_prevalence=0.002,  # From simple.py
    analyzers=[rs.InfectedStrainStats()],
    networks=ss.RandomNet(n_contacts=7),  # From simple.py
    demographics=[
        ss.Births(birth_rate=ss.peryear(70)),  # From simple.py
        ss.Deaths(death_rate=ss.peryear(20)),  # From simple.py
    ],
)

print(f"\nRunning simulation...")
sim.run()

# Check infections
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()
        print(f"\n✓ Total infections: {len(df)}")

        if len(df) > 0:
            print(f"✓ Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")

            # Check yearly breakdown
            print("\nInfections by year:")
            yearly_infections = []
            for year in range(10):
                year_data = df[(df['CollectionTime'] >= year) & (df['CollectionTime'] < year+1)]
                yearly_infections.append(len(year_data))
                print(f"  Year {year}: {len(year_data)}")

            # Check years 1-9 (calibration window)
            years_1_9 = df[(df['CollectionTime'] >= 1) & (df['CollectionTime'] < 9)]
            avg_per_year = len(years_1_9) / 8
            years_with_infections = sum(1 for count in yearly_infections[1:] if count > 0)

            print(f"\n✓ Infections in years 1-9: {len(years_1_9)}")
            print(f"✓ Average per year: {avg_per_year:.1f}")
            print(f"✓ Years with infections (after year 0): {years_with_infections}/9")

            if years_with_infections >= 5:
                print(f"\n✓✓✓ SUCCESS! Model sustains endemic circulation ✓✓✓")
            elif years_with_infections > 0:
                print(f"\n⚠ Partial success - some circulation but may not be sustained")
            else:
                print(f"\n✗ FAILED - Disease dies out after year 0")
        else:
            print(f"\n✗ FAILED - No infections at all")

print("\n" + "="*60)
print("✓ Test complete")
print("="*60)
