"""
Test different parameter combinations to achieve endemic circulation.

Problem: 90-day immunity delay is too long - creates persistent immunity wave
Solution: Try shorter delays with balanced transmission to allow continuous circulation
"""
import rotasim as rs
import starsim as ss
import numpy as np

print("="*80)
print("Testing Parameter Combinations for Endemic Circulation")
print("="*80)

# Test combinations of immunity delay and transmission
test_configs = [
    {"immunity_delay": 30, "base_beta": 0.015, "init_prev": 0.002},
    {"immunity_delay": 30, "base_beta": 0.02, "init_prev": 0.002},
    {"immunity_delay": 14, "base_beta": 0.02, "init_prev": 0.002},
    {"immunity_delay": 14, "base_beta": 0.03, "init_prev": 0.001},
]

for config in test_configs:
    print(f"\n{'='*80}")
    print(f"Testing: immunity_delay={config['immunity_delay']} days, "
          f"base_beta={config['base_beta']}, init_prev={config['init_prev']}")
    print(f"{'='*80}")

    # Temporarily modify immunity connector
    from rotasim.immunity import RotaImmunityConnector

    # Create custom immunity connector with desired delay
    custom_immunity = RotaImmunityConnector(
        immunity_waning_delay=ss.days(config['immunity_delay'])
    )

    sim = rs.Sim(
        n_agents=5000,
        start='2000-01-01',
        stop='2010-01-01',
        verbose=False,
        scenario='baseline',
        base_beta=config['base_beta'],
        override_prevalence=config['init_prev'],
        analyzers=[rs.InfectedStrainStats()],
        connectors=[custom_immunity, rs.RotaReassortmentConnector()],
        demographics=[
            ss.Births(birth_rate=ss.peryear(50)),
            ss.Deaths(death_rate=ss.peryear(10)),
        ],
    )

    print(f"Running simulation...")
    sim.run()

    # Check infections
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            df = analyzer.to_df()

            if len(df) == 0:
                print(f"✗ FAILED - No infections at all")
                continue

            print(f"\nTotal infections: {len(df)}")
            print(f"Time range: {df['CollectionTime'].min():.2f} - {df['CollectionTime'].max():.2f} years")

            # Check each year
            yearly_infections = []
            for year in range(10):
                year_data = df[(df['CollectionTime'] >= year) & (df['CollectionTime'] < year+1)]
                yearly_infections.append(len(year_data))

            print("\nInfections by year:")
            for year, count in enumerate(yearly_infections):
                print(f"  Year {year}: {count}")

            # Check years 1-9 (calibration window, excluding burn-in year 0)
            years_1_9 = df[(df['CollectionTime'] >= 1) & (df['CollectionTime'] < 9)]
            avg_per_year = len(years_1_9) / 8

            # Check if sustained: need infections in multiple years, not just year 0
            years_with_infections = sum(1 for count in yearly_infections[1:] if count > 0)

            if years_with_infections >= 5 and avg_per_year > 50:
                print(f"\n✓ SUCCESS! Model sustains endemic circulation")
                print(f"  Years with infections (after year 0): {years_with_infections}/9")
                print(f"  Average in years 1-9: {avg_per_year:.1f} infections/year")
                print(f"\n  *** FOUND WORKING PARAMETERS ***")
                print(f"  immunity_delay: {config['immunity_delay']} days")
                print(f"  base_beta: {config['base_beta']}")
                print(f"  init_prev: {config['init_prev']}")
                break
            elif years_with_infections > 0:
                print(f"\n⚠ Partial success:")
                print(f"  Years with infections (after year 0): {years_with_infections}/9")
                print(f"  Average in years 1-9: {avg_per_year:.1f} infections/year")
                print(f"  (Need more sustained circulation)")
            else:
                print(f"\n✗ FAILED - Disease dies out after year 0")
                print(f"  Years with infections (after year 0): {years_with_infections}/9")

print("\n" + "="*80)
print("✓ Test complete")
print("="*80)
