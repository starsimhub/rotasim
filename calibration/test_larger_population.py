"""
Test with larger population (10,000 agents) and low rel_beta
"""
import sciris as sc
import rotasim as rs
import starsim as ss

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*80)
print("Testing Larger Population with Low rel_beta")
print("="*80)

# Test different rel_beta values with larger population
# Previous tests showed 0.01 and below cause die-out
# Let's test higher values that might sustain endemic circulation
rel_beta_values = [0.1, 0.05, 0.03, 0.02, 0.015]

for rel_beta in rel_beta_values:
    print(f"\n{'='*80}")
    print(f"Testing rel_beta = {rel_beta}")
    print(f"{'='*80}")

    # Create sim with 10,000 agents
    sim = rs.Sim(
        n_agents=10000,
        start='2000-01-01',
        stop='2010-01-01',
        verbose=False,
        scenario='baseline',
        base_beta=0.16,
        override_prevalence=0.002,
        analyzers=[rs.InfectedStrainStats()],
        networks=ss.RandomNet(n_contacts=7),
        demographics=[
            ss.Births(birth_rate=ss.peryear(70)),
            ss.Deaths(death_rate=ss.peryear(20)),
        ],
    )

    # Initialize and apply rel_beta
    sim.init()
    for disease in sim.diseases.values():
        if hasattr(disease, 'pars') and hasattr(disease.pars, 'beta'):
            disease.pars.beta = disease.pars.beta * rel_beta

    print(f"\nRunning simulation with n_agents=10000...")
    sim.run()

    # Check results
    for analyzer in sim.analyzers.values():
        if type(analyzer).__name__ == 'InfectedStrainStats':
            df = analyzer.to_df()

            print(f"\n✓ Total infections: {len(df)}")

            if len(df) > 0:
                # Check infections per year
                df['Year'] = (df['CollectionTime']).astype(int)
                infections_by_year = df.groupby('Year').size()

                print(f"\nInfections by year:")
                for year in range(10):
                    count = infections_by_year.get(year, 0)
                    print(f"  Year {year}: {count}")

                # Check years 1-9
                years_1_9 = df[(df['CollectionTime'] >= 1) & (df['CollectionTime'] < 9)]
                if len(years_1_9) > 100:
                    print(f"\n✓ SUCCESS: {len(years_1_9)} infections in years 1-9 (sustained circulation)")

                    # Process through incidence calculator
                    incidence_df = process_incidence.process_model(dat=df, popsize=10000, verbose=False)

                    if len(incidence_df) > 0:
                        avg_incidence = incidence_df['inci'].mean()
                        print(f"\n✓ Average incidence: {avg_incidence:.1f} per 100,000 per year")

                        # Compare to target
                        data = process_incidence.process_data()
                        target_avg = data['inci'].mean()
                        print(f"✓ Target: {target_avg:.1f} per 100,000 per year")

                        ratio = avg_incidence / target_avg
                        print(f"✓ Ratio: {ratio:.2f}x")

                        if ratio < 2.0:
                            print(f"\n✓✓✓ EXCELLENT! Within 2x of target!")
                            print(f"*** FOUND GOOD PARAMETERS: rel_beta={rel_beta}, n_agents=10000 ***")
                            break
                        elif ratio < 10.0:
                            print(f"\n✓✓ GOOD! Within 10x of target")
                    else:
                        print(f"\n⚠ process_model returned empty dataframe")
                else:
                    print(f"\n⚠ Only {len(years_1_9)} infections in years 1-9 - may not be sustained")
            else:
                print(f"\n✗ NO INFECTIONS - disease died out!")

print("\n" + "="*80)
print("✓ Test complete")
print("="*80)
