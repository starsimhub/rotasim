"""
Test a single trial with low rel_beta to see if disease dies out
"""
import sciris as sc
import rotasim as rs
import starsim as ss

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*80)
print("Testing Single Trial with rel_beta=0.002")
print("="*80)

# Create sim
sim = rs.Sim(
    n_agents=5000,
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

rel_beta = 0.002
print(f"\nApplying rel_beta = {rel_beta}")
for disease in sim.diseases.values():
    if hasattr(disease, 'pars') and hasattr(disease.pars, 'beta'):
        # Just multiply - no need to print details
        disease.pars.beta = disease.pars.beta * rel_beta
print(f"  Applied rel_beta={rel_beta} to all diseases")

print(f"\nRunning simulation...")
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

            # Process through incidence calculator
            incidence_df = process_incidence.process_model(dat=df, popsize=5000, verbose=False)

            if len(incidence_df) > 0:
                avg_incidence = incidence_df['inci'].mean()
                print(f"\n✓ Average incidence: {avg_incidence:.1f} per 100,000 per year")

                # Compare to target
                data = process_incidence.process_data()
                target_avg = data['inci'].mean()
                print(f"✓ Target: {target_avg:.1f} per 100,000 per year")

                ratio = avg_incidence / target_avg
                print(f"✓ Ratio: {ratio:.2f}x")
            else:
                print(f"\n⚠ process_model returned empty dataframe")
        else:
            print(f"\n✗ NO INFECTIONS - disease died out!")

print("\n" + "="*80)
print("✓ Test complete")
print("="*80)
