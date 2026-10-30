"""
Test how symptomatic filtering (first 3 infections only) affects incidence calculation
"""
import sciris as sc
import rotasim as rs
import starsim as ss

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

print("="*80)
print("Testing Symptomatic Filtering")
print("="*80)

# Run simulation
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

print("\nRunning simulation...")
sim.run()

# Get infection data
for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()

        print(f"\nRaw infection counts:")
        print(f"  Total infections (all): {len(df)}")

        # Check infections per person
        infections_per_person = df.groupby('id').size()
        print(f"  Max infections per person: {infections_per_person.max()}")
        print(f"  Mean infections per person: {infections_per_person.mean():.2f}")
        print(f"  Median infections per person: {infections_per_person.median():.1f}")

        # Distribution
        print(f"\n  Distribution of infections per person:")
        for n in [1, 2, 3, 4, 5, 10, 20, 50]:
            count = (infections_per_person >= n).sum()
            print(f"    >= {n} infections: {count} people")

        # Now process through incidence calculator
        print(f"\n{'='*80}")
        print("Processing through incidence calculator (with symptomatic filtering)...")
        print(f"{'='*80}")

        incidence_df = process_incidence.process_model(dat=df, popsize=5000, verbose=True)

        print(f"\nProcessed incidence (symptomatic only):")
        print(incidence_df)

        # Calculate average across ages
        avg_incidence = incidence_df['inci'].mean()
        print(f"\n✓ Average incidence across all ages: {avg_incidence:.1f} per 100,000 per year")

        # Compare to target
        data = process_incidence.process_data()
        target_avg = data['inci'].mean()
        print(f"✓ Target average from calibration data: {target_avg:.1f} per 100,000 per year")

        ratio = avg_incidence / target_avg
        print(f"\n✓ Model/Target ratio: {ratio:.2f}x")

        if ratio < 2.0:
            print(f"✓✓✓ EXCELLENT! Within 2x of target!")
        elif ratio < 10.0:
            print(f"✓✓ GOOD! Within 10x of target")
        elif ratio < 100.0:
            print(f"✓ Fair - within 100x of target")
        else:
            print(f"⚠ Still far from target")

print("\n" + "="*80)
print("✓ Test complete")
print("="*80)
