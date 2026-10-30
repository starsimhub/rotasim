"""
Check the age distribution produced by the model vs target
"""
import sciris as sc
import rotasim as rs
import starsim as ss

thisdir = sc.thispath(__file__)
process_incidence = sc.importbypath(thisdir / 'process_incidence.py')

# Quick test - no reporting rate
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

print("Running simulation...")
sim.run()

for analyzer in sim.analyzers.values():
    if type(analyzer).__name__ == 'InfectedStrainStats':
        df = analyzer.to_df()

        # Process through incidence calculator
        incidence_df = process_incidence.process_model(dat=df, popsize=5000, verbose=False)

        print('\nModel output (no reporting rate):')
        print(incidence_df)

        if len(incidence_df) > 0:
            total = incidence_df['inci'].sum()
            print(f'\n{"="*60}')
            print('MODEL age distribution:')
            print(f'{"="*60}')
            for idx, row in incidence_df.iterrows():
                pct = (row['inci'] / total) * 100
                print(f"  Age {row['ages']}: {pct:5.1f}% ({row['inci']:8.1f} per 100k)")

            print(f'\n{"="*60}')
            print('TARGET age distribution (for comparison):')
            print(f'{"="*60}')
            data = process_incidence.process_data()
            target_total = data['inci'].sum()
            for idx, row in data.iterrows():
                pct = (row['inci'] / target_total) * 100
                print(f"  Age {row['ages']}: {pct:5.1f}% ({row['inci']:8.1f} per 100k)")

            print(f'\n{"="*60}')
            print('COMPARISON:')
            print(f'{"="*60}')
            print(f"{'Age':<10} {'Model %':<12} {'Target %':<12} {'Difference':<12}")
            print("-"*50)
            for idx, row in incidence_df.iterrows():
                model_pct = (row['inci'] / total) * 100
                if idx < len(data):
                    target_pct = (data.iloc[idx]['inci'] / target_total) * 100
                    diff = model_pct - target_pct
                    print(f"{row['ages']:<10} {model_pct:5.1f}%      {target_pct:5.1f}%      {diff:+5.1f}%")

print("\n✓ Done")
