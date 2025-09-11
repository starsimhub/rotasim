import sciris as sc
import rotasim as rs
import starsim as ss
import matplotlib.pyplot as plt

with sc.timer():
    sim = rs.Sim(
        verbose=True,
        n_agents=50000,
        initial_strains='high_diversity', # in utils.py, options are 'high_diversity', 'low_diversity', 'custom'
        fitness_scenario='default',
        init_prev=.001,
        start='2000-01-01',
        stop='2001-01-01',
        dt=1,
        unit='days',
        analyzers=[rs.EventStats(), rs.StrainStats()],
        networks=ss.RandomNet(n_contacts=7),
        # demographics=[ss.Births(birth_rate=ss.peryear(0.5/70)), ss.Deaths(death_rate=ss.peryear(1/70))],
        base_beta=0.15,
        use_preferred_partners=True # if True, preferred partners are used for reassortment (see utils.py).
    )
    sim.init()
    sim.run()

    print(sim)



    # Extract strain data from analyzers
    for analyzer_name, analyzer in sim.analyzers.items():
        if analyzer_name == 'strainstats':
            # continue
        # if hasattr(analyzer, 'results'):
            results = analyzer.results

            # Plot strain counts over time
            plt.figure(figsize=(12, 6))
            # Find all strain count attributes (e.g., 'G1P8_count', 'G2P4_count')
            strain_count_keys = [key for key in results.keys() if key.endswith(' count')]

            for strain_key in strain_count_keys:
                strain_name = strain_key.replace(' count', '')  # Remove '_count' suffix
                counts = results[strain_key]
                plt.plot(sim.timevec, counts, 'o-', label=strain_name, alpha=0.7)

            plt.xlabel('Time (days)')
            plt.ylabel('Strain Count')
            plt.title('Rotavirus Strain Dynamics')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.show()
            break
