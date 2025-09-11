import sciris as sc
import rotasim as rs
import starsim as ss

with sc.timer():
    sim = rs.Sim(
        verbose=True,
        # to_csv=False,
        n_agents=50000,
        # timelimit=10,
        # rota_kwargs={"vaccination_time": 5, "time_to_equilibrium": 2},
        initial_strains='low_diversity',
        fitness_scenario='2',
        init_prev=.001,
        start='2000-01-01',
        stop='2001-01-01',
        dt=1,
        unit='days',
        analyzers=[rs.EventStats(), rs.StrainStats()],
        networks=ss.RandomNet(n_contacts=7),
        demographics=[ss.Births(birth_rate=ss.peryear(0.5/70)), ss.Deaths(death_rate=ss.peryear(1/70))],
        base_beta=0.1,
    )
    sim.init()
    sim.run()

    # events = sim.connectors["rota"].event_dict
    print(sim)

    import matplotlib.pyplot as plt

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
