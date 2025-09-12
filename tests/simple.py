import sciris as sc
import rotasim as rs
import starsim as ss
import matplotlib.pyplot as plt

with sc.timer():
    sim = rs.Sim(
        verbose=True,
        n_agents=50000,
        initial_strains='high_diversity', # see utils.py for options
        fitness_scenario='default', # see utils.py for options
        init_prev=.0001, # percent of population initially infected with each strain
        start='2000-01-01', # simulation start date
        stop='2000-06-01', # simulation end date
        dt=1, # timestep size
        unit='days', # timestep units
        analyzers=[rs.EventStats(), rs.StrainStats()], # analyzers to collect data
        networks=ss.RandomNet(n_contacts=10), # contact network. n_contacts controls the number of contacts per agent per timestep
        demographics=[ss.Births(birth_rate=ss.peryear(70)), ss.Deaths(death_rate=ss.peryear(20))], # simple demographics, birth and death rates per 1000 per year
        base_beta=0.07, # base transmission rate (will be modified by fitness)
        use_preferred_partners=True # if True, preferred partners are used for reassortment (see utils.py).
    )
    sim.init()
    sim.run()

    print(sim)


    # Generate summary plot of strains
    # Extract strain data from analyzers
    for analyzer_name, analyzer in sim.analyzers.items():
        if analyzer_name == 'strainstats':
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
