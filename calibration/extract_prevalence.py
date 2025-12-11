"""
Extract endemic prevalence from a calibration run

This script runs a simulation with given parameters and calculates
the endemic prevalence over time.
"""
import sys
import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

# Fix for PyCharm: Remove parent directory from sys.path
problem_path = '/Users/aliciakraay/PycharmProjects/ryan_rotasim'
if problem_path in sys.path:
    sys.path.remove(problem_path)

thisdir = sc.thispath(__file__)

def calculate_endemic_prevalence(base_beta=0.4, reporting_rate=0.01, n_agents=100000,
                                  burn_in_years=5, measurement_years=5):
    """
    Run simulation and calculate endemic prevalence

    Args:
        base_beta: Transmission rate
        reporting_rate: Reporting rate (doesn't affect prevalence)
        n_agents: Population size
        burn_in_years: Years to reach equilibrium
        measurement_years: Years to measure prevalence after burn-in

    Returns:
        dict with prevalence statistics
    """
    print(f"\nRunning simulation with:")
    print(f"  base_beta: {base_beta}")
    print(f"  n_agents: {n_agents}")
    print(f"  burn_in: {burn_in_years} years")
    print(f"  measurement: {measurement_years} years")

    # Create simulation
    people = ss.People(n_agents=n_agents, age_data=thisdir / 'uk_age_data.csv')
    sim = rs.Sim(
        n_agents=n_agents,
        start=0,
        stop=365 * (burn_in_years + measurement_years),  # Convert years to days
        dt=1,
        verbose=False,
        scenario='single',
        people=people,
        networks=ss.RandomNet(n_contacts=7),
        demographics=[
            ss.Births(birth_rate=ss.peryear(13)),
            ss.Deaths(death_rate=ss.peryear(6)),
        ],
    )

    # Initialize first
    sim.init()

    # Update base_beta after init
    for disease in sim.diseases.values():
        disease.pars.beta = ss.perday(base_beta)
    print("\nRunning simulation...")
    sim.run()

    # Extract prevalence time series
    disease = sim.diseases[0]

    # Calculate prevalence at each timepoint (after burn-in)
    burn_in_days = 365 * burn_in_years
    measurement_days = 365 * measurement_years

    prevalence_over_time = []
    symptomatic_prev_over_time = []
    asymptomatic_prev_over_time = []

    for ti in range(int(burn_in_days), int(burn_in_days + measurement_days)):
        if ti < len(disease.results['n_infected']):
            n_infected = disease.results['n_infected'][ti]
            n_alive = disease.results['n_alive'][ti] if 'n_alive' in disease.results else n_agents

            prevalence = n_infected / n_alive if n_alive > 0 else 0
            prevalence_over_time.append(prevalence)

    prevalence_array = np.array(prevalence_over_time)

    # Calculate statistics
    results = {
        'mean_prevalence': np.mean(prevalence_array),
        'median_prevalence': np.median(prevalence_array),
        'min_prevalence': np.min(prevalence_array),
        'max_prevalence': np.max(prevalence_array),
        'std_prevalence': np.std(prevalence_array),
        'mean_prevalence_pct': np.mean(prevalence_array) * 100,
        'n_agents': n_agents,
        'base_beta': base_beta,
    }

    return results

if __name__ == '__main__':
    print("="*60)
    print("Endemic Prevalence Calculator")
    print("="*60)

    # You can change these parameters to match your calibration
    results = calculate_endemic_prevalence(
        base_beta=0.4,  # Change this to your calibrated value
        n_agents=100000,
        burn_in_years=5,
        measurement_years=5,
    )

    print("\n" + "="*60)
    print("Endemic Prevalence Results")
    print("="*60)
    print(f"\nBase beta: {results['base_beta']}")
    print(f"Population: {results['n_agents']:,}")
    print(f"\nPrevalence (proportion infected):")
    print(f"  Mean:   {results['mean_prevalence']:.6f} ({results['mean_prevalence_pct']:.4f}%)")
    print(f"  Median: {results['median_prevalence']:.6f}")
    print(f"  Range:  {results['min_prevalence']:.6f} - {results['max_prevalence']:.6f}")
    print(f"  Std:    {results['std_prevalence']:.6f}")

    print("\n" + "="*60)
    print("Interpretation:")
    print("="*60)
    print(f"At equilibrium, approximately {results['mean_prevalence_pct']:.2f}% of the")
    print(f"population is infected with rotavirus at any given time.")
    print(f"\nThis means about {int(results['mean_prevalence'] * results['n_agents']):,} people")
    print(f"out of {results['n_agents']:,} are infected on average.")
