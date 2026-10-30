"""
Helper function to initialize a population with UK age distribution
"""

import numpy as np
import starsim as ss
import sciris as sc

def create_uk_population(n_agents=5000):
    """
    Create a People object with detailed UK age distribution

    Breaks down the 5+ category into more realistic age bins:
    - 5-18 years (children/teens)
    - 18-65 years (working age adults)
    - 65+ years (elderly)

    Args:
        n_agents (int): Number of agents to create

    Returns:
        ss.People: People object with correct age distribution
    """

    # Detailed UK age distribution
    # Breaking down the 93.81% "5+ years" category more realistically
    age_data = np.array([
        [0, 0.0126],      # 0-1 years: 1.26%
        [1, 0.0127],      # 1-2 years: 1.27%
        [2, 0.0366],      # 2-5 years: 3.66%
        [5, 0.18],        # 5-18 years: ~18% (school age)
        [18, 0.63],       # 18-65 years: ~63% (working age)
        [65, 0.13],       # 65+ years: ~13% (elderly)
        [100, 0.0],       # Upper bound (value=0 marks end)
    ])

    # Create People object with age distribution
    people = ss.People(n_agents, age_data=age_data)

    # Verify the distribution
    ages_years = people.age.values / 365.25  # Convert days to years

    print(f"Created population of {n_agents} agents with UK age distribution:")
    print(f"  Age range: {ages_years.min():.1f} - {ages_years.max():.1f} years")
    print(f"  Mean age: {ages_years.mean():.1f} years")
    print(f"  Median age: {np.median(ages_years):.1f} years")

    # Verify distribution
    age_bins_check = [
        (0, 1, 0.0126, '0-1'),
        (1, 2, 0.0127, '1-2'),
        (2, 5, 0.0366, '2-5'),
        (5, 18, 0.18, '5-18'),
        (18, 65, 0.63, '18-65'),
        (65, 100, 0.13, '65+'),
    ]

    print("\n  Age distribution verification:")
    print(f"  {'Age Range':12s} {'Count':>8s} {'Actual %':>10s} {'Target %':>10s} {'Error':>10s}")
    print("  " + "-"*55)
    for low, high, target_prop, label in age_bins_check:
        mask = (ages_years >= low) & (ages_years < high)
        count = mask.sum()
        actual_prop = count / n_agents
        error = actual_prop - target_prop
        print(f"  {label:12s} {count:8d} {actual_prop*100:9.2f}% {target_prop*100:9.2f}% {error*100:+9.2f}pp")

    return people


def create_uk_population_simple(n_agents=5000):
    """
    Simplified version matching calibration targets exactly

    UK Age Distribution (from calibration targets):
    - 0-1 years: 1.26%
    - 1-2 years: 1.27%
    - 2-5 years: 3.66%
    - 5+ years: 93.81%
    """

    # Create age_data in format expected by starsim
    # Format: Nx2 array with [age_bin_start, proportion/count]
    # Note: Ages in starsim age_data should be in YEARS (it converts to days internally)
    age_data = np.array([
        [0, 0.0126],      # 0-1 years: 1.26%
        [1, 0.0127],      # 1-2 years: 1.27%
        [2, 0.0366],      # 2-5 years: 3.66%
        [5, 0.9381],      # 5+ years: 93.81%
        [80, 0.0],        # Upper bound for 5+ category (value=0 to mark end)
    ])

    # Create People object with age distribution
    people = ss.People(n_agents, age_data=age_data)

    # Verify the distribution
    ages_years = people.age.values / 365.25  # Convert days to years

    # Print summary
    print(f"Created UK population with {n_agents} agents:")
    print(f"  Age range: {ages_years.min():.1f} - {ages_years.max():.1f} years")
    print(f"  Mean age: {ages_years.mean():.1f} years")
    print(f"  Median age: {np.median(ages_years):.1f} years")

    age_summary = {
        '<1 y': ((ages_years >= 0) & (ages_years < 1)).sum() / n_agents * 100,
        '1-2 y': ((ages_years >= 1) & (ages_years < 2)).sum() / n_agents * 100,
        '2-5 y': ((ages_years >= 2) & (ages_years < 5)).sum() / n_agents * 100,
        '>=5 y': (ages_years >= 5).sum() / n_agents * 100,
    }

    targets = {'<1 y': 1.26, '1-2 y': 1.27, '2-5 y': 3.66, '>=5 y': 93.81}

    print(f"\n  {'Age':8s} {'Actual %':>10s} {'Target %':>10s} {'Error':>10s}")
    print("  " + "-"*42)
    for age in ['<1 y', '1-2 y', '2-5 y', '>=5 y']:
        actual = age_summary[age]
        target = targets[age]
        error = actual - target
        print(f"  {age:8s} {actual:9.2f}% {target:9.2f}% {error:+9.2f}pp")

    return people


if __name__ == "__main__":
    print("="*80)
    print("Testing UK Population Initialization")
    print("="*80)

    print("\nOption 1: Detailed age distribution")
    print("-"*80)
    people1 = create_uk_population(5000)

    print("\n" + "="*80)
    print("\nOption 2: Simple (matches calibration targets)")
    print("-"*80)
    people2 = create_uk_population_simple(5000)

    print("\n" + "="*80)
    print("✓ Both population initializations work!")
    print("  Recommended: Use create_uk_population_simple() for calibration")
    print("="*80)
