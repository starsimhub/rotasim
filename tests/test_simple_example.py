"""
Simple example demonstrating the clean Rotasim v2 API
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Rotasim, RotaImmunityConnector


def main():
    print("=== Simple Rotasim v2 Example ===\n")
    
    # Example 1: Basic usage with defaults
    print("1. Basic usage (default immunity connector):")
    sim1 = Rotasim(initial_strains=[(1, 8), (2, 4)])
    print(f"   Created simulation with {sim1.get_strain_summary()['total_diseases']} total strains")
    print(f"   Using fitness scenario: {sim1.fitness_scenario}")
    print()
    
    # Example 2: Custom fitness scenario
    print("2. Custom fitness scenario:")
    sim2 = Rotasim(
        initial_strains=[(1, 8), (2, 4), (3, 6)], 
        fitness_scenario='high_diversity',
        base_beta=0.15
    )
    print(f"   Created simulation with high diversity fitness")
    print()
    
    # Example 3: Custom connectors (standard ss.Sim pattern)
    print("3. Custom connectors:")
    custom_immunity = RotaImmunityConnector(waning_rate=1/365)  # Slower waning
    sim3 = Rotasim(
        initial_strains=[(1, 8), (2, 4)],
        connectors=[custom_immunity]
    )
    print(f"   Created simulation with custom immunity connector")
    print()
    
    # Example 4: No immunity connector
    print("4. No immunity (standard ss.Sim pattern):")
    sim4 = Rotasim(
        initial_strains=[(1, 8), (2, 4)],
        connectors=[]  # No connectors
    )
    print(f"   Created simulation without immunity")
    print()
    
    # Example 5: Custom simulation parameters (standard ss.Sim pattern)
    print("5. Custom simulation parameters:")
    sim5 = Rotasim(
        initial_strains=[(1, 8), (2, 4)],
        n_agents=5000,
        start=2025,  # Start year
        stop=2030,   # End year  
        dt=7  # Weekly timesteps (7 days)
    )
    print(f"   Created simulation: {sim5.pars.n_agents} agents, {sim5.pars.start}-{sim5.pars.stop}")
    print()
    
    print("✅ All examples completed successfully!")
    print("The new Rotasim API follows standard ss.Sim patterns while providing convenient multi-strain setup.")


if __name__ == "__main__":
    main()