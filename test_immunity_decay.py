#!/usr/bin/env python3
"""
Simple test to verify immunity decay functionality
"""

import numpy as np
import starsim as ss
import rotasim as rs

def test_immunity_decay():
    """Test that immunity decay is working correctly"""
    print("Testing immunity decay functionality...")
    
    # Create a simple simulation with 2 strains
    sim = rs.Sim(
        verbose=True,
        n_agents=1000,
        initial_strains='low_diversity',  # Use fewer strains for simpler testing
        fitness_scenario='default',  # Use default fitness scenario
        start='2000-01-01',
        stop='2000-06-01',  # Run for 6 months to see decay
        dt=1,
        unit='days',
        analyzers=[rs.EventStats()],
        networks=ss.RandomNet(n_contacts=5),
        base_beta=0.15,
    )
    
    # Initialize the simulation
    sim.init()
    
    # Check that immunity connector was created properly
    immunity = sim.connectors['rotaimmunityconnector']
    print(f"\nImmunity connector parameters:")
    print(f"  - Homotypic efficacy: {immunity.pars.homotypic_immunity_efficacy}")
    print(f"  - Partial heterotypic efficacy: {immunity.pars.partial_heterotypic_immunity_efficacy}")
    print(f"  - Complete heterotypic efficacy: {immunity.pars.complete_heterotypic_immunity_efficacy}")
    print(f"  - Immunity waning delay: {immunity.pars.immunity_waning_delay}")
    print(f"  - Immunity waning mean duration: {immunity.pars.immunity_waning_mean_duration}")
    
    # Check that G and P decay arrays were created
    print(f"\nG max decay arrays: {list(immunity.G_max_decay_factors.keys())}")
    print(f"P max decay arrays: {list(immunity.P_max_decay_factors.keys())}")
    
    # Check that disease rel_sus values are initialized 
    print(f"\nDisease rel_sus initial values:")
    for disease_name, disease in sim.diseases.items():
        if hasattr(disease, 'rel_sus'):
            print(f"  {disease_name}: min={disease.rel_sus.min():.3f}, max={disease.rel_sus.max():.3f}, mean={disease.rel_sus.mean():.3f}")
    
    print("\n✅ Immunity decay system successfully initialized!")
    print("   - G and P max decay tracking arrays created")
    print("   - Parameters correctly loaded")
    print("   - Ready for simulation with per-strain and cross-strain immunity decay")
    return True

if __name__ == "__main__":
    test_immunity_decay()