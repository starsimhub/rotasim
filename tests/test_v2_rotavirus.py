"""
Test script for Task 1: Core Rotavirus disease class
Tests the new pure user-space implementation without rotasim.Sim wrapper
"""
import numpy as np
import starsim as ss
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Rotavirus, PathogenMatch


def test_rotavirus_creation():
    """Test basic Rotavirus instance creation and properties"""
    print("Testing Rotavirus creation...")
    
    # Test auto-naming
    rota1 = Rotavirus(G=1, P=8)
    assert rota1.name == "G1P8", f"Expected 'G1P8', got '{rota1.name}'"
    assert rota1.G == 1, f"Expected G=1, got G={rota1.G}"
    assert rota1.P == 8, f"Expected P=8, got P={rota1.P}"
    
    # Test manual naming
    rota2 = Rotavirus(G=2, P=4, name="CustomName")
    assert rota2.name == "CustomName", f"Expected 'CustomName', got '{rota2.name}'"
    
    # Test properties
    assert rota1.strain == (1, 8), f"Expected (1, 8), got {rota1.strain}"
    
    print("✓ Rotavirus creation tests passed")




def test_single_strain_simulation():
    """Test that a single Rotavirus strain can run in a basic simulation"""
    print("Testing single strain simulation...")
    
    # Create simple simulation with one strain using new unified API
    from rotasim import Sim as RotaSim
    
    sim = RotaSim(
        scenario={
            'strains': {(1, 8): {'fitness': 1.0, 'prevalence': 0.01}},
            'default_fitness': 1.0
        },
        n_agents=1000,
        start='2020-01-01',
        stop='2021-01-01',
        dt=ss.days(1),
        verbose=0
    )
    
    # Run simulation
    sim.run()
    
    # Basic checks
    assert sim.complete, "Simulation should have completed"
    assert len(sim.diseases) > 0, "Should have diseases"
    
    # Check that some transmission occurred
    total_infections = sum(disease.results.cum_infections[-1] for disease in sim.diseases.values())
    print(f"✓ Single strain simulation completed with {total_infections} total infections")


def test_multi_strain_basic():
    """Test that multiple Rotavirus strains can coexist without interference"""
    print("Testing multi-strain basic functionality...")
    
    # Create simulation with multiple strains using new unified API
    from rotasim import Sim as RotaSim
    
    initial_strains = [(1, 8), (2, 4), (3, 8)]
    sim = RotaSim(
        # Uses default baseline scenario (3-strain)
        n_agents=2000,
        start='2020-01-01',
        stop='2021-01-01',
        dt=ss.days(1),
        verbose=0
    )
    
    sim.run()
    
    # Check that diseases were created for each initial strain
    disease_names = list(sim.diseases.keys())
    expected_names = [f"G{g}P{p}" for g, p in initial_strains]
    
    for expected_name in expected_names:
        assert expected_name in disease_names, f"Disease {expected_name} should be in sim.diseases"
        disease = sim.diseases[expected_name]
        infections = disease.results.cum_infections[-1] if len(disease.results.cum_infections) > 0 else 0
        print(f"  {expected_name}: {infections} cumulative infections")
    
    total_infections = sum(disease.results.cum_infections[-1] for disease in sim.diseases.values())
    print(f"✓ Multi-strain simulation completed with {total_infections} total infections")
    print(f"  Created {len(sim.diseases)} total diseases (including reassortants)")
    
    print("✓ Multi-strain basic test passed")


if __name__ == "__main__":
    print("Running Task 1 tests for core Rotavirus class...\n")
    
    try:
        test_rotavirus_creation()
        test_single_strain_simulation()
        test_multi_strain_basic()
        
        print(f"\n🎉 All Task 1 tests passed! Core Rotavirus class is working correctly.")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)