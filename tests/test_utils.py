"""
Test script for utility functions
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import generate_gp_reassortments, list_scenarios, SCENARIOS
from rotasim import Sim


def test_generate_gp_reassortments():
    """Test G,P combination generation"""
    print("Testing G,P reassortment generation...")
    
    # Test basic case
    initial = [(1, 8), (2, 4)]
    combinations = generate_gp_reassortments(initial)
    expected = [(1, 8), (1, 4), (2, 8), (2, 4)]
    assert set(combinations) == set(expected), f"Expected {expected}, got {combinations}"
    
    # Test single strain
    single = [(1, 8)]
    combinations = generate_gp_reassortments(single)
    assert combinations == [(1, 8)], f"Single strain should return itself, got {combinations}"
    
    # Test three strains
    three = [(1, 8), (2, 4), (3, 6)]
    combinations = generate_gp_reassortments(three)
    assert len(combinations) == 9, f"3x3 should give 9 combinations, got {len(combinations)}"
    assert (1, 6) in combinations, "Should include reassortant (1,6)"
    assert (3, 8) in combinations, "Should include reassortant (3,8)"
    
    print("G,P reassortment generation tests passed")


def test_unified_scenarios():
    """Test unified scenario system"""
    print("Testing unified scenario system...")
    
    # Test that built-in scenarios exist
    scenarios = list_scenarios()
    assert 'simple' in scenarios
    assert 'baseline' in scenarios
    assert 'high_diversity' in scenarios
    assert len(scenarios) >= 5, f"Expected at least 5 scenarios, got {len(scenarios)}"
    
    # Test SCENARIOS dictionary structure
    assert 'simple' in SCENARIOS
    simple_scenario = SCENARIOS['simple']
    assert 'strains' in simple_scenario
    assert 'default_fitness' in simple_scenario
    assert 'description' in simple_scenario
    
    # Test simple scenario structure
    assert (1, 8) in simple_scenario['strains']
    assert (2, 4) in simple_scenario['strains']
    assert simple_scenario['strains'][(1, 8)]['fitness'] == 1.0
    assert simple_scenario['strains'][(1, 8)]['prevalence'] == 0.01
    
    print("Unified scenario system tests passed")


def test_sim_strain_creation():
    """Test strain disease creation via Sim class"""
    print("Testing Sim strain creation...")
    
    # Test basic creation using simple scenario
    sim = Sim(
        scenario='simple',
        base_beta=0.1, 
        verbose=0
    )
    sim.init()  # Initialize to access diseases
    
    # Should create 4 diseases
    assert len(sim.diseases) == 4, f"Expected 4 diseases, got {len(sim.diseases)}"
    
    # Check names
    disease_list = list(sim.diseases.values())
    names = [d.name for d in disease_list]
    expected_names = ['G1P8', 'G1P4', 'G2P8', 'G2P4']
    assert set(names) == set(expected_names), f"Expected {expected_names}, got {names}"
    
    # Check that initial strains have prevalence
    initial_strains = [(1, 8), (2, 4)]  # From the scenario we created
    for disease in disease_list:
        if (disease.G, disease.P) in initial_strains:
            assert disease.pars.init_prev.pars['p'] == 0.01, f"{disease.name} should have init_prev=0.01"
        else:
            assert disease.pars.init_prev.pars['p'] == 0.0, f"{disease.name} should have init_prev=0.0"
    
    # Check fitness adjustment by examining the stored parameters
    g1p8_disease = next(d for d in disease_list if d.name == 'G1P8')
    g2p4_disease = next(d for d in disease_list if d.name == 'G2P4')
    
    # Check that diseases were created with correct G,P attributes
    assert g1p8_disease.G == 1 and g1p8_disease.P == 8
    assert g2p4_disease.G == 2 and g2p4_disease.P == 4
    
    # Check strain properties
    assert g1p8_disease.strain == (1, 8)
    assert g2p4_disease.strain == (2, 4)
    
    print("Sim strain creation tests passed")


def test_scenario_validation():
    """Test scenario validation"""
    print("Testing scenario validation...")
    
    # Test valid scenarios work
    for scenario_name in ['simple', 'baseline', 'high_diversity']:
        try:
            sim = Sim(scenario=scenario_name, verbose=0)
            print(f"  OK {scenario_name} scenario validated successfully")
        except Exception as e:
            assert False, f"Valid scenario {scenario_name} should not raise error: {e}"
    
    # Test invalid scenario
    try:
        sim = Sim(scenario='invalid_scenario_name', verbose=0)
        assert False, "Invalid scenario should raise ValueError"
    except ValueError:
        print("  OK Invalid scenario correctly rejected")
    
    print("Scenario validation tests passed")


if __name__ == "__main__":
    print("Running utility function tests...\n")
    
    try:
        test_generate_gp_reassortments()
        test_unified_scenarios()
        test_sim_strain_creation()
        test_scenario_validation()
        
        print(f"\nAll utility function tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)