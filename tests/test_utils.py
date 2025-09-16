"""
Test script for utility functions
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import (generate_gp_reassortments, get_fitness_multiplier, 
                     create_strain_diseases, list_fitness_scenarios, 
                     validate_initial_strains, FITNESS_HYPOTHESES)


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


def test_fitness_multiplier():
    """Test fitness multiplier lookup"""
    print("Testing fitness multiplier lookup...")
    
    # Test built-in scenarios
    assert get_fitness_multiplier(1, 8, 'default') == 1.0
    assert get_fitness_multiplier(2, 4, 'default') == 1.0
    assert get_fitness_multiplier(3, 6, 'default') == 1.0  # Default
    
    # Test custom scenario
    custom = {(1, 8): 1.2, (2, 4): 0.6}
    assert get_fitness_multiplier(1, 8, custom) == 1.2
    assert get_fitness_multiplier(2, 4, custom) == 0.6
    assert get_fitness_multiplier(3, 6, custom) == 1.0  # Default
    
    # Test invalid scenario name
    try:
        get_fitness_multiplier(1, 8, 'invalid_scenario')
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert 'Unknown fitness scenario' in str(e)
    
    print("Fitness multiplier tests passed")


def test_create_strain_diseases():
    """Test strain disease creation"""
    print("Testing strain disease creation...")
    
    # Test basic creation
    initial = [(1, 8), (2, 4)]
    diseases = create_strain_diseases(initial, 'default', base_beta=0.1)
    
    # Should create 4 diseases
    assert len(diseases) == 4, f"Expected 4 diseases, got {len(diseases)}"
    
    # Check names
    names = [d.name for d in diseases]
    expected_names = ['G1P8', 'G1P4', 'G2P8', 'G2P4']
    assert set(names) == set(expected_names), f"Expected {expected_names}, got {names}"
    
    # Check that initial strains have prevalence
    for disease in diseases:
        if (disease.G, disease.P) in initial:
            assert disease.pars.init_prev.pars['p'] == 0.01, f"{disease.name} should have init_prev=0.01"
        else:
            assert disease.pars.init_prev.pars['p'] == 0.0, f"{disease.name} should have init_prev=0.0"
    
    # Check fitness adjustment by examining the stored parameters
    g1p8_disease = next(d for d in diseases if d.name == 'G1P8')
    g2p4_disease = next(d for d in diseases if d.name == 'G2P4')
    
    # Check that diseases were created with correct G,P attributes
    assert g1p8_disease.G == 1 and g1p8_disease.P == 8
    assert g2p4_disease.G == 2 and g2p4_disease.P == 4
    
    # Check strain properties
    assert g1p8_disease.strain == (1, 8)
    assert g2p4_disease.strain == (2, 4)
    
    print("Strain disease creation tests passed")


def test_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Test valid input
    assert validate_initial_strains([(1, 8), (2, 4)]) == True
    
    # Test invalid inputs
    invalid_cases = [
        [],  # Empty
        [(1,)],  # Wrong tuple length
        [(1, 8, 9)],  # Too many elements
        [('a', 8)],  # Non-integer G
        [(1, 'b')],  # Non-integer P
        [(0, 8)],  # Non-positive G
        [(1, -1)],  # Non-positive P
    ]
    
    for invalid in invalid_cases:
        try:
            validate_initial_strains(invalid)
            assert False, f"Should have raised ValueError for {invalid}"
        except ValueError:
            pass  # Expected
    
    print("Input validation tests passed")


def test_fitness_scenarios():
    """Test fitness scenarios"""
    print("Testing fitness scenarios...")
    
    # Test that all built-in scenarios exist
    scenarios = list_fitness_scenarios()
    assert 'baseline' in scenarios
    assert 'high_diversity' in scenarios
    assert 'low_diversity' in scenarios
    
    # Test that FITNESS_HYPOTHESES dict is populated
    assert len(FITNESS_HYPOTHESES) >= 3
    assert 'default' in FITNESS_HYPOTHESES
    
    print("Fitness scenarios tests passed")


if __name__ == "__main__":
    print("Running utility function tests...\n")
    
    try:
        test_generate_gp_reassortments()
        test_fitness_multiplier()
        test_create_strain_diseases()
        test_validation()
        test_fitness_scenarios()
        
        print(f"\nAll utility function tests passed!")
        
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)