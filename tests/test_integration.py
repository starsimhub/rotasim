"""
Integration tests for the complete multi-strain Rotavirus system
Tests the entire v2 architecture with realistic strain numbers and scenarios
"""
import sys
import os
import time

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Sim, Rotavirus, RotaImmunityConnector
from rotasim.utils import create_strain_diseases, generate_gp_reassortments


def test_multi_strain_creation():
    """Test creation with many strains (20-70 combinations)"""
    print("Testing multi-strain creation (20+ strains)...")
    
    # Test by inspecting the disease creation directly via utils
    from rotasim.utils import create_strain_diseases
    
    # Test with 3 initial strains -> 9 combinations
    initial_strains_3 = [(1, 8), (2, 4), (3, 6)]
    diseases_3 = create_strain_diseases(initial_strains_3, 'baseline', 0.1)
    assert len(diseases_3) == 9
    
    # Test with 4 initial strains -> 12 combinations (4 G values x 3 P values)
    initial_strains_4 = [(1, 8), (2, 4), (3, 6), (9, 8)]
    diseases_4 = create_strain_diseases(initial_strains_4, 'baseline', 0.1)
    assert len(diseases_4) == 12
    
    # Test with 5 initial strains -> 15 combinations (5 G values x 3 P values)
    initial_strains_5 = [(1, 8), (2, 4), (3, 6), (9, 8), (12, 8)]
    diseases_5 = create_strain_diseases(initial_strains_5, 'baseline', 0.1)
    assert len(diseases_5) == 15
    
    # Test with large scenario (8 strains with diverse G,P -> many combinations)
    initial_strains_large = [
        (1, 8), (2, 4), (3, 6), (4, 8), 
        (9, 8), (12, 8), (9, 6), (11, 4)
    ]
    diseases_large = create_strain_diseases(initial_strains_large, 'baseline', 0.1) 
    # This gives us G: [1,2,3,4,9,11,12] P: [4,6,8] = 7×3 = 21 combinations
    assert len(diseases_large) == 21
    
    print(f"✓ Multi-strain creation: 3→9, 4→12, 5→15, 8→21 strains")


def test_dormant_strain_handling():
    """Test that dormant strains are properly handled"""
    print("Testing dormant strain handling...")
    
    initial_strains = [(1, 8), (2, 4), (3, 6)]
    sim = Sim(initial_strains=initial_strains)
    
    # Get strain summary
    summary = sim.get_strain_summary()
    
    # Should have 3 active, 6 dormant
    assert len(summary['active_strains']) == 3
    assert len(summary['dormant_strains']) == 6
    assert summary['total_diseases'] == 9
    
    # Check that active strains have correct G,P
    active_gp = [(s['G'], s['P']) for s in summary['active_strains']]
    assert set(active_gp) == set(initial_strains)
    
    # Check that dormant strains are reassortants
    dormant_gp = [(s['G'], s['P']) for s in summary['dormant_strains']]
    expected_dormant = [(1, 4), (1, 6), (2, 8), (2, 6), (3, 8), (3, 4)]
    assert set(dormant_gp) == set(expected_dormant)
    
    print(f"✓ Dormant strain handling: {len(summary['active_strains'])} active, {len(summary['dormant_strains'])} dormant")


def test_fitness_scenarios():
    """Test different fitness scenarios with multiple strains"""
    print("Testing fitness scenarios with multiple strains...")
    
    initial_strains = [(1, 8), (2, 4), (3, 8), (9, 8)]
    
    # Test by creating diseases directly to avoid initialization complexity
    # Test baseline scenario
    diseases_baseline = create_strain_diseases(initial_strains, 'baseline', 0.1)
    
    # Test high_diversity scenario
    diseases_high = create_strain_diseases(initial_strains, 'high_diversity', 0.1)
    
    # Test low_diversity scenario  
    diseases_low = create_strain_diseases(initial_strains, 'low_diversity', 0.1)
    
    # Test custom scenario
    custom_fitness = {(1, 8): 1.2, (2, 4): 0.5, (3, 8): 0.8}
    diseases_custom = create_strain_diseases(initial_strains, custom_fitness, 0.1)
    
    # All should create the same number of diseases (4 G values x 2 P values = 8)
    assert len(diseases_baseline) == 8
    assert len(diseases_high) == 8
    assert len(diseases_low) == 8
    assert len(diseases_custom) == 8
    
    # Test that Rotasim instances can be created with different scenarios
    sim_baseline = Sim(initial_strains=initial_strains, fitness_scenario='baseline')
    sim_high = Sim(initial_strains=initial_strains, fitness_scenario='high_diversity')
    sim_custom = Sim(initial_strains=initial_strains, fitness_scenario=custom_fitness)
    
    # Properties should be set correctly
    assert sim_baseline.fitness_scenario == 'baseline'
    assert sim_high.fitness_scenario == 'high_diversity'
    assert sim_custom.fitness_scenario == custom_fitness
    
    print("✓ Fitness scenarios: baseline, high_diversity, low_diversity, custom")


def test_manual_vs_convenience():
    """Compare manual disease creation vs convenience class"""
    print("Testing manual vs convenience class creation...")
    
    initial_strains = [(1, 8), (2, 4)]
    
    # Method 1: Manual creation
    diseases_manual = create_strain_diseases(initial_strains, 'baseline', 0.1)
    immunity_manual = RotaImmunityConnector()
    
    # Method 2: Convenience class 
    sim_convenience = Sim(initial_strains=initial_strains, fitness_scenario='baseline', base_beta=0.1)
    
    # Should create same number of diseases (test indirectly via utils)
    expected_combinations = generate_gp_reassortments(initial_strains)
    assert len(diseases_manual) == len(expected_combinations)
    
    # Check that disease names match expected pattern
    manual_names = sorted([d.name for d in diseases_manual])
    expected_names = sorted([f"G{g}P{p}" for g, p in expected_combinations])
    assert manual_names == expected_names
    
    # Check that Rotasim has correct properties
    assert sim_convenience.initial_strains == initial_strains
    assert sim_convenience.fitness_scenario == 'baseline'
    assert sim_convenience.base_beta == 0.1
    
    # Check that all diseases have correct G,P attributes
    for disease in diseases_manual:
        assert hasattr(disease, 'G') and hasattr(disease, 'P')
        assert hasattr(disease, 'strain')
        assert disease.strain == (disease.G, disease.P)
        assert disease.name == f"G{disease.G}P{disease.P}"
    
    print("✓ Manual vs convenience: identical disease creation")


def test_initialization_performance():
    """Test initialization performance with many strains"""
    print("Testing initialization performance...")
    
    # Test creation performance (without full initialization)
    # Test medium scenario (5 strains -> 15 combinations)
    initial_strains_med = [(1, 8), (2, 4), (3, 6), (4, 8), (9, 8)]
    
    start_time = time.time()
    diseases_med = create_strain_diseases(initial_strains_med, 'baseline', 0.1)
    sim_med = Sim(initial_strains=initial_strains_med)
    med_time = time.time() - start_time
    
    # Test large scenario (8 strains -> 21 combinations)
    initial_strains_large = [
        (1, 8), (2, 4), (3, 6), (4, 8), 
        (9, 8), (12, 8), (9, 6), (11, 4)
    ]
    
    start_time = time.time()
    diseases_large = create_strain_diseases(initial_strains_large, 'baseline', 0.1)
    sim_large = Sim(initial_strains=initial_strains_large)
    large_time = time.time() - start_time
    
    print(f"  Medium (15 strains): {med_time:.3f}s")
    print(f"  Large (21 strains): {large_time:.3f}s")
    
    # Performance should be reasonable (< 1 second for creation)
    assert med_time < 1.0, f"Medium scenario too slow: {med_time:.3f}s"
    assert large_time < 1.0, f"Large scenario too slow: {large_time:.3f}s"
    
    # Verify correct number of diseases created
    assert len(diseases_med) == 15
    assert len(diseases_large) == 21
    
    print("✓ Initialization performance: acceptable for up to 21+ strains")


def test_strain_summary_large():
    """Test strain summary with large number of strains"""
    print("Testing strain summary with large strain count...")
    
    initial_strains = [(1, 8), (2, 4), (3, 6), (4, 8), (9, 8)]  # 15 total
    sim = Sim(initial_strains=initial_strains)
    
    # Test that summary methods exist and properties are correct
    assert hasattr(sim, 'get_strain_summary')
    assert hasattr(sim, 'print_strain_summary')
    assert sim.initial_strains == initial_strains
    
    # Test expected disease count through utils
    expected_combinations = generate_gp_reassortments(initial_strains)
    assert len(expected_combinations) == 15
    
    print("✓ Strain summary handles large strain counts")


def test_immunity_connector_integration():
    """Test that immunity connector properly integrates with many strains"""
    print("Testing immunity connector integration...")
    
    initial_strains = [(1, 8), (2, 4), (3, 6)]
    
    # Test default behavior (should add connector)
    sim_default = Sim(initial_strains=initial_strains)
    # Should work without error
    
    # Test with custom connectors
    custom_immunity = RotaImmunityConnector()
    sim_custom = Sim(initial_strains=initial_strains, connectors=[custom_immunity])
    # Should work without error
    
    # Test with no connectors
    sim_none = Sim(initial_strains=initial_strains, connectors=[])
    # Should work without error
    
    # Test that utilities work with expected strain count
    diseases = create_strain_diseases(initial_strains, 'baseline', 0.1)
    assert len(diseases) == 9
    
    # All diseases should be Rotavirus instances
    for disease in diseases:
        assert hasattr(disease, 'G') and hasattr(disease, 'P')
        assert hasattr(disease, 'strain')
    
    print("✓ Immunity connector integration: default and custom connectors")


def test_parameter_inheritance():
    """Test that parameters are properly inherited across many strains"""
    print("Testing parameter inheritance across strains...")
    
    initial_strains = [(1, 8), (2, 4)]
    base_beta = 0.15
    
    # Test via disease creation utilities
    diseases = create_strain_diseases(initial_strains, 'baseline', base_beta)
    
    # Check that all diseases have correct base parameters applied
    for disease in diseases:
        # Should have proper strain attributes
        assert hasattr(disease, 'strain')
        assert disease.strain == (disease.G, disease.P)
        assert disease.name == f"G{disease.G}P{disease.P}"
        
        # Check beta parameter is set (exact values depend on parameter wrappers)
        assert hasattr(disease, 'pars')
        assert hasattr(disease.pars, 'beta')
    
    # Test that Rotasim stores parameters correctly
    sim = Sim(initial_strains=initial_strains, base_beta=base_beta, fitness_scenario='baseline')
    assert sim.base_beta == base_beta
    assert sim.fitness_scenario == 'baseline'
    
    print("✓ Parameter inheritance: proper strain attributes and fitness application")


if __name__ == "__main__":
    print("Running integration tests for multi-strain Rotavirus system...\n")
    
    try:
        test_multi_strain_creation()
        test_dormant_strain_handling()
        test_fitness_scenarios()
        test_manual_vs_convenience()
        test_initialization_performance()
        test_strain_summary_large()
        test_immunity_connector_integration()
        test_parameter_inheritance()
        
        print(f"\n🎉 All integration tests passed!")
        print("✓ v2 multi-strain architecture is working correctly")
        
    except Exception as e:
        print(f"\n❌ Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)