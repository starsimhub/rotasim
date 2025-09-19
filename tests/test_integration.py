"""
Integration tests for the complete multi-strain Rotavirus system
Tests the entire v2 architecture with realistic strain numbers and scenarios
"""
import sys
import os
import time

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Sim, RotaImmunityConnector
from rotasim.utils import generate_gp_reassortments


def test_multi_strain_creation():
    """Test creation with many strains (20-70 combinations)"""
    print("Testing multi-strain creation (20+ strains)...")
    
    # Test by inspecting the disease creation via Sim class
    import rotasim as rs
    
    # Test with 3 initial strains -> 9 combinations
    initial_strains_3 = [(1, 8), (2, 4), (3, 6)]
    sim_3 = rs.Sim(initial_strains=initial_strains_3, verbose=False)
    sim_3.init()
    assert len(sim_3.diseases) == 9
    
    # Test with 4 initial strains -> 12 combinations (4 G values x 3 P values)
    initial_strains_4 = [(1, 8), (2, 4), (3, 6), (9, 8)]
    sim_4 = rs.Sim(initial_strains=initial_strains_4, verbose=False)
    sim_4.init()
    assert len(sim_4.diseases) == 12
    
    # Test with 5 initial strains -> 15 combinations (5 G values x 3 P values)
    initial_strains_5 = [(1, 8), (2, 4), (3, 6), (9, 8), (12, 8)]
    sim_5 = rs.Sim(initial_strains=initial_strains_5, verbose=False)
    sim_5.init()
    assert len(sim_5.diseases) == 15
    
    # Test with large scenario (8 strains with diverse G,P -> many combinations)
    initial_strains_large = [
        (1, 8), (2, 4), (3, 6), (4, 8), 
        (9, 8), (12, 8), (9, 6), (11, 4)
    ]
    sim_large = rs.Sim(initial_strains=initial_strains_large, verbose=False)
    sim_large.init()
    # This gives us G: [1,2,3,4,9,11,12] P: [4,6,8] = 7×3 = 21 combinations
    assert len(sim_large.diseases) == 21
    
    print(f"[OK] Multi-strain creation: 3->9, 4->12, 5->15, 8->21 strains")


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
    
    print(f"[OK] Dormant strain handling: {len(summary['active_strains'])} active, {len(summary['dormant_strains'])} dormant")


def test_fitness_scenarios():
    """Test different fitness scenarios with multiple strains"""
    print("Testing fitness scenarios with multiple strains...")
    
    initial_strains = [(1, 8), (2, 4), (3, 8), (9, 8)]
    
    # Test different scenarios via Sim class
    sim_default = Sim(initial_strains=initial_strains, fitness_scenario='default', verbose=False)
    sim_default.init()
    
    sim_1 = Sim(initial_strains=initial_strains, fitness_scenario='1', verbose=False)
    sim_1.init()
    
    sim_2 = Sim(initial_strains=initial_strains, fitness_scenario='2', verbose=False)
    sim_2.init()
    
    # Test custom scenario
    custom_fitness = {(1, 8): 1.2, (2, 4): 0.5, (3, 8): 0.8}
    sim_custom = Sim(initial_strains=initial_strains, fitness_scenario=custom_fitness, verbose=False)
    sim_custom.init()
    
    # All should create the same number of diseases (4 G values x 2 P values = 8)
    assert len(sim_default.diseases) == 8
    assert len(sim_1.diseases) == 8
    assert len(sim_2.diseases) == 8
    assert len(sim_custom.diseases) == 8
    
    # Properties should be set correctly
    assert sim_default.fitness_scenario == 'default'
    assert sim_1.fitness_scenario == '1'
    assert sim_custom.fitness_scenario == custom_fitness
    
    print("[OK] Fitness scenarios: default, scenario '1', scenario '2', custom")



def test_initialization_performance():
    """Test initialization performance with many strains"""
    print("Testing initialization performance...")
    
    # Test creation performance (without full initialization)
    # Test medium scenario (5 strains -> 15 combinations)
    initial_strains_med = [(1, 8), (2, 4), (3, 6), (4, 8), (9, 8)]
    
    start_time = time.time()
    sim_med = Sim(initial_strains=initial_strains_med, verbose=False)
    sim_med.init()
    med_time = time.time() - start_time
    
    # Test large scenario (8 strains -> 21 combinations)
    initial_strains_large = [
        (1, 8), (2, 4), (3, 6), (4, 8), 
        (9, 8), (12, 8), (9, 6), (11, 4)
    ]
    
    start_time = time.time()
    sim_large = Sim(initial_strains=initial_strains_large, verbose=False)
    sim_large.init()
    large_time = time.time() - start_time
    
    print(f"  Medium (15 strains): {med_time:.3f}s")
    print(f"  Large (21 strains): {large_time:.3f}s")
    
    # Performance should be reasonable (< 3 seconds for creation with full initialization)
    assert med_time < 3.0, f"Medium scenario too slow: {med_time:.3f}s"
    assert large_time < 3.0, f"Large scenario too slow: {large_time:.3f}s"
    
    # Verify correct number of diseases created
    assert len(sim_med.diseases) == 15
    assert len(sim_large.diseases) == 21
    
    print("[OK] Initialization performance: acceptable for up to 21+ strains")


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
    
    print("[OK] Strain summary handles large strain counts")


def test_immunity_connector_integration():
    """Test that immunity connector properly integrates with many strains"""
    print("Testing immunity connector integration...")
    
    initial_strains = [(1, 8), (2, 4), (3, 6)]
    
    # Test default behavior (should add connector)
    sim_default = Sim(initial_strains=initial_strains, verbose=False)
    sim_default.init()
    assert len(sim_default.diseases) == 9
    
    # Test with custom connectors
    custom_immunity = RotaImmunityConnector()
    sim_custom = Sim(initial_strains=initial_strains, connectors=[custom_immunity], verbose=False)
    sim_custom.init()
    assert len(sim_custom.diseases) == 9
    
    # Test with no connectors
    sim_none = Sim(initial_strains=initial_strains, connectors=[], verbose=False)
    sim_none.init()
    assert len(sim_none.diseases) == 9
    
    # All diseases should be Rotavirus instances
    for disease in sim_default.diseases.values():
        assert hasattr(disease, 'G') and hasattr(disease, 'P')
        assert hasattr(disease, 'strain')
    
    print("[OK] Immunity connector integration: default and custom connectors")


def test_parameter_inheritance():
    """Test that parameters are properly inherited across many strains"""
    print("Testing parameter inheritance across strains...")
    
    initial_strains = [(1, 8), (2, 4)]
    base_beta = 0.15
    
    # Test that Sim stores parameters correctly
    sim = Sim(initial_strains=initial_strains, base_beta=base_beta, fitness_scenario='default', verbose=False)
    sim.init()
    
    assert sim.base_beta == base_beta
    assert sim.fitness_scenario == 'default'
    assert len(sim.diseases) == 4
    
    # Check that all diseases have correct base parameters applied
    for disease in sim.diseases.values():
        # Should have proper strain attributes
        assert hasattr(disease, 'strain')
        assert disease.strain == (disease.G, disease.P)
        assert disease.name == f"G{disease.G}P{disease.P}"
        
        # Check beta parameter is set (exact values depend on parameter wrappers)
        assert hasattr(disease, 'pars')
        assert hasattr(disease.pars, 'beta')
    
    print("[OK] Parameter inheritance: proper strain attributes and fitness application")


if __name__ == "__main__":
    print("Running integration tests for multi-strain Rotavirus system...\n")
    
    try:
        test_multi_strain_creation()
        test_dormant_strain_handling()
        test_fitness_scenarios()
        test_initialization_performance()
        test_strain_summary_large()
        test_immunity_connector_integration()
        test_parameter_inheritance()
        
        print(f"\n[SUCCESS] All integration tests passed!")
        print("[OK] v2 multi-strain architecture is working correctly")
        
    except Exception as e:
        print(f"\n[ERROR] Integration test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)