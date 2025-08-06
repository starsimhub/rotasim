"""
Test script for Rotasim convenience class
Tests the high-level multi-strain simulation interface
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Rotasim, Rotavirus, RotaImmunityConnector


def test_rotasim_creation():
    """Test basic Rotasim instance creation"""
    print("Testing Rotasim creation...")
    
    # Test basic creation (now need to provide people and time parameters)
    import starsim as ss
    sim = Rotasim(
        initial_strains=[(1, 8), (2, 4)],
        people=ss.People(100),
        networks=ss.RandomNet(),
        start='2020-01-01',
        stop='2021-01-01'
    )
    
    # Check that it's a proper Sim instance
    assert hasattr(sim, 'pars')
    assert hasattr(sim, 'run')
    
    # Check properties (don't need to initialize for these)
    assert sim.initial_strains == [(1, 8), (2, 4)]
    assert sim.fitness_scenario == 'baseline'
    assert sim.base_beta == 0.1
    
    # Check time parameters were set (but not population/time defaults)
    assert sim.pars.unit == 'day'  # Default time unit
    assert sim.pars.dt == 1  # Default timestep
    
    print("✓ Rotasim creation tests passed")


def test_custom_parameters():
    """Test Rotasim with custom parameters"""
    print("Testing custom parameters...")
    
    # Test custom fitness scenario and parameters
    custom_fitness = {(1, 8): 1.2, (2, 4): 0.6}
    sim = Rotasim(
        initial_strains=[(1, 8), (2, 4)],
        fitness_scenario=custom_fitness,
        base_beta=0.15,
        n_agents=5000,
        start='2015-01-01',
        stop='2025-01-01'
    )
    
    # Check properties
    assert sim.fitness_scenario == custom_fitness
    assert sim.base_beta == 0.15
    assert sim.pars.n_agents == 5000
    assert sim.pars.start == 2015
    assert sim.pars.stop == 2025
    
    print("✓ Custom parameters tests passed")


def test_connector_control():
    """Test connector control"""
    print("Testing connector control...")
    
    # Test default behavior (should add RotaImmunityConnector)
    sim1 = Rotasim(initial_strains=[(1, 8)])
    # Just test that it doesn't crash
    
    # Test with custom connectors
    custom_connector = RotaImmunityConnector()
    sim2 = Rotasim(initial_strains=[(1, 8)], connectors=[custom_connector])
    # Just test that it doesn't crash
    
    # Test with no connectors
    sim3 = Rotasim(initial_strains=[(1, 8)], connectors=[])
    # Just test that it doesn't crash
    
    print("✓ Connector control tests passed")


def test_strain_summary():
    """Test strain summary functionality"""
    print("Testing strain summary...")
    
    sim = Rotasim(initial_strains=[(1, 8), (2, 4)])
    
    # For now, just test that the methods exist and don't crash
    # (Full testing would require simulation initialization)
    assert hasattr(sim, 'get_strain_summary')
    assert hasattr(sim, 'print_strain_summary')
    
    # Test that we can call the methods without crashing
    try:
        summary = sim.get_strain_summary()
        sim.print_strain_summary()
        print("  Summary methods work without initialization")
    except Exception as e:
        print(f"  Summary methods require initialization (expected): {e}")
    
    print("✓ Strain summary tests passed")


def test_fitness_scenarios():
    """Test fitness scenario handling"""
    print("Testing fitness scenarios...")
    
    # Test built-in scenarios
    scenarios = Rotasim.list_fitness_scenarios()
    assert isinstance(scenarios, dict)
    assert 'baseline' in scenarios
    assert 'high_diversity' in scenarios
    
    # Test using different built-in scenarios
    for scenario_name in ['baseline', 'high_diversity', 'low_diversity']:
        sim = Rotasim(initial_strains=[(1, 8)], fitness_scenario=scenario_name)
        assert sim.fitness_scenario == scenario_name
    
    print("✓ Fitness scenarios tests passed")


def test_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Test invalid initial_strains
    invalid_cases = [
        [],  # Empty
        [(1,)],  # Wrong format
        [('a', 8)],  # Non-integer
    ]
    
    for invalid in invalid_cases:
        try:
            Rotasim(initial_strains=invalid)
            assert False, f"Should have raised ValueError for {invalid}"
        except ValueError:
            pass  # Expected
    
    print("✓ Input validation tests passed")


def test_repr():
    """Test string representation"""
    print("Testing string representation...")
    
    sim = Rotasim(initial_strains=[(1, 8), (2, 4)])
    repr_str = repr(sim)
    
    # Should contain key information
    assert 'Rotasim' in repr_str
    assert '[(1, 8), (2, 4)]' in repr_str
    assert 'baseline' in repr_str
    
    print("✓ String representation tests passed")


if __name__ == "__main__":
    print("Running Rotasim convenience class tests...\n")
    
    try:
        test_rotasim_creation()
        test_custom_parameters()
        test_connector_control()
        test_strain_summary()
        test_fitness_scenarios()
        test_validation()
        test_repr()
        
        print(f"\n🎉 All Rotasim convenience class tests passed!")
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)