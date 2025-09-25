"""
Test script for Rotasim convenience class
Tests the high-level multi-strain simulation interface
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Sim, Rotavirus, RotaImmunityConnector


def test_rotasim_creation():
    """Test basic Rotasim instance creation"""
    print("Testing Rotasim creation...")
    
    # Test basic creation using new unified API
    import starsim as ss
    sim = Sim(
        n_agents=100,
        start='2020-01-01',
        stop='2021-01-01'
    )
    
    # Check that it's a proper Sim instance
    assert hasattr(sim, 'pars')
    assert hasattr(sim, 'run')
    
    # Check properties (don't need to initialize for these)
    assert sim.initial_strains == [(1, 8), (2, 4), (3, 8)]  # baseline scenario strains
    assert sim.scenario == 'baseline'
    assert sim.base_beta == 0.1
    
    # Check time parameters were set
    assert sim.pars.dt == 1  # Default timestep in days
    
    print("OK Rotasim creation tests passed")


def test_custom_parameters():
    """Test Rotasim with custom parameters"""
    print("Testing custom parameters...")
    
    # Test custom scenario with overrides
    sim = Sim(
        override_fitness={(1, 8): 1.2, (2, 4): 0.6},
        base_beta=0.15,
        n_agents=5000,
        start='2015-01-01',
        stop='2025-01-01'
    )
    
    # Check properties
    assert sim.scenario == 'baseline'
    assert sim.base_beta == 0.15
    assert sim.pars.n_agents == 5000
    assert sim.pars.start == '2015-01-01'
    assert sim.pars.stop == '2025-01-01'
    
    # Check override was applied
    final_scenario = sim.final_scenario
    assert final_scenario['strains'][(1, 8)]['fitness'] == 1.2
    assert final_scenario['strains'][(2, 4)]['fitness'] == 0.6
    
    print("OK Custom parameters tests passed")


def test_connector_control():
    """Test connector control"""
    print("Testing connector control...")
    
    # Test default behavior (should add RotaImmunityConnector)
    sim1 = Sim()
    # Just test that it doesn't crash
    
    # Test with custom connectors
    custom_connector = RotaImmunityConnector()
    sim2 = Sim(connectors=[custom_connector])
    # Just test that it doesn't crash
    
    # Test with no connectors
    sim3 = Sim(connectors=[])
    # Just test that it doesn't crash
    
    print("OK Connector control tests passed")


def test_strain_summary():
    """Test strain summary functionality"""
    print("Testing strain summary...")
    
    sim = Sim()
    
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
    
    print("OK Strain summary tests passed")


def test_fitness_scenarios():
    """Test unified scenario handling"""
    print("Testing unified scenarios...")
    
    # Test built-in scenarios through utils
    from rotasim.utils import list_scenarios
    scenarios = list_scenarios()
    assert isinstance(scenarios, dict)
    
    # Test that we can get available scenarios dynamically
    assert len(scenarios) > 0
    print(f"  Found {len(scenarios)} unified scenarios")
    
    # Test using 'baseline' scenario (should always be available)
    if 'baseline' in scenarios:
        sim_baseline = Sim()
        assert sim_baseline.scenario == 'baseline'
        print("  OK Baseline scenario works")
    
    # Test using first available scenario (whatever it is)
    first_scenario = list(scenarios.keys())[0]
    sim_first = Sim(scenario=first_scenario)
    assert sim_first.scenario == first_scenario
    print(f"  OK Scenario '{first_scenario}' works")
    
    print("OK Unified scenarios tests passed")


def test_validation():
    """Test input validation"""
    print("Testing input validation...")
    
    # Test invalid scenario names
    invalid_scenarios = [
        'nonexistent_scenario',
        'invalid-name',
        123,  # Wrong type
    ]
    
    for invalid in invalid_scenarios:
        try:
            Sim(scenario=invalid)
            assert False, f"Should have raised ValueError for {invalid}"
        except (ValueError, TypeError):
            pass  # Expected
    
    # Test invalid custom scenario format
    invalid_custom_scenarios = [
        {},  # Missing strains
        {'strains': []},  # Empty strains
        {'strains': {(1, 8): 'invalid'}},  # Wrong strain data format
    ]
    
    for invalid in invalid_custom_scenarios:
        try:
            Sim(scenario=invalid)
            assert False, f"Should have raised ValueError for {invalid}"
        except ValueError:
            pass  # Expected
    
    print("OK Input validation tests passed")


def test_repr():
    """Test string representation"""
    print("Testing string representation...")
    
    sim = Sim()
    repr_str = repr(sim)
    
    # Should contain key information
    assert 'Sim' in repr_str  # Class name should be in repr
    assert 'scenario=baseline' in repr_str  # Should show scenario
    assert 'strains=3' in repr_str  # Should show strain count
    print(f"  Repr string: {repr_str}")  # Debug output
    
    print("OK String representation tests passed")


if __name__ == "__main__":
    print("Running Rotasim convenience class tests...\n")
    
    try:
        test_rotasim_creation()
        test_custom_parameters()
        test_connector_control()
        test_strain_summary()
        test_fitness_scenarios()
        # Rename to reflect that we're now testing unified scenarios
        test_validation()
        test_repr()
        
        print(f"\nOK All Rotasim convenience class tests passed!")
        
    except Exception as e:
        print(f"\nFAILED Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)