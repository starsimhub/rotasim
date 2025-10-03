"""
Check results against baseline values.

NB: the two tests could be combined into one, but are left separate for clarity.
"""

import sciris as sc
import rotasim as rs
import starsim as ss

N = 500  # Smaller population for faster testing
timelimit = 1  # 1 year for faster testing
verbose = False

def test_default():
    """Test basic v2 simulation with default parameters"""
    sc.heading("Testing v2 default parameters")
    
    # Test basic simulation setup and run using new unified API
    with sc.timer() as T:
        sim = rs.Sim(
            scenario='simple',
            n_agents=N, 
            start='2020-01-01',
            stop='2020-04-01',  # Just 3 months for testing
            dt=ss.days(1),
            verbose=verbose
        )
        sim.run()
    
    print(f"V2 default simulation completed in {T.elapsed:.2f} seconds")
    
    # Basic validation
    assert len(sim.diseases) == 4  # 2 initial + 2 reassortants
    assert len(sim.connectors) == 2  # Immunity + Reassortment
    
    # Check that some infections occurred
    total_infections = sum(disease.results.cum_infections[-1] for disease in sim.diseases.values())
    assert total_infections > 0, "No infections occurred in simulation"
    
    print(f"Total cumulative infections: {total_infections}")
    
    # Additional validation
    assert len(sim.diseases) == 4, f"Expected 4 diseases, got {len(sim.diseases)}"
    assert len(sim.connectors) == 2, f"Expected 2 connectors, got {len(sim.connectors)}"


def test_alt():
    """Test v2 simulation with alternate parameters"""
    sc.heading("Testing v2 alternate parameters")
    
    # Test with baseline scenario and overrides
    with sc.timer() as T:
        sim = rs.Sim(
            # Use default baseline scenario
            override_prevalence=0.02,  # Override all prevalence values
            base_beta=0.15,
            n_agents=N,
            start='2020-01-01', 
            stop='2020-04-01',  # Just 3 months for testing
            dt=ss.days(1),
            verbose=verbose
        )
        sim.run()
    
    print(f"V2 alternate simulation completed in {T.elapsed:.2f} seconds")
    
    # Validation - baseline scenario has 3 strains, so 3x3=9 total diseases but 6 unique (G,P) combinations
    assert len(sim.diseases) == 6  # baseline scenario has 3 initial strains
    
    total_infections = sum(disease.results.cum_infections[-1] for disease in sim.diseases.values())
    assert total_infections > 0, "No infections occurred in alternate simulation"
    
    print(f"Total cumulative infections (alternate): {total_infections}")
    
    # Additional validation for alternate simulation
    assert len(sim.diseases) == 6, f"Expected 6 diseases, got {len(sim.diseases)}"
    assert len(sim.connectors) == 2, f"Expected 2 connectors, got {len(sim.connectors)}"

def test_basic_functionality():
    """Test that basic v2 functionality works correctly"""
    sc.heading("Testing v2 basic functionality")
    
    # Test strain generation with single strain
    sim = rs.Sim(scenario={
        'strains': {(1, 8): {'fitness': 1.0, 'prevalence': 0.01}},
        'default_fitness': 1.0
    })
    strain_summary = sim.get_strain_summary()
    assert strain_summary['total_diseases'] == 1  # Single strain = no reassortants
    
    # Test multi-strain setup  
    sim = rs.Sim(scenario='simple')
    strain_summary = sim.get_strain_summary()
    assert strain_summary['total_diseases'] == 4  # 2x2 reassortants
    
    print("V2 basic functionality tests passed")

if __name__ == "__main__":
    try:
        test_basic_functionality()
        test_default()
        test_alt()
        print("\n[SUCCESS] All v2 rotasim tests passed!")
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
