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
    
    # Test basic simulation setup and run
    with sc.timer() as T:
        sim = rs.Sim(
            initial_strains=[(1, 8), (2, 4)],
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
    return sim


def test_alt():
    """Test v2 simulation with alternate parameters"""
    sc.heading("Testing v2 alternate parameters")
    
    # Test with custom fitness scenario and different parameters
    with sc.timer() as T:
        sim = rs.Sim(
            initial_strains=[(1, 8), (2, 4)],  # Use 2 strains instead of 3 for speed
            fitness_scenario='high_diversity',
            base_beta=0.15,
            init_prev=0.02,
            n_agents=N,
            start='2020-01-01', 
            stop='2020-04-01',  # Just 3 months for testing
            dt=ss.days(1),
            verbose=verbose
        )
        sim.run()
    
    print(f"V2 alternate simulation completed in {T.elapsed:.2f} seconds")
    
    # Validation
    assert len(sim.diseases) == 4  # 2 initial strains = 2x2 reassortants
    
    total_infections = sum(disease.results.cum_infections[-1] for disease in sim.diseases.values())
    assert total_infections > 0, "No infections occurred in alternate simulation"
    
    print(f"Total cumulative infections (alternate): {total_infections}")
    return sim

def test_basic_functionality():
    """Test that basic v2 functionality works correctly"""
    sc.heading("Testing v2 basic functionality")
    
    # Test strain generation
    sim = rs.Sim(initial_strains=[(1, 8)])
    strain_summary = sim.get_strain_summary()
    assert strain_summary['total_diseases'] == 1  # Single strain = no reassortants
    
    # Test multi-strain setup  
    sim = rs.Sim(initial_strains=[(1, 8), (2, 4)])
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
