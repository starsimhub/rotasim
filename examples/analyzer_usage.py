#!/usr/bin/env python3
"""
Example: Using analyzers with Rotasim simulations

This example demonstrates the correct way to use analyzers with Rotasim v2 architecture.
Key concepts:
- Starsim creates new analyzer instances during simulation setup
- Access analyzers via sim.analyzers['module_name'] using ndict keys
- Multiple analyzers can be used together for comprehensive data collection
"""
import starsim as ss
import rotasim as rs

def analyzer_usage_example():
    """Demonstrate proper analyzer usage with Rotasim"""
    
    print("=== Rotasim Analyzer Usage Example ===\n")
    print("Creating simulation with multiple analyzers...")

    # Create population and network
    pop = ss.People(n_agents=1000)
    net = ss.RandomNet()

    # Create multiple analyzers for comprehensive data collection
    strain_analyzer = rs.StrainStats()    # Track strain proportions and counts
    event_analyzer = rs.EventStats()      # Track simulation events
    age_analyzer = rs.AgeStats()          # Track age distribution
    
    # Create simulation with analyzers
    sim = rs.Rotasim(
        people=pop,
        networks=net,
        initial_strains=[(1, 8), (2, 4)],  # Two initial strains 
        analyzers=[strain_analyzer, event_analyzer, age_analyzer],
        start='2020-01-01', 
        stop='2020-03-01',  # Short simulation for demo
        unit='day',
        dt=1          # Daily timesteps
    )

    print("✓ Simulation created successfully")
    print(f"  Initial strains: {sim.initial_strains}")
    print(f"  Population: {sim.pars.n_agents} agents")
    print(f"  Duration: {sim.pars.start}-{sim.pars.stop}")
    
    # Run simulation
    print("\nRunning simulation...")
    sim.run()
    print("✓ Simulation completed!")
    
    # IMPORTANT: Access analyzers from sim.analyzers, not original references
    # Starsim creates new analyzer instances during setup
    print(f"\nAnalyzer access:")
    print(f"  sim.analyzers type: {type(sim.analyzers)}")
    print(f"  Available keys: {list(sim.analyzers.keys())}")
    
    # Access analyzers using ndict keys (recommended approach)
    print(f"\n=== Results Collection ===")
    
    # StrainStats: Track strain dynamics
    if 'strainstats' in sim.analyzers:
        strain_results = sim.analyzers['strainstats']
        strain_df = strain_results.to_df()
        
        if strain_df is not None:
            print(f"✓ StrainStats: {strain_df.shape} (rows × columns)")
            print(f"  Columns: {list(strain_df.columns)[:3]}...")  # Show first 3
            print(f"  V1-compatible format: '(G, P, A, B) proportion/count'")
        else:
            print("❌ StrainStats: No data collected")
    
    # EventStats: Track simulation events  
    if 'eventstats' in sim.analyzers:
        event_results = sim.analyzers['eventstats']
        event_df = event_results.to_df()
        
        if event_df is not None:
            print(f"✓ EventStats: {event_df.shape}")
            print(f"  Events tracked: {list(event_df.columns)}")
        else:
            print("❌ EventStats: No data collected")
            
    # AgeStats: Track population age distribution
    if 'agestats' in sim.analyzers:
        age_results = sim.analyzers['agestats']
        age_df = age_results.to_df()
        
        if age_df is not None:
            print(f"✓ AgeStats: {age_df.shape}")
            print(f"  Age bins: {list(age_df.columns)[:5]}...")  # Show first 5
        else:
            print("❌ AgeStats: No data collected")
    
    print(f"\n=== Key Takeaways ===")
    print(f"1. Use sim.analyzers['module_name'] to access analyzer results")
    print(f"2. Module names come from the 'module=' parameter in ss.Result()")
    print(f"3. Multiple analyzers work together seamlessly")
    print(f"4. All analyzers maintain v1 CSV output compatibility")
    
    return sim

if __name__ == "__main__":
    sim = analyzer_usage_example()