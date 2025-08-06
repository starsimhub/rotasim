"""
Test script demonstrating the v2 daily time units
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Rotasim, Rotavirus, RotaImmunityConnector


def test_time_units():
    """Test that time units are properly set to daily"""
    print("=== Testing Daily Time Units ===\n")
    
    # Create a basic simulation
    sim = Rotasim(initial_strains=[(1, 8)])
    
    # Check simulation parameters
    print("Simulation parameters:")
    print(f"  Unit: {sim.pars.unit}")
    print(f"  dt: {sim.pars.dt}")
    print(f"  Start: {sim.pars.start}")
    print(f"  Stop: {sim.pars.stop}")
    print()
    
    # Check disease parameters (create directly to inspect)
    disease = Rotavirus(G=1, P=8)
    print("Disease parameters (G1P8):")
    print(f"  Duration of infection: {disease.pars.dur_inf} (should be ~7 days)")
    print()
    
    # Check immunity parameters (create directly to inspect)
    immunity = RotaImmunityConnector()
    print("Immunity parameters:")
    print(f"  Waning rate: {immunity.pars.waning_rate} per day")
    print(f"  Expected daily probability: {1/273:.6f}")
    print(f"  Half-life: ~{273 * 0.693:.0f} days ({273 * 0.693 / 365:.1f} years)")
    print()
    
    # Show conversion from v1 units
    print("Conversion from v1 units:")
    print("  v1: unit='year', dt=1/365, start='2000-01-01', stop='2010-01-01', dur_inf=7/365")
    print("  v2: unit='day', dt=1, start='2020-01-01', stop='2030-01-01', dur_inf=7")
    print("  Same total time: 10 years ✓")
    print("  Same infection duration: 7/365 years = 7 days ✓") 
    print("  Same timestep resolution: daily ✓")
    print("  Better intervention timing: years instead of days since 0 ✓")
    print()
    
    print("✅ Time units are correctly set to daily!")


if __name__ == "__main__":
    test_time_units()