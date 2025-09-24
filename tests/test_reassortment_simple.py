"""
Simple test to isolate reassortment connector issues
"""
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Sim, RotaReassortmentConnector


def test_basic_creation():
    """Test basic creation without full initialization"""
    print("=== Basic Creation Test ===")
    
    # Test connector creation
    connector = RotaReassortmentConnector()
    print(f"✓ Created connector: {type(connector).__name__}")
    print(f"  Probability parameter: {connector.pars.reassortment_prob}")
    
    # Test Rotasim creation with no connectors (should avoid the issue)
    print("\n=== Rotasim with No Connectors ===")
    sim = Sim(initial_strains=[(1, 8), (2, 4)], connectors=[])
    print("✓ Created Rotasim with no connectors")
    
    # Try to initialize this one
    try:
        sim.init()
        print("✓ Initialized successfully with no connectors")
        print(f"  Connectors: {list(sim.connectors.keys())}")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
    
    print("\n=== Test Single Connector ===")
    # Try with just reassortment connector
    reassortment = RotaReassortmentConnector()
    sim2 = Sim(initial_strains=[(1, 8), (2, 4)], connectors=[reassortment])
    print("✓ Created Rotasim with only reassortment connector")
    
    try:
        sim2.init()
        print("✓ Initialized successfully with reassortment connector")
    except Exception as e:
        print(f"❌ Failed to initialize: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_basic_creation()