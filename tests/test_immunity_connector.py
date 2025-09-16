"""
Test script for RotaImmunityConnector
"""
import numpy as np
import starsim as ss
import sys
import os

# Add rotasim to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rotasim import Rotavirus, RotaImmunityConnector, PathogenMatch


def test_immunity_connector_creation():
    """Test basic connector creation and initialization"""
    print("Testing immunity connector creation...")
    
    # Create connector
    connector = RotaImmunityConnector()
    
    # Check parameter defaults
    assert hasattr(connector.pars, 'homotypic_immunity_efficacy')
    assert hasattr(connector.pars, 'full_waning_rate')
    
    # Check that state arrays are defined (but not initialized yet)
    assert hasattr(connector, 'exposed_G_bitmask')
    assert hasattr(connector, 'exposed_P_bitmask')
    assert hasattr(connector, 'has_immunity')
    
    print("[OK] Immunity connector creation tests passed")


def test_strain_matching():
    """Test static strain matching method"""
    print("Testing strain matching...")
    
    # Test with Rotavirus instances
    rota_g1p8 = Rotavirus(G=1, P=8)
    rota_g1p4 = Rotavirus(G=1, P=4)  # Same G, different P
    rota_g2p8 = Rotavirus(G=2, P=8)  # Different G, same P  
    rota_g2p4 = Rotavirus(G=2, P=4)  # Different G, different P
    
    # Test homotypic matching
    assert RotaImmunityConnector.match_strain(rota_g1p8, rota_g1p8) == PathogenMatch.HOMOTYPIC
    assert RotaImmunityConnector.match_strain(rota_g1p8, (1, 8)) == PathogenMatch.HOMOTYPIC
    
    # Test partial heterotypic matching (shared G or P)
    assert RotaImmunityConnector.match_strain(rota_g1p8, rota_g1p4) == PathogenMatch.PARTIAL_HETERO  # Same G
    assert RotaImmunityConnector.match_strain(rota_g1p8, rota_g2p8) == PathogenMatch.PARTIAL_HETERO  # Same P
    assert RotaImmunityConnector.match_strain(rota_g1p8, (1, 4)) == PathogenMatch.PARTIAL_HETERO
    assert RotaImmunityConnector.match_strain(rota_g1p8, (2, 8)) == PathogenMatch.PARTIAL_HETERO
    
    # Test complete heterotypic matching (no shared G,P)
    assert RotaImmunityConnector.match_strain(rota_g1p8, rota_g2p4) == PathogenMatch.COMPLETE_HETERO
    assert RotaImmunityConnector.match_strain(rota_g1p8, (2, 4)) == PathogenMatch.COMPLETE_HETERO
    
    print("[OK] Strain matching tests passed")


def test_connector_auto_detection():
    """Test that the connector auto-detects Rotavirus diseases"""
    print("Testing connector auto-detection...")
    
    # Create diseases and connector
    diseases = [
        Rotavirus(G=1, P=8, name="G1P8"),
        Rotavirus(G=2, P=4, name="G2P4"),
        Rotavirus(G=3, P=6, name="G3P6"),
    ]
    
    connector = RotaImmunityConnector()
    
    # Create simulation
    sim = ss.Sim(
        diseases=diseases,
        connectors=[connector],
        networks='random',
        n_agents=1000,
        start='2020-01-01',
        stop='2020-01-08',  # Very short sim just for initialization testing
        dt=ss.days(1),
        verbose=0
    )
    
    # Initialize simulation (this calls init_post)
    sim.init()
    
    # Check that we're looking at the same connector
    sim_connector = sim.connectors[0]  # Get connector from sim
    print(f"  - Same connector instance? {connector is sim_connector}")
    print(f"  - connector id: {id(connector)}")
    print(f"  - sim_connector id: {id(sim_connector)}")
    
    # Use the connector that's actually in the simulation
    connector = sim_connector
    
    # Check that connector found all diseases
    print(f"  - connector.rota_diseases length: {len(connector.rota_diseases)}")
    print(f"  - Expected: 3")
    assert len(connector.rota_diseases) == 3, f"Expected 3 diseases, got {len(connector.rota_diseases)}"
    assert all(isinstance(d, Rotavirus) for d in connector.rota_diseases)
    
    # Check that bitmask mappings were created
    assert len(connector.G_to_bit) == 3  # G1, G2, G3
    assert len(connector.P_to_bit) == 3  # P8, P4, P6
    assert len(connector.disease_G_masks) == 3
    assert len(connector.disease_P_masks) == 3
    
    print(f"  - Found diseases: {[d.name for d in connector.rota_diseases]}")
    print(f"  - G mappings: {connector.G_to_bit}")
    print(f"  - P mappings: {connector.P_to_bit}")
    
    print("[OK] Connector auto-detection tests passed")


if __name__ == "__main__":
    print("Running immunity connector tests...\n")
    
    try:
        test_immunity_connector_creation()
        test_strain_matching()
        test_connector_auto_detection()
        
        print(f"\n[SUCCESS] All immunity connector tests passed!")
        
    except Exception as e:
        print(f"\n[ERROR] Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)