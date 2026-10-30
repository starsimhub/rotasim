"""
Diagnose if the immunity connector fix is working correctly

This test directly checks:
1. Are adults getting baseline_immunity = 0.99?
2. Are adults getting rel_sus ~= 0.01?
3. Is this being maintained throughout the simulation?
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("DIAGNOSTIC: Testing Immunity Connector Fix")
print("="*80)

# Define diagnostic intervention
class ImmunityDiagnostic:
    """Intervention that checks immunity and susceptibility values"""
    __name__ = 'ImmunityDiagnostic'

    def __init__(self):
        self.initialized = False
        self.check_days = [1, 10, 100, 365, 365*2]  # Days to check

    def __call__(self, sim):
        if not self.initialized and sim.ti == 0:
            self.initialized = True

        # Check at specific timepoints
        if sim.ti in self.check_days:
            self._check_immunity_and_susceptibility(sim)

    def _check_immunity_and_susceptibility(self, sim):
        """Check baseline immunity and rel_sus values"""
        # Get immunity connector
        immunity_connector = None
        for connector in sim.connectors.values():
            if type(connector).__name__ == 'RotaImmunityConnector':
                immunity_connector = connector
                break

        if immunity_connector is None:
            print(f"\nDay {sim.ti}: ERROR - No immunity connector found!")
            return

        # Get disease
        disease = None
        for d in sim.diseases.values():
            if hasattr(d, 'G') and hasattr(d, 'P'):
                disease = d
                break

        if disease is None:
            print(f"\nDay {sim.ti}: ERROR - No disease found!")
            return

        # Get ages
        ages_years = sim.people.age.values
        adult_mask = ages_years >= 5
        child_mask = ages_years < 5

        # Get baseline immunity values
        adult_baseline_imm = immunity_connector.baseline_immunity[adult_mask]
        child_baseline_imm = immunity_connector.baseline_immunity[child_mask]

        # Get rel_sus values
        adult_rel_sus = disease.rel_sus[adult_mask]
        child_rel_sus = disease.rel_sus[child_mask]

        print(f"\n{'='*80}")
        print(f"Day {sim.ti} (Year {sim.ti/365:.2f})")
        print(f"{'='*80}")

        print(f"\nAdults (>=5y): {np.sum(adult_mask)} agents")
        print(f"  Baseline immunity:")
        print(f"    mean={adult_baseline_imm.mean():.4f}")
        print(f"    min={adult_baseline_imm.min():.4f}")
        print(f"    max={adult_baseline_imm.max():.4f}")
        print(f"    Expected: ALL should be 0.99")

        print(f"  Relative susceptibility:")
        print(f"    mean={adult_rel_sus.mean():.4f}")
        print(f"    min={adult_rel_sus.min():.4f}")
        print(f"    max={adult_rel_sus.max():.4f}")
        print(f"    Expected: max ~0.01 (1 - 0.99)")

        print(f"\nChildren (<5y): {np.sum(child_mask)} agents")
        print(f"  Baseline immunity:")
        print(f"    mean={child_baseline_imm.mean():.4f}")
        print(f"    min={child_baseline_imm.min():.4f}")
        print(f"    max={child_baseline_imm.max():.4f}")
        print(f"    Expected: ALL should be 0.0")

        print(f"  Relative susceptibility:")
        print(f"    mean={child_rel_sus.mean():.4f}")
        print(f"    min={child_rel_sus.min():.4f}")
        print(f"    max={child_rel_sus.max():.4f}")
        print(f"    Expected: ~1.0 for naive children")

        # Diagnostic checks
        if adult_baseline_imm.min() < 0.98:
            print(f"\n⚠ WARNING: Some adults have baseline immunity < 0.98!")
            print(f"  This means the fix is NOT working correctly")
        else:
            print(f"\n✓ All adults have baseline immunity >= 0.98")

        if adult_rel_sus.max() > 0.02:
            print(f"⚠ WARNING: Some adults have rel_sus > 0.02!")
            print(f"  This means immunity is not being properly applied")
        else:
            print(f"✓ Adult susceptibility looks correct")

# Create simulation with custom immunity connector
print("\nCreating simulation with custom immunity connector...")
print("  adult_baseline_immunity = 0.99")
print("  adult_age_threshold = 5.0")
print("  homotypic_immunity_efficacy = 0.2 (low - allows reinfections)")

sim = rs.Sim(
    n_agents=1000,
    start='2003-01-01',
    stop='2005-01-01',  # 2 years
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.01,
    networks='random',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            homotypic_immunity_efficacy=0.2,  # LOW - allows reinfections
            adult_baseline_immunity=0.99,  # HIGH - protects adults
            adult_age_threshold=5.0,  # Adults are >=5 years
        ),
    ],
    interventions=[ImmunityDiagnostic()],
)

print("\n✓ Simulation created")

# Run simulation
print("\n" + "="*80)
print("RUNNING SIMULATION...")
print("="*80)
sim.run()
print("\n✓ Simulation complete")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("\nIf adults have baseline_immunity < 0.98 or rel_sus > 0.02,")
print("then the fix is NOT working correctly.")
