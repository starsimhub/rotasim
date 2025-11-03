"""
Diagnose if baseline immunity is properly affecting rel_sus values

This test will:
1. Set high baseline immunity (0.99) for adults
2. Print actual rel_sus values for adults vs children
3. Check if rel_sus correctly reflects baseline immunity
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("SUSCEPTIBILITY DIAGNOSTIC TEST")
print("="*80)
print("\nThis test checks if baseline immunity properly reduces rel_sus values")
print("="*80)

# Define diagnostic intervention
class SusceptibilityDiagnostic:
    """Intervention that checks susceptibility values"""
    __name__ = 'SusceptibilityDiagnostic'

    def __init__(self):
        self.initialized = False
        self.check_days = [0, 1, 10, 100, 365, 365*2]  # Days to check

    def __call__(self, sim):
        if not self.initialized and sim.ti == 0:
            # Initial setup - set high adult baseline immunity
            self._setup_baseline_immunity(sim)
            self.initialized = True

        # Check susceptibility at specific timepoints
        if sim.ti in self.check_days:
            self._check_susceptibility(sim)

    def _setup_baseline_immunity(self, sim):
        """Set high baseline immunity for adults"""
        immunity_connector = None
        for connector in sim.connectors.values():
            if type(connector).__name__ == 'RotaImmunityConnector':
                immunity_connector = connector
                break

        if immunity_connector is None:
            print("ERROR: Could not find immunity connector")
            return

        # Set immunity parameters
        immunity_connector.pars['homotypic_immunity_efficacy'] = 0.2
        immunity_connector.pars['partial_heterotypic_immunity_efficacy'] = 0.1
        immunity_connector.pars['complete_heterotypic_immunity_efficacy'] = 0.05
        immunity_connector.pars['maternal_immunity_efficacy'] = 0.0

        # Set high baseline immunity for adults
        ages_years = sim.people.age.values
        adult_mask = ages_years >= 5

        immunity_connector.baseline_immunity[:] = 0.0
        immunity_connector.baseline_immunity[adult_mask] = 0.99

        print("\n✓ Set baseline immunity:")
        print(f"  Adults (>=5y): {np.sum(adult_mask)} agents with 0.99 baseline immunity")
        print(f"  Children (<5y): {np.sum(~adult_mask)} agents with 0.0 baseline immunity")

    def _check_susceptibility(self, sim):
        """Check and report susceptibility values"""
        # Get immunity connector
        immunity_connector = None
        for connector in sim.connectors.values():
            if type(connector).__name__ == 'RotaImmunityConnector':
                immunity_connector = connector
                break

        if immunity_connector is None:
            return

        # Get disease
        disease = None
        for d in sim.diseases.values():
            if hasattr(d, 'G') and hasattr(d, 'P'):
                disease = d
                break

        if disease is None:
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
        print(f"  Baseline immunity: mean={adult_baseline_imm.mean():.3f}, min={adult_baseline_imm.min():.3f}, max={adult_baseline_imm.max():.3f}")
        print(f"  Relative susceptibility: mean={adult_rel_sus.mean():.3f}, min={adult_rel_sus.min():.3f}, max={adult_rel_sus.max():.3f}")
        print(f"  Expected rel_sus if baseline immunity working: max=0.01 (1-0.99)")

        print(f"\nChildren (<5y): {np.sum(child_mask)} agents")
        print(f"  Baseline immunity: mean={child_baseline_imm.mean():.3f}, min={child_baseline_imm.min():.3f}, max={child_baseline_imm.max():.3f}")
        print(f"  Relative susceptibility: mean={child_rel_sus.mean():.3f}, min={child_rel_sus.min():.3f}, max={child_rel_sus.max():.3f}")
        print(f"  Expected rel_sus if no immunity: ~1.0")

        # Diagnostic check
        adult_max_sus = adult_rel_sus.max()
        if adult_max_sus > 0.02:  # Should be ~0.01
            print(f"\n⚠ WARNING: Adult max rel_sus is {adult_max_sus:.3f} (expected ~0.01)")
            print(f"  This suggests baseline immunity is NOT being properly applied!")
        else:
            print(f"\n✓ Adult susceptibility looks correct (max rel_sus = {adult_max_sus:.3f})")

# Create simulation
print("\nCreating simulation...")
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
    interventions=[SusceptibilityDiagnostic()],
)

print("\n✓ Simulation created")

# Run simulation
print("\n" + "="*80)
print("RUNNING SIMULATION...")
print("="*80)
sim.run()
print("\n✓ Simulation complete")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nIf adult max rel_sus is consistently > 0.02, then baseline immunity")
print("is NOT being properly applied to reduce susceptibility.")
