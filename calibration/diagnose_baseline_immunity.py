"""
Diagnose if baseline immunity is persisting properly for adults over time
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("BASELINE IMMUNITY PERSISTENCE DIAGNOSTIC")
print("="*80)

# Define helper functions
def setup_high_adult_immunity(sim):
    """Set high adult baseline immunity"""
    immunity_connector = None
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            immunity_connector = connector
            break

    if immunity_connector is not None:
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

        if sim.pars.verbose:
            n_adults = np.sum(adult_mask)
            print(f"\n✓ Set baseline immunity:")
            print(f"  Adults (>=5y) with 0.99 immunity: {n_adults}")
            print(f"  Children (<5y) with 0.0 immunity: {len(ages_years) - n_adults}")

class CustomInit:
    """Intervention to modify sim state after initialization"""
    __name__ = 'CustomInit'

    def __init__(self):
        self.initialized = False

    def __call__(self, sim):
        if not self.initialized and sim.ti == 0:
            setup_high_adult_immunity(sim)
            self.initialized = True
        return

# Store initial UIDs globally
initial_uids = None

# Custom init that tracks state
class DiagnosticInit:
    """Intervention to modify sim state and track initial population"""
    __name__ = 'DiagnosticInit'

    def __init__(self):
        self.initialized = False

    def __call__(self, sim):
        global initial_uids

        if not self.initialized and sim.ti == 0:
            # Set baseline immunity
            setup_high_adult_immunity(sim)

            # Store initial UIDs globally
            initial_uids = set(range(len(sim.people)))

            # Get immunity connector and check initial state
            immunity_connector = None
            for connector in sim.connectors.values():
                if type(connector).__name__ == 'RotaImmunityConnector':
                    immunity_connector = connector
                    break

            if immunity_connector is not None:
                ages = sim.people.age.values
                adult_mask = ages >= 5
                child_mask = ages < 5

                adult_immunity = immunity_connector.baseline_immunity[adult_mask]
                child_immunity = immunity_connector.baseline_immunity[child_mask]

                print("\n" + "="*80)
                print("INITIAL STATE (t=0)")
                print("="*80)
                print(f"Adults (>=5y): {np.sum(adult_mask)}")
                print(f"  Mean baseline immunity: {adult_immunity.mean():.3f}")
                print(f"  Should be: 0.99")
                print(f"\nChildren (<5y): {np.sum(child_mask)}")
                print(f"  Mean baseline immunity: {child_immunity.mean():.3f}")
                print(f"  Should be: 0.0")

            self.initialized = True
        return

# Create simulation
print("\nCreating simulation...")
sim = rs.Sim(
    n_agents=1000,  # Small population for quick test
    start='2003-01-01',
    stop='2013-01-01',  # 10 years
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.001,
    networks='random',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    interventions=[DiagnosticInit()],
)

print("\n✓ Simulation created")

# Run for 10 years
print("\n" + "="*80)
print("RUNNING SIMULATION FOR 10 YEARS...")
print("="*80)
sim.run()
print("  ✓ Simulation complete")

# Check state after 10 years
print("\n" + "="*80)
print("STATE AFTER 10 YEARS")
print("="*80)

# Get immunity connector
immunity_connector = None
for connector in sim.connectors.values():
    if type(connector).__name__ == 'RotaImmunityConnector':
        immunity_connector = connector
        break

if immunity_connector is None:
    print("ERROR: Could not find immunity connector")
    sys.exit(1)

ages = sim.people.age.values
adult_mask = ages >= 5
child_mask = ages < 5

# Check which agents are from the initial population vs new births
current_uids = set(range(len(sim.people)))
new_uids = current_uids - initial_uids
n_new_births = len(new_uids)

print(f"Total agents: {len(sim.people)}")
print(f"  From initial population: {len(sim.people) - n_new_births}")
print(f"  New births: {n_new_births}")

# Analyze immunity by group
adult_immunity = immunity_connector.baseline_immunity[adult_mask]
child_immunity = immunity_connector.baseline_immunity[child_mask]

print(f"\nAdults (>=5y): {np.sum(adult_mask)}")
print(f"  Mean baseline immunity: {adult_immunity.mean():.3f}")
print(f"  Should still be: 0.99")
print(f"  Min immunity: {adult_immunity.min():.3f}")
print(f"  Max immunity: {adult_immunity.max():.3f}")

# Check specifically for adults who were children or new births
adult_uids = np.where(adult_mask)[0]
adults_from_initial = [uid for uid in adult_uids if uid in initial_uids]
adults_new_births = [uid for uid in adult_uids if uid in new_uids]

print(f"\n  Adults from initial pop: {len(adults_from_initial)}")
if len(adults_from_initial) > 0:
    immunity_initial = immunity_connector.baseline_immunity[adults_from_initial]
    print(f"    Mean immunity: {immunity_initial.mean():.3f} (should be 0.99)")

print(f"  Adults from new births: {len(adults_new_births)}")
if len(adults_new_births) > 0:
    immunity_newborns = immunity_connector.baseline_immunity[adults_new_births]
    print(f"    Mean immunity: {immunity_newborns.mean():.3f} (probably 0.0!)")

print(f"\nChildren (<5y): {np.sum(child_mask)}")
print(f"  Mean baseline immunity: {child_immunity.mean():.3f}")

# Diagnosis
print("\n" + "="*80)
print("DIAGNOSIS")
print("="*80)

if len(adults_new_births) > 0:
    immunity_newborns = immunity_connector.baseline_immunity[adults_new_births]
    if immunity_newborns.mean() < 0.5:
        print("✗ PROBLEM CONFIRMED:")
        print("  Agents born during simulation don't get adult baseline immunity")
        print("  when they age into adulthood (>=5 years)")
        print("\n  This is why high adult immunity isn't working:")
        print("  - At t=0: All adults have 0.99 immunity")
        print("  - New births: Get default 0.0 immunity")
        print("  - When births age to adults: Immunity stays at 0.0 (not updated!)")
        print("  - Over time: Adult population gets diluted with 0-immunity agents")
        print("\n  SOLUTION:")
        print("  Baseline immunity must be set based on AGE every timestep,")
        print("  not just once at initialization.")
    else:
        print("✓ No problem detected - immunity is being maintained")
else:
    print("No adults from new births yet (need longer simulation)")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
