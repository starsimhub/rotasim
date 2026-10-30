"""
Diagnose the population age distribution to understand why infections stay in adults

This test checks what proportion of the population is adults vs children
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim/calibration')

import numpy as np
import rotasim as rs
import starsim as ss

print("="*80)
print("POPULATION AGE DISTRIBUTION DIAGNOSTIC")
print("="*80)

# Define population tracking intervention
class PopulationTracker:
    """Track population age distribution over time"""
    __name__ = 'PopulationTracker'

    def __init__(self):
        self.check_years = [0, 1, 2, 5, 10]

    def __call__(self, sim):
        year = sim.ti / 365.0
        if any(abs(year - check_year) < 0.01 for check_year in self.check_years):
            self._report_population(sim, year)

    def _report_population(self, sim, year):
        ages_years = sim.people.age.values

        infants = np.sum(ages_years < 1)
        age_1_2 = np.sum((ages_years >= 1) & (ages_years < 2))
        age_2_5 = np.sum((ages_years >= 2) & (ages_years < 5))
        children_total = np.sum(ages_years < 5)
        adults = np.sum(ages_years >= 5)
        total = len(ages_years)

        print(f"\n{'='*80}")
        print(f"Year {year:.1f}: Population = {total} agents")
        print(f"{'='*80}")
        print(f"Age distribution:")
        print(f"  <1 year:    {infants:5d} ({infants/total*100:5.1f}%)")
        print(f"  1-2 years:  {age_1_2:5d} ({age_1_2/total*100:5.1f}%)")
        print(f"  2-5 years:  {age_2_5:5d} ({age_2_5/total*100:5.1f}%)")
        print(f"  <5 years:   {children_total:5d} ({children_total/total*100:5.1f}%)")
        print(f"  >=5 years:  {adults:5d} ({adults/total*100:5.1f}%)")
        print(f"{'='*80}")

        # Calculate expected infection distribution assuming random mixing
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

        # Calculate effective susceptible pools
        child_mask = ages_years < 5
        adult_mask = ages_years >= 5

        child_rel_sus = disease.rel_sus[child_mask]
        adult_rel_sus = disease.rel_sus[adult_mask]

        child_sus_pool = np.sum(child_rel_sus)
        adult_sus_pool = np.sum(adult_rel_sus)
        total_sus_pool = child_sus_pool + adult_sus_pool

        expected_child_pct = child_sus_pool / total_sus_pool * 100 if total_sus_pool > 0 else 0
        expected_adult_pct = adult_sus_pool / total_sus_pool * 100 if total_sus_pool > 0 else 0

        print(f"\nEffective susceptible pools (sum of rel_sus):")
        print(f"  Children (<5y): {child_sus_pool:7.1f} ({expected_child_pct:5.1f}% of total)")
        print(f"  Adults (>=5y):  {adult_sus_pool:7.1f} ({expected_adult_pct:5.1f}% of total)")
        print(f"\nWith random mixing, expect:")
        print(f"  {expected_child_pct:.1f}% of infections in children")
        print(f"  {expected_adult_pct:.1f}% of infections in adults")

# Create simulation matching the test parameters
print("\nCreating simulation with same parameters as test...")
print("  n_agents = 5000")
print("  birth_rate = 13 per 1000 per year")
print("  death_rate = 6 per 1000 per year")
print("  adult_baseline_immunity = 0.99")

sim = rs.Sim(
    n_agents=5000,
    start='2003-01-01',
    stop='2013-01-01',  # 10 years
    verbose=True,
    scenario='single',
    base_beta=0.16,
    override_prevalence=0.002,
    networks='random',
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
    connectors=[
        rs.RotaImmunityConnector(
            homotypic_immunity_efficacy=0.2,
            adult_baseline_immunity=0.99,
            adult_age_threshold=5.0,
        ),
    ],
    interventions=[PopulationTracker()],
)

print("\n" + "="*80)
print("RUNNING SIMULATION...")
print("="*80)
sim.run()
print("\n✓ Simulation complete")

print("\n" + "="*80)
print("DIAGNOSTIC COMPLETE")
print("="*80)
print("\nIf the effective susceptible pool for adults is >> than for children,")
print("then the population imbalance explains why adults get most infections.")
