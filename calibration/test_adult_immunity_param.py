"""
Test to verify adult_baseline_immunity parameter is passed correctly
"""
import sys
sys.path.append('/Users/aliciakraay/PycharmProjects/rotasim')

import numpy as np
import sciris as sc
import starsim as ss
import rotasim as rs

print("="*80)
print("TESTING ADULT_BASELINE_IMMUNITY PARAMETER PASSING")
print("="*80)

# Test 1: Direct parameter passing during sim creation
print("\n1. Test direct parameter passing during initialization...")
test_value_1 = 0.85

sim1 = rs.Sim(
    n_agents=1000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    connectors=[
        rs.RotaImmunityConnector(adult_baseline_immunity=test_value_1)
    ],
)

sim1.init()
actual_value_1 = sim1.connectors.rotaimmunityconnector.pars['adult_baseline_immunity']
print(f"   Expected: {test_value_1:.2f}")
print(f"   Actual:   {actual_value_1:.2f}")
print(f"   {'✓ PASS' if abs(actual_value_1 - test_value_1) < 0.01 else '✗ FAIL'}")

# Test 2: Setting parameter AFTER init (like calibrate_uk.py does)
print("\n2. Test setting parameter AFTER init (calibrate_uk.py method)...")
test_value_2 = 0.92

sim2 = rs.Sim(
    n_agents=1000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    connectors=[
        rs.RotaImmunityConnector()  # Default value (0.0)
    ],
)

sim2.init()

# This is what calibrate_uk.py does on line 282
sim2.connectors.rotaimmunityconnector.pars.adult_baseline_immunity = test_value_2

actual_value_2 = sim2.connectors.rotaimmunityconnector.pars['adult_baseline_immunity']
print(f"   Expected: {test_value_2:.2f}")
print(f"   Actual:   {actual_value_2:.2f}")
print(f"   {'✓ PASS' if abs(actual_value_2 - test_value_2) < 0.01 else '✗ FAIL'}")

# Test 3: Verify it's actually used in baseline_immunity array
print("\n3. Test if parameter is actually used in baseline_immunity array...")
test_value_3 = 0.88
adult_age_threshold = 5.0

sim3 = rs.Sim(
    n_agents=1000,
    start='2010-01-01',
    stop='2010-02-01',
    verbose=False,
    scenario='single',
    connectors=[
        rs.RotaImmunityConnector()
    ],
    demographics=[
        ss.Births(birth_rate=ss.peryear(13)),
        ss.Deaths(death_rate=ss.peryear(6)),
    ],
)

sim3.init()

# Set parameter after init (like calibrate_uk.py)
sim3.connectors.rotaimmunityconnector.pars.adult_baseline_immunity = test_value_3

# Call initialize_immunity (like calibrate_uk.py does on line 283)
sim3.connectors.rotaimmunityconnector.initialize_immunity(
    min_age=adult_age_threshold,
    max_age=125,
    min_exposures=2,
    max_exposures=3
)

# Check if adults have the correct baseline immunity
adults = sim3.people.age.values >= adult_age_threshold
baseline_immunity_adults = sim3.connectors.rotaimmunityconnector.baseline_immunity[adults]

if len(baseline_immunity_adults) > 0:
    mean_adult_immunity = baseline_immunity_adults.mean()
    print(f"   Expected adult baseline immunity: {test_value_3:.2f}")
    print(f"   Actual mean adult baseline immunity: {mean_adult_immunity:.2f}")
    print(f"   {'✓ PASS' if abs(mean_adult_immunity - test_value_3) < 0.01 else '✗ FAIL'}")

    # Show distribution
    num_adults = adults.sum()
    num_with_immunity = (baseline_immunity_adults > 0).sum()
    print(f"   Adults in population: {num_adults}")
    print(f"   Adults with baseline immunity: {num_with_immunity}")
else:
    print("   ✗ FAIL - No adults in population to test")

# Test 4: Verify it persists through step() method
print("\n4. Test if parameter persists through step() method...")
sim3.step()  # Run one timestep

# Check again after step
baseline_immunity_adults_after = sim3.connectors.rotaimmunityconnector.baseline_immunity[adults]
if len(baseline_immunity_adults_after) > 0:
    mean_after = baseline_immunity_adults_after.mean()
    print(f"   Expected after step: {test_value_3:.2f}")
    print(f"   Actual after step: {mean_after:.2f}")
    print(f"   {'✓ PASS' if abs(mean_after - test_value_3) < 0.01 else '✗ FAIL'}")
else:
    print("   ✗ FAIL - No adults in population after step")

print("\n" + "="*80)
print("SUMMARY")
print("="*80)
print("\nThe parameter passing mechanism in calibrate_uk.py (line 282) should work")
print("because it sets the parameter BEFORE calling initialize_immunity() (line 283).")
print("\nHowever, init_post() runs with the DEFAULT value (0.0) during sim.init().")
print("This is okay because initialize_immunity() is called explicitly afterwards")
print("and it uses the updated parameter value.")
print("="*80)
