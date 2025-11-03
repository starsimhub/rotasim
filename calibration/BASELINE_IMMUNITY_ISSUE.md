# Baseline Immunity Issue - CONFIRMED

## Problem Summary

The `baseline_immunity` parameter in the immunity connector is **NOT being maintained properly over time**. This causes adult baseline immunity to erode during long simulations with births and deaths.

## Root Cause

**Baseline immunity is only set ONCE at initialization** (t=0), but is never updated for:
1. New births during the simulation
2. Children who age into adulthood during the simulation

## Evidence from Diagnostic Test

Ran a 10-year simulation (1000 agents) with:
- Adult baseline immunity set to 0.99 at t=0
- Child baseline immunity set to 0.0 at t=0

### Results after 10 years:

**Initial state (t=0):**
- Adults (>=5y): 924 agents
- Mean baseline immunity: 0.990 ✓ (as expected)

**After 10 years:**
- Adults (>=5y): 990 agents
- Mean baseline immunity: **0.877** ✗ (should still be 0.99!)
- Min immunity: **0.000** ✗ (some adults have zero immunity!)
- Max immunity: 0.990 ✓ (original adults retained their immunity)
- New births during simulation: 69 agents

### What Happened:

1. At t=0, all 924 adults were assigned `baseline_immunity = 0.99`
2. Over 10 years, 69 new births occurred
3. These newborns got the default `baseline_immunity = 0.0`
4. As these children aged, some crossed the 5-year threshold and became "adults"
5. Their `baseline_immunity` was **NEVER updated** to 0.99
6. The adult population now contains a mix:
   - Original adults with 0.99 immunity
   - Children-turned-adults with 0.0 immunity
7. This caused the mean to drop to 0.877

## Impact on Calibration

This explains why the high adult immunity test (test_high_adult_immunity.py) failed:
- Expected: 70%+ child infections when adults have 0.99 immunity
- Observed: 94.5% adult infections (immunity not working)

The baseline immunity eroded from 0.99 to ~0.88 over the 10-year simulation, reducing adult protection and allowing infections to remain concentrated in adults instead of shifting to children.

## Solution Required

Baseline immunity needs to be **age-dependent and applied continuously**, not set once at initialization.

### Option 1: Update in connector step() method
Add logic to the immunity connector's `step()` method to set baseline immunity based on current age:

```python
def step(self):
    if len(self.rota_diseases) == 0:
        return

    # Update age-dependent baseline immunity
    ages_years = self.sim.people.age.values
    adult_mask = ages_years >= 5
    self.baseline_immunity[adult_mask] = self.pars.get('adult_baseline_immunity', 0.0)
    self.baseline_immunity[~adult_mask] = 0.0

    # Update cross-immunity protection
    self._update_cross_immunity()
```

### Option 2: Add calibration parameter
Add `adult_baseline_immunity` as a parameter to the immunity connector and apply it automatically based on age.

## Files Created

- `/Users/aliciakraay/PycharmProjects/rotasim/calibration/diagnose_baseline_immunity.py` - Diagnostic test that confirmed the issue
- `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_high_adult_immunity.py` - Test showing baseline immunity not working as expected
- `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_random_mixing.py` - Test confirming transmission dynamics work correctly

## CRITICAL UPDATE: Dynamic Immunity Test Reveals Deeper Problem

### Test with Perfect Dynamic Immunity Updating (test_dynamic_baseline_immunity.py)

To definitively test whether erosion was the only problem, implemented a test where baseline immunity is updated EVERY timestep:
- Adults (>=5y) ALWAYS get 0.99 baseline immunity (no erosion possible)
- Children (<5y) ALWAYS get 0.0 baseline immunity
- Infections seeded in adults initially
- Random mixing network

### Results - TEST FAILED:
```
Year 0 (seeded in adults):     88.9% adult infections ✓
Years 1-3 (early transition):  84.5% adult infections
Years 3-6 (mid transition):    77.2% adult infections
Years 6-10 (late - expected shift): 88.8% adult infections ✗

Years 6-10 child infections: 11.2% (expected >70%)
```

### Conclusion

**The baseline immunity erosion is NOT the only problem!**

Even with perfect 0.99 baseline immunity maintained every single timestep for all adults, infections do not shift to children. This reveals a fundamental issue with:

1. **How baseline immunity is applied to susceptibility** - The immunity connector may not be correctly incorporating baseline_immunity into rel_sus calculations
2. **When baseline immunity is applied** - It may be set but then overwritten or ignored during transmission
3. **The immunity calculation logic** - The `np.maximum(acquired_immunity_protection, baseline_immunity)` may not be working as expected
4. **Transmission dynamics** - Something may be preventing immunity from properly reducing adult susceptibility

### True Root Cause

The issue is NOT just that baseline immunity erodes (though that is also true). The issue is that **baseline immunity does not properly protect adults from infection** even when it's present and maintained.

## STATUS UPDATE: Immunity Fix Works, But Transmission Bug Found!

### Fix Implemented and Verified ✓

**Modified `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py`:**
- Added `adult_baseline_immunity` and `adult_age_threshold` parameters
- Modified `step()` method to update baseline immunity every timestep based on age
- Verified: Adults maintain baseline_immunity = 0.99, rel_sus = 0.01 throughout simulation

### Critical Discovery: Transmission Bug!

**Paradox at Year 10 with random mixing:**

Population structure:
- Children (<5y): 6.3% of population
- Adults (>=5y): 93.7% of population

Effective susceptible pools (sum of rel_sus):
- Children: 245.3 (83.1% of total susceptible pool)
- Adults: 49.7 (16.9% of total susceptible pool)

**Expected with random mixing:** 83.1% child infections, 16.9% adult infections

**Actual observed:** 9.3% child infections, 90.7% adult infections

**Conclusion:** The immunity system is working correctly (adults have rel_sus = 0.01), but there's a fundamental bug in how `rel_sus` is being used in transmission probability calculations. Infections are going to the OPPOSITE group from what the susceptibility values predict!

## Next Steps

1. ~~Fix the baseline immunity system to be age-dependent~~ ✓ DONE - works correctly
2. ~~Add diagnostic logging to verify rel_sus values~~ ✓ DONE - confirmed working
3. **URGENT: Investigate transmission probability calculations** - rel_sus values are correct but not being applied correctly
4. **Check Rotavirus disease class** - Look at how rel_sus is used in infection probability
5. **Examine network/contact model** - Verify random mixing is actually random
