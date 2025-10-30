# UK Calibration Debugging - Bug Report and Fixes

## Summary

Successfully debugged the UK calibration errors. **No array mismatch errors occurred** - the script runs without crashes. However, discovered **3 critical bugs** preventing correct calibration.

## Bugs Found and Fixed

### ✅ Bug #1: Age Recording in Infections (FIXED)
**Location**: `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/analyzers.py:504`

**Problem**:
```python
age_years = self.sim.people.age[agent_id]  # Assumes age is in years
```

In starsim, `people.age` is stored in DAYS, not years. This caused:
- An infant who is 10 days old was recorded as "10 years old" → ">=5 y" category
- 97.77% of infections miscategorized as adults when they were actually infants

**Fix Applied**:
```python
age_days = self.sim.people.age[agent_id]
age_years = age_days / 365.25  # Convert from days to years
age_category = self._get_age_category(age_years)
```

**Status**: ✅ FIXED

---

### ✅ Bug #2: Maternal Immunity Age Calculation (FIXED)
**Location**: `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py:312`

**Problem**:
```python
agent_ages_days = self.sim.people.age.values * 365.25  # WRONG! Age is already in days
```

This multiplied age (already in days) by 365.25, giving completely incorrect values for maternal immunity calculations.

**Fix Applied**:
```python
agent_ages_days = self.sim.people.age.values  # Age is already in days
```

**Status**: ✅ FIXED (though maternal_immunity_efficacy=0 in current calibration, so not actively affecting results)

---

### 🚨 Bug #3: Population Not Aging (CRITICAL - NOT YET FIXED)
**Symptom**: After 10-year simulation:
- Max age: 70 days (0.19 years)
- 100% of population remains <1 year old
- No adults in population after 10 years

**Test Evidence** (`test_population_aging.py`):
```
Final population after 10 years:
  Total agents: 1069
  Age range: 0-70 days (0.00-0.19 years)
  Age distribution: 100% <1 year old
```

**Root Cause**: Starsim does **not automatically age the population**. The `Births` and `Deaths` modules handle population changes, but ages remain static unless manually updated.

**Impact on Calibration**:
- All infections occur in infants (because everyone IS an infant)
- Age distribution cannot match UK targets (13.8% age 0, 27.7% age 1, 46.9% age 2, 11.7% age 5+)
- Incidence is way too high (~650% over target) because infants keep getting reinfected
- Calibration completely fails

**Status**: 🚨 CRITICAL BUG - REQUIRES FIX

---

### 🚨 Bug #4: Excessive Reinfection Rate (CRITICAL - NOT YET FIXED)
**Symptom**:
- Mean: 56.65 infections per person in 5 years (1825 days)
- With 91-day (13-week) immunity, max should be ~20 infections
- Getting reinfected every 32 days on average

**Possible Causes**:
1. Population not aging → all infants → very high transmission in crowded infant population
2. Immunity waning not working correctly
3. Infection counting issue (counting each disease separately?)
4. SIRS cycle too fast

**Status**: 🚨 REQUIRES INVESTIGATION after fixing Bug #3

---

## Recommended Next Steps

### Step 1: Fix Population Aging

**Option A - Manual Aging in Rotasim** (RECOMMENDED):
Add an aging mechanism in rotasim's main simulation step:

```python
# In rotasim/rotasim.py or appropriate location
def step_people(self):
    """Update ages at each timestep"""
    if hasattr(self.people, 'age'):
        dt_days = self.pars.dt if isinstance(self.pars.dt, (int, float)) else self.pars.dt.days
        self.people.age[:] += dt_days  # Increment all ages by dt
```

**Option B - Use Starsim Plugin**:
Check if there's a starsim demographics plugin for aging, or create one.

**Option C - Initial Age Distribution**:
Initialize population with realistic age distribution instead of all age=0:

```python
# When creating initial population
import numpy as np

# UK age distribution (approximate)
age_dist = {
    (0, 1): 0.0126,    # 0-1 years: 1.26%
    (1, 2): 0.0127,    # 1-2 years: 1.27%
    (2, 5): 0.0366,    # 2-5 years: 3.66%
    (5, 100): 0.9381,  # 5+ years: 93.81%
}

# Sample initial ages from distribution
n_agents = len(sim.people)
ages = sample_from_age_distribution(n_agents, age_dist)
sim.people.age[:] = ages * 365.25  # Convert years to days
```

### Step 2: Test Aging Fix

Run `test_population_aging.py` and verify:
```
✅ Max age after 10 years: ~3650 days (10 years)
✅ Age distribution matches UK demographics
```

### Step 3: Rerun Calibration

After fixing aging:
1. Run `calibrate_uk.py`
2. Check that age distribution improves
3. Verify infection rates are reasonable (~20 per 100k instead of 650 per 100k)

### Step 4: Investigate Reinfection Rate

If still too high after fixing aging:
- Check immunity waning implementation
- Verify SIRS cycle (7 days infected → 91 days immune → susceptible)
- Check if `InfectedStrainStats` is overcounting (counting each strain separately vs episodes)

---

## Current Calibration Status

### What Works ✅
- Script runs without array mismatch errors
- Age recording now correct (after Bug #1 fix)
- Data processing time windows correct (years 5-10 for UK with burn-in)
- Calibration loop functional (finds best parameters)

### What Doesn't Work ❌
- **Population aging** - everyone stays as infants
- **Age distribution** - 98.7% adults in results when everyone is actually an infant
- **Incidence magnitude** - 650% too high (10.3 vs 1.4 per 100k)
- **Reinfection rate** - 56 infections/person in 5 years (unrealistic)

### Calibration Results (Before Fixing Aging)
```
Target:  1.4 per 100k
After calibration:  10.3 per 100k (+650% error)

Age Distribution:
  Age      Target    Actual    Error
  0        13.8%     0.0%      -13.8pp
  1        27.7%     0.0%      -27.6pp
  2        46.9%     1.2%      -45.6pp
  5+       11.7%     98.7%     +87.0pp
```

---

## Files Modified

1. `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/analyzers.py` (line 504-506)
   - Fixed age recording to convert days to years

2. `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py` (line 312)
   - Fixed maternal immunity age calculation

3. `/Users/aliciakraay/PycharmProjects/rotasim/calibration/process_incidence.py` (lines 80, 128)
   - Updated time windows from years 11-19 to years 5-10 for UK calibration

## Diagnostic Scripts Created

1. `test_calibrate_uk_debug.py` - Quick calibration test with 2 trials
2. `diagnose_age_issue.py` - Detailed age distribution analysis
3. `test_population_aging.py` - Simple test to verify population aging

---

## Conclusion

The "array mismatch errors" mentioned by the user do not occur - the script runs successfully. However, **population aging is completely broken**, causing all other calibration problems. Once population aging is fixed, the calibration should improve dramatically.

**Priority**: Fix Bug #3 (population aging) first, then reassess.
