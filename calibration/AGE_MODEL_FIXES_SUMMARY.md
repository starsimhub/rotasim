# Age Model Implementation - Fixes and Issues Summary

**Date:** November 25, 2025
**Session:** Debugging age-based symptom models

## Problems Identified

### 1. **Filter Order Bug** ✅ FIXED
**Issue:** Age-based models were applying filters in wrong order:
- WRONG: reporting_rate * severity filter → age-based symptom filter
- CORRECT: age-based symptom filter → reporting_rate * severity filter

**Impact:** Double filtering caused near-zero incidence (0.01 * 0.05 * 0.12 = 0.00006 = 0.006% pass rate)

**Files Fixed:**
- `process_incidence_uk_age.py` - Added `reporting_rate` parameter, applies reporting filter AFTER age filter
- `calibrate_uk_age_model.py` - Modified to pass `reporting_rate` to `process_model()` for age-based models only
- `evaluate_age_model_fit.py` - Updated to use correct filter order

**Code Changes:**
```python
# In process_incidence_uk_age.py (lines 228-236)
# Apply reporting filter AFTER age-based symptom filter
if reporting_rate is not None and 'severity' in initial5_symptomatic.columns:
    initial5_symptomatic['reported'] = np.random.random(len(initial5_symptomatic)) < (reporting_rate * initial5_symptomatic['severity'])
    initial5_reported = initial5_symptomatic[initial5_symptomatic['reported']].copy()
    initial5_symptomatic = initial5_reported
```

### 2. **Evaluation Code Parameter Bug** ✅ IDENTIFIED
**Issue:** `evaluate_age_model_fit.py` wasn't properly setting `base_beta` before simulation

**Root Cause:** Trying to set `disease.pars.beta` AFTER `sim.init()` is too late - transmission dynamics already computed

**Evidence:**
- Calibration results show working incidence: 2.79-5.03 per 100k
- Evaluation with manual sim creation showed: 0.00 per 100k
- All infections occurred at time 0.00-0.08 (initialization only), then zero transmission

### 3. **Calibration Framework Hanging** ⚠️ NEW ISSUE
**Issue:** Attempted fix using `UKAgeCalibration.run_sim()` method causes process to hang

**Symptoms:**
- Processes spawn but show 0.0% CPU usage
- No output produced after 15+ minutes
- Both full evaluation and simple tests hang

**Status:** UNRESOLVED - Need alternative approach

## What's Working

### Calibration ✅
The calibration framework IS working correctly:
- `calibrate_uk_age_model.py` produces valid results
- All three models (infection_number, age_and_infection, age_and_infection_simple) complete
- Incidence values: 2.79-5.03 per 100k (below target of 27.56, but non-zero)

### Model Implementation ✅
The three-model framework is correctly implemented:
1. **infection_number**: Infection-based immunity ON, infection-based severity ON
2. **age_and_infection**: Infection-based immunity ON, infection-based severity ON, age-based symptoms ON
3. **age_and_infection_simple**: Infection-based immunity ON, constant severity, age-based symptoms ON

### Analyzer ✅
The `InfectedStrainStats` analyzer correctly supports:
- `use_infection_based_severity=True` - Severity varies by infection number
- `use_infection_based_severity=False, constant_severity=0.05` - Constant severity

## Remaining Issues

### High Priority
1. **Evaluation code needs proper `base_beta` handling**
   - Can't use simple approach (sets too late)
   - Can't use calibration framework (hangs)
   - Need alternative method

2. **Low calibrated incidence**
   - Target: 27.56 per 100k
   - Achieved: 2.79-5.03 per 100k (10-18% of target)
   - May need wider parameter ranges or higher beta values

### Recommendations

**For Evaluation:**
Option A: Use calibration results directly without re-running
- Calibration already produces summary statistics
- Can extract MLE parameters and GOF from calibration output

Option B: Debug why calibration framework hangs when called from evaluation
- Investigate `run_sim()` method
- Check for multiprocessing/initialization conflicts

Option C: Find correct way to set `base_beta` before `sim.init()`
- Study how calibration.py's `translate_pars()` works
- May need to create sim copy with updated parameters

**For Calibration:**
- Increase `base_beta` range (current: [0.35, 0.55] for infection_number)
- Try wider ranges: [0.5, 2.0] to achieve target incidence
- Re-run calibration for all three models with adjusted ranges

## Files Modified

### Core Implementation
- `rotasim/rotasim/analyzers.py` - Added `use_infection_based_severity` parameter
- `calibration/process_incidence_uk_age.py` - Fixed filter order, added `reporting_rate` parameter
- `calibration/calibrate_uk_age_model.py` - Updated to pass `reporting_rate` correctly

### Evaluation (Multiple Attempts)
- `calibration/evaluate_age_model_fit.py` - Original, has base_beta bug (sets too late)
- `calibration/evaluate_age_model_fit_v2.py` - Uses calibration framework (hangs)
- Neither working - needs alternative approach

### Testing
- `calibration/test_transmission.py` - Diagnostic script to check transmission
- Shows zero infections with default setup (confirms evaluation bug)

## Next Steps

1. **Immediate:** Decide on evaluation approach (A, B, or C above)
2. **Short-term:** Adjust calibration parameter ranges and re-run
3. **Medium-term:** Validate model fits and compare across three models
4. **Long-term:** Use calibrated models for infection burden analysis (infections per child per year)

## Key Insights

- **Calibration works** - The issue was only with evaluation code
- **Filter order matters** - Age filter must come before reporting filter for age-based models
- **Parameter ranges need expansion** - Current ranges may be too conservative to hit target incidence
- **Framework complexity** - Direct use of calibration framework for evaluation causes issues
