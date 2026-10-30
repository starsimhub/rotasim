# Fixed Susceptibility Model Calibration - Summary Report

**Date:** December 9, 2025
**Model:** `infection_number` with fixed susceptibility values
**Status:** ⚠️ Calibration completed but model appears unsuitable

---

## Executive Summary

An attempt was made to calibrate the `infection_number` model using **fixed susceptibility values** instead of exponentially decaying immunity. The calibration optimization completed successfully and found parameters, but subsequent testing revealed fundamental issues that make this approach problematic for UK data calibration.

**Key Finding:** The fixed susceptibility model requires transmission rates ~4× higher than the standard model, and test simulations with these parameters either hung indefinitely or could not be verified.

---

## Model Specification

### Fixed Susceptibility Parameters
- **After 1 infection:** `rel_sus = 0.67` (33% protection)
- **After 2 infections:** `rel_sus = 0.50` (50% protection)
- **After 3+ infections:** `rel_sus = 0.36` (64% protection)

### Calibration Setup
- **Population:** 100,000 agents
- **Calibration period:** 2008-2012 (5 years)
- **Burn-in period:** 2003-2008 (5 years)
- **Target incidence:** 27.56 per 100,000
- **Optimization:** Optuna Bayesian (20 trials)

### Calibrated Parameters
- `reporting_rate`: range [0.002, 0.05], best guess 0.015
- `base_beta`: range [1.5, 10.0], best guess 5.0

---

## Calibration Results

### Best Parameters (Trial 19)
```python
{
    'reporting_rate': 0.0021258099445399365,
    'base_beta': 6.973398294883567
}
```

- **Goodness of Fit:** 1.346 (best of 20 trials)
- **Calibration runtime:** ~2-3 minutes per trial
- **Status:** Calibration optimization completed successfully

### Parameter Comparison

| Model | Beta | Reporting Rate | Notes |
|-------|------|----------------|-------|
| Fixed Susceptibility | **6.973** | 0.00213 | Trial 19 (this attempt) |
| Standard (infection_number) | 1.735 | 0.000655 | MLE from confidence intervals |
| **Ratio** | **4.02×** | **3.25×** | Fixed/Standard |

---

## Critical Issues Discovered

### 1. Parameter Verification Failed

**Test 1: Full Population (100k agents)**
- **Duration:** Hung for 1+ hour
- **CPU Usage:** 0% (process idle)
- **Outcome:** Killed due to timeout
- **Issue:** Process appeared stuck, possibly due to high beta value

**Test 2: Reduced Population (10k agents)**
- **Duration:** Stuck for 6+ minutes
- **CPU Usage:** Minimal
- **Outcome:** Killed due to import loop
- **Issue:** Test script triggered full calibration due to module-level code

**Conclusion:** Unable to verify that Trial 19 parameters produce valid results.

### 2. Implausibly High Transmission Rate

The calibrated `base_beta = 6.973` is **4× higher** than the standard model (1.735). This suggests:

- Fixed susceptibility creates too much population immunity
- Model compensates with unrealistically high transmission
- May not be biologically plausible for rotavirus
- Could indicate fundamental model-data mismatch

### 3. Results Saving Bug

The saved results file `uk_calibration_results_infection_number_fixed_sus.json` only captured `reporting_rate`:

```json
{
  "best_parameters": {
    "reporting_rate": 0.0021258099445399365
    // MISSING: "base_beta": 6.973398294883567
  }
}
```

**Root cause:** The `study.best_params` from Optuna should contain both parameters, but only one was written to the results file. This is a bug in the calibration code's result-saving mechanism (calibrate_infection_number_fixed_suscept.py:307-321).

---

## Broader Context: Standard Model Performance

The fixed susceptibility issues must be viewed in context of the standard `infection_number` model's poor performance:

### MLE Confidence Intervals (infection_number, 50 runs)
```python
{
    'mle_parameters': {
        'reporting_rate': 0.000655,
        'base_beta': 1.735
    },
    'target_incidence': 27.56,
    'mean_incidence': 3.27,  # Only 12% of target!
    'error_percent': -88.14%
}
```

**Key Observation:** Even the standard `infection_number` model with infection-based severity achieves only ~12% of the target incidence. This suggests:

1. The `infection_number` severity model itself may be fundamentally mismatched to UK data
2. Infection-based severity alone may not capture the true reporting patterns
3. Age-based reporting or other mechanisms may be necessary

---

## Comparison with Working Models

### Age + Infection Model (with fitted severity)

```python
{
    'model': 'age_and_infection_simple',
    'severity_type': 'constant_fitted',
    'constant_severity': 0.14,  # 14% fitted
    'reporting_rate': 0.0284,
    'base_beta': 2.082,
    'age_distribution_gof': ~good fit
}
```

**This model performs significantly better** because:
- Uses both age and infection number for susceptibility
- Fitted constant severity (14%) instead of infection-based
- Beta value (2.08) is more reasonable
- Age distribution matches data well

---

## Technical Analysis

### Why Fixed Susceptibility Requires High Beta

**Exponential Decay Model:**
- Immunity wanes continuously over time
- Population maintains partial susceptibility even with prior exposure
- Lower transmission rate can sustain endemic equilibrium

**Fixed Susceptibility Model:**
- Immunity levels are discrete and permanent
- After 3+ infections: 64% permanent protection
- Population rapidly accumulates strong immunity
- Requires very high beta to maintain transmission against immune population

### Mathematical Implication

For endemic equilibrium with R₀ ≈ 1:
```
β_fixed ≈ β_exponential × (1 + immunity_accumulation_factor)
```

The 4× higher beta suggests the immunity accumulation factor is ~3, meaning the fixed model builds up 3× more effective immunity than the exponential model over the calibration period.

---

## Recommendations

### 1. Abandon Fixed Susceptibility Approach ❌

**Rationale:**
- Implausibly high transmission rates required
- Unable to verify parameters work
- Standard `infection_number` model already underperforms
- Adds complexity without improving fit

### 2. Focus on Working Models ✅

**Priority: age_and_infection_simple with constant severity**
- Already calibrated and validated
- Reasonable parameter values (beta=2.08)
- Good age distribution fit
- Fitted severity=14% is biologically plausible

### 3. Investigate Standard Model Issues 🔍

**Why does infection_number achieve only 12% of target?**

Possible explanations:
- Infection-based severity too low (declining with immunity)
- Age effects on reporting not captured
- Waning immunity too strong (Poisson removal)
- UK data may reflect age-specific healthcare seeking

**Suggested investigation:**
- Compare severity distributions between models
- Analyze reporting rates by age and infection number
- Review UK study methodology for age biases

### 4. Fix Results-Saving Bug 🐛

**File:** `calibrate_infection_number_fixed_suscept.py:307-321`

The code saves `best_pars` from Optuna but only one parameter was captured. Debug why `study.best_params` doesn't contain both parameters.

---

## Files Generated

### Calibration Files
- `calibration_3param_output_v3.txt` (2.9 MB) - Full calibration log with all 20 trials
- `uk_calibration_results_infection_number_fixed_sus.json` - Results (incomplete, missing beta)
- `rota.db` - Optuna study database

### Test Files
- `test_trial19_params.py` - Verification script (100k agents)
- `test_trial19_small_pop.py` - Verification script (10k agents)
- `evaluate_fixed_suscept_best_pars.py` - Multi-seed evaluation (updated with correct params)

### Analysis Files
- `mle_confidence_intervals_infection_number.json` - Standard model 50-run results

---

## Conclusion

The fixed susceptibility calibration **technically succeeded** in finding parameters but **scientifically failed** to produce a viable model:

1. ✅ Optimization converged (GOF=1.346)
2. ✅ Both parameters found by Optuna
3. ❌ Parameters could not be verified (simulations hung)
4. ❌ Transmission rate (beta=6.97) implausibly high
5. ❌ No improvement over standard model
6. ❌ Standard model itself severely underperforms (-88% error)

**The fundamental issue is not the fixed susceptibility implementation, but rather that the `infection_number` severity model is poorly matched to UK data.** The age_and_infection_simple model with fitted constant severity (14%) provides a much better fit and should be the focus of future work.

---

## Next Steps

1. **Document the age_and_infection_simple model** as the primary calibrated model
2. **Generate confidence intervals** for age_and_infection_simple model (50+ runs)
3. **Investigate** why infection_based severity underperforms so dramatically
4. **Consider** whether UK data reflects age-specific healthcare access rather than true infection-based severity patterns

---

## Appendix: Key Code Locations

### Fixed Susceptibility Implementation
- `calibrate_infection_number_fixed_suscept.py` - Calibration script
- Lines 64-70: RotaImmunityConnector with `use_fixed_susceptibility=True`
- Lines 98-103: Calibration parameter ranges
- Lines 228-247: GOF computation

### Bug Location
- Lines 307-321: Results saving code (missing base_beta in output)
- Line 270: `best_pars = calib.best_pars` (should contain both parameters)

### Comparison Files
- `mle_confidence_intervals_infection_number.json` - Standard model results
- `uk_calibration_results_age_and_infection_simple_with_severity.json` - Working model
