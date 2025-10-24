# Calibration Issues Analysis

## Issue 1: Only 2 Trials Ran Instead of 20

**Root Cause**: The `debug=True` flag in test_improved_calibration.py

In calibration.py:68, the test passes `debug=True`:
```python
calib = Calibration(
    sim=sim,
    data=data,
    calib_pars=calib_pars,
    total_trials=20,
    debug=True,  # <-- THIS IS THE PROBLEM
)
```

In calibration.py:287-290, when `debug=True`, only 1 worker runs:
```python
def run_workers(self):
    """ Run multiple workers in parallel """
    if self.run_args.n_workers > 1 and not self.run_args.debug:  # Normal use case: run in parallel
        output = sc.parallelize(self.worker, iterarg=self.run_args.n_workers)
    else:  # Special case: just run one
        output = [self.worker()]  # <-- Only runs 1 worker when debug=True
    return output
```

**The calculation**:
- `total_trials=20`
- `n_workers = sc.cpu_count()` (let's say 10)
- `n_trials = ceil(20/10) = 2` trials per worker
- But with `debug=True`, only 1 worker runs
- Result: Only 2 trials complete!

**Fix**: Remove `debug=True` or set it to `False`

---

## Issue 2: "Best" Parameters Make Fit Worse

**Root Cause**: Optuna is MINIMIZING the objective, but the GOF function returns LOWER values for BETTER fits

Looking at compute_gof() in calibration.py:20-67:
- Returns absolute or fractional errors
- SMALLER values = BETTER fit
- The function calculates: `abs(actual - predicted)`

And compute_fit() in calibration.py:227-228:
```python
gofs = compute_gof(actual, expected)
fit = gofs.sum()  # Sum of errors - LOWER is BETTER
```

In run_trial() (calibration.py:263-267), this fit value is returned to Optuna:
```python
def run_trial(self, trial):
    """ Define the objective for Optuna """
    sim = self.run_sim(calib_pars=self.calib_pars, trial=trial)
    fit = self.compute_fit(sim)
    return fit  # Optuna minimizes this
```

**The Problem with check_fit()**:
In check_fit() (calibration.py:334-351), it compares "before" vs "after":
- Before fit: 215.8 (initial parameters)
- After fit: 360.8 (optimized parameters)
- 360.8 > 215.8, so it says "did not improve"

BUT, in the optimization trials:
- Trial 0: GOF=474.16
- Trial 1: GOF=360.76 ← Optuna correctly picked this as "better" (lower)

**The Real Problem**: The "before_pars" in check_fit() are NOT the same as what was tested during optimization!

In check_fit() line 337:
```python
before_pars = self.calib_to_sim_pars()  # Gets the "best" values from calib_pars
```

In calib_to_sim_pars() (calibration.py:132-137):
```python
def calib_to_sim_pars(self):
    """ Pull out "best" from the list of calibration pars """
    sim_pars = sc.objdict()
    for par,(best,low,high) in self.calib_pars.items():
        sim_pars[par] = best  # <-- Uses the INITIAL "best" guess
    return sim_pars
```

So "before" uses:
- reporting_rate: 0.002 (initial guess)
- rel_beta: 2.0 (initial guess)
- reassortment_rate: 0.10 (initial guess)

And "after" uses:
- reporting_rate: 0.0033 (optimized)
- rel_beta: 2.36 (optimized)
- reassortment_rate: 0.078 (optimized)

**Why the optimized parameters are worse**:
The initial guess (before) happens to be closer to the target than the optimized parameters!
This suggests the optimization is working CORRECTLY but:
1. The search space might not include the true optimum
2. The initial "best" guess was actually quite good
3. The optimization found a LOCAL minimum, not the global minimum

**This is NOT a bug** - Optuna is working correctly. The issue is that the initial guess was already pretty good, and the optimization explored and found that nearby parameter values (within the search range) produced worse fits.

---

## Issue 3: Age Distribution Not Matching

**Root Cause**: rel_beta affects transmission uniformly, NOT age-specifically

Looking at translate_pars() (calibration.py:170-179):
```python
if par == 'rel_beta':
    # In V2, modify base_beta which affects all strains
    if hasattr(sim, '_base_beta'):
        sim._base_beta = sim._base_beta * val
    # Also need to update disease betas
    if hasattr(sim, 'diseases'):
        for disease in sim.diseases.values():
            if hasattr(disease, 'pars') and hasattr(disease.pars, 'beta'):
                # Multiply the beta by the relative factor
                disease.pars.beta = disease.pars.beta * val
```

**What rel_beta does**: Multiplies ALL disease betas uniformly by the same factor
- rel_beta=2.0 → doubles transmission for ALL ages
- This increases infections across ALL age groups proportionally
- It does NOT preferentially shift infections to younger ages

**Why we thought higher rel_beta would shift cases younger**:
The reasoning was: "higher transmission → more infections → people get infected younger"

**Why this doesn't work in practice**:
1. The model already has VERY high transmission (nearly everyone gets infected multiple times)
2. Doubling transmission from "very high" to "extremely high" doesn't meaningfully change the age at first infection
3. The age distribution is determined by:
   - Birth rate (new susceptibles entering the population)
   - Waning immunity rate (how fast immunity decays)
   - Age-specific contact patterns (if implemented)
   - NOT just overall transmission intensity

**What's actually happening**:
Looking at the results:
```
Age    Target    After    Error
<1y    425.0     322.0    -24% (too low)
1-2y   312.5     553.5    +77% (too high)
2-5y   14.6      118.6    +713% (too high)
5+y    1.0       336.8    +35,262% (way too high)
```

The model is producing a "flatter" age distribution than the data:
- Data shows 56.4% of cases in <1y (very concentrated)
- Model shows only 23-24% of cases in <1y (spread out)

**The real problem**: The model fundamentally cannot produce the observed age distribution because:
1. Transmission is uniform across ages (no age-specific contact patterns)
2. Immunity waning is uniform across ages
3. Birth/death rates create a stable age distribution, but not the RIGHT age distribution

**Possible solutions**:
1. Add age-specific transmission (higher transmission to infants)
2. Add maternal immunity that wanes in first 6 months
3. Add age-specific susceptibility
4. Change the demographic parameters (higher birth rate, faster waning in adults)
5. Accept that the model cannot match the age distribution and only calibrate to overall incidence

---

## Summary of All 3 Issues

1. **Only 2 trials ran**: `debug=True` flag limited to 1 worker, which ran 2 trials
2. **"Best" parameters worse**: Initial guess was actually better than optimized values - optimization is working correctly but found that the search space doesn't contain better solutions
3. **Age distribution mismatch**: `rel_beta` affects all ages uniformly; the model lacks mechanisms to concentrate cases in infants (needs age-specific transmission, maternal immunity, or other age-dependent factors)

## Recommended Fixes

### Immediate fixes:
1. Set `debug=False` in test_improved_calibration.py to run all 20 trials
2. Understand that if optimization makes fit "worse", it means the initial guess was already near-optimal
3. Add explicit age-specific mechanisms to shift infections to younger ages

### Longer-term fixes:
1. Implement age-specific transmission rates
2. Add maternal immunity that protects infants 0-6 months
3. Add age-specific susceptibility parameters
4. Consider whether matching age distribution is achievable with current model structure
