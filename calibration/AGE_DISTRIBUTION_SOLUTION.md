# UK Calibration - Age Distribution Fix Required

## Current Status

**What's Working:**
- ✅ Age-targeted seeding implemented (infections start in children)
- ✅ Calibration runs successfully (24/24 trials complete)
- ✅ Overall incidence improved significantly (273% → 101% error)

**What's NOT Working:**
- ❌ Age distribution still wrong: 95.3% in adults vs. 11.7% target
- ❌ Age GOF no improvement despite age-targeted seeding

## Why Age-Targeted Seeding Wasn't Enough

Test results show that even with proper seeding in children:
```
Initial seeding (CollectionTime=0.0):
  <1y: 10% ✓ (target 13.8%)
  1-2y: 20% ✓ (target 27.7%)
  2-5y: 40% ✓ (target 46.9%)
  5+y: 30% ✓ (target 11.6%)

Final infections after 5 years:
  <5y: 4.2% ❌ (target ~86%)
  5+y: 95.8% ❌ (target ~14%)
```

**The disease starts in children but rapidly spreads to adults.**

## Root Cause: No Age-Dependent Susceptibility

Current model assumptions:
- All agents have equal baseline susceptibility (rel_sus = 1.0)
- Only protection comes from acquired immunity (after infection)
- UK population: 94% adults, 6% children

Result:
- Random mixing → 94% of contacts are with adults
- Equal susceptibility → 94% of transmissions go to adults
- Children are too small a pool to sustain the epidemic

## The Solution: Age-Dependent Susceptibility

Rotavirus epidemiology shows adults are intrinsically less susceptible due to:
1. Repeated childhood exposures building lasting immunity
2. Physiological differences (gut maturation, microbiome)
3. Behavioral factors (hygiene, less hand-to-mouth contact)

### Proposed Implementation

Add age-dependent baseline susceptibility in `rotasim/immunity.py`:

```python
def apply_immunity(self):
    """Update relative susceptibility based on immunity"""

    # Get agent ages in years
    ages_years = self.sim.people.age.values / 365.25

    # Base age-dependent susceptibility (before acquired immunity)
    # Values based on rotavirus epidemiology literature
    base_sus = np.ones(len(self.sim.people))

    # Children <5 years: fully susceptible
    base_sus[ages_years < 5] = 1.0

    # Adults 5+ years: reduced susceptibility
    # (representing cumulative childhood exposures)
    base_sus[ages_years >= 5] = 0.05  # 95% reduction

    # ... rest of acquired immunity logic ...

    # Final susceptibility combines age and acquired immunity
    disease.rel_sus[:] = base_sus * (1 - acquired_immunity_protection)
```

### Calibratable Parameter

Make adult susceptibility a calibration parameter:

```python
calib_pars = sc.objdict(
    ...
    adult_base_susceptibility=[0.05, 0.01, 0.2],  # Best, low, high
)
```

This would let calibration find the right balance between:
- Too low: No adult infections at all
- Too high: Too many adult infections (current problem)
- Just right: Match observed ~86% in children, ~14% in adults

## Alternative: Age-Assortative Mixing

Another approach is age-assortative contact patterns:
- Children preferentially contact other children
- Adults preferentially contact other adults

This would require:
1. Multiple network layers (child-child, child-adult, adult-adult)
2. Age-specific contact rates
3. More complex than susceptibility approach

**Recommendation**: Start with age-dependent susceptibility (simpler, more direct).

## Expected Results After Fix

With adult_base_susceptibility = 0.05:
- Children <5: High attack rate (~50-80% over 5 years)
- Adults 5+: Low attack rate (~1-5% over 5 years)
- Overall age distribution: ~80-90% in children (matching target ~86%)

## Implementation Priority

**HIGH PRIORITY**: This is a fundamental model limitation that prevents calibration from succeeding.

Without this fix:
- ✅ Incidence can be calibrated (already improved)
- ❌ Age distribution cannot be calibrated (structurally impossible)

With this fix:
- ✅ Both incidence and age distribution can be calibrated
- ✅ Model will match real rotavirus epidemiology
