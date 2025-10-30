# UK Calibration - Root Cause Identified

## Problem Summary

The UK calibration shows **95.8% of infections in adults (≥5 years)**, when rotavirus epidemiology shows it should primarily affect children <5 years.

## Root Cause: Age-Blind Initial Seeding

### Current Implementation

```python
init_prev=ss.bernoulli(p=0.002)  # Randomly infects 0.2% of ALL agents
```

This seeds infections UNIFORMLY across all ages:
- UK population: 93.8% adults, 6.2% children
- Initial infections: 93.8% adults, 6.2% children (proportional to population)

### Why This Causes Adult-Dominant Infections

1. **Initial immunity**: Adults get 93.8% of initial infections → build immunity first
2. **Transmission dynamics**: Adults (94% of population) mostly contact other adults
3. **Endemic equilibrium**: Model reaches state where adults have high immunity, occasional infections
4. **Children underrepresented**: Only 6% of population, so even with high susceptibility, they're a small pool

### Test Results Confirming This

From `test_infection_ages.py`:
```
Population initialized:
  <1 year: 1.3%
  1-2 years: 1.3%
  2-5 years: 3.7%
  >=5 years: 93.8%

Infections recorded:
  0-2 months: 0.0%
  2-12 months: 0.2%
  12-24 months: 0.7%
  24-60 months: 3.4%
  60+ months: 95.8%  ← PROBLEM
```

## Epidemiologically Correct Behavior

For rotavirus:
1. **Initial seeding should target children** - rotavirus is endemic in pediatric populations
2. **Age-specific susceptibility** - children should be more susceptible than adults
3. **Maternal immunity** - infants <6 months should have some protection
4. **Natural boosting** - adults get exposed but usually asymptomatic

## Potential Solutions

### Option 1: Age-Dependent Initial Seeding
Modify initial prevalence to preferentially seed in children:
```python
# Instead of uniform bernoulli, use age-dependent initialization
# Seed 80% of infections in children <5, 20% in adults
```

### Option 2: Age-Dependent Susceptibility
Add `rel_sus` based on age:
```python
# Children <5: rel_sus = 1.0 (fully susceptible)
# Adults 5+: rel_sus = 0.1 (90% reduced susceptibility)
```

### Option 3: Both
Combine age-dependent seeding AND susceptibility for realistic epidemiology.

### Option 4: Burn-in from Pediatric-Only Population
Initialize with mostly children, let reach endemic equilibrium, then add adults.

## Questions for User

1. **Is age-dependent susceptibility already in the model?**
   - Should check immunity connector for age-specific rel_sus

2. **How should initial infections be seeded?**
   - Random uniform (current - WRONG for rotavirus)
   - Age-targeted (children only)
   - Age-weighted (higher probability in children)

3. **Should we modify starsim's Infection class?**
   - Or handle this in custom initialization?

## Impact on Calibration

The current calibration is trying to fit:
- Target: 86% of cases in children <5 years
- Model: 95.8% of cases in adults ≥5 years

**No amount of parameter tuning can fix this** - it's a structural model issue.

## Next Steps

1. Decide on age-dependent seeding strategy
2. Implement age-specific susceptibility if not already present
3. Retest to verify children now get majority of infections
4. Re-run calibration with corrected model

## Files Affected

- `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/rotavirus.py` - Initial prevalence seeding
- `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py` - Age-specific susceptibility
- `/Users/aliciakraay/PycharmProjects/rotasim/calibration/calibrate_uk.py` - Initialization logic
