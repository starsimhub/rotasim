# Age-Assortative Networks: Implementation and Findings

## Summary

Age-assortative contact networks were successfully implemented, but **do not solve the age distribution problem** when combined with moderate adult immunity (50%). Mathematical analysis shows why.

## What Was Fixed

### Bug in Original Implementation

**Problem**: `AgeAssortativeNet` extended `ss.Network` instead of `ss.DynamicNetwork`
- Networks extending `ss.Network` don't automatically regenerate contacts each timestep
- `add_pairs()` was only called during initialization (ti=None) when all agents had age=0
- Result: All agents were classified as "children" and contacts never updated with real ages

**Fix**: Changed to extend `ss.DynamicNetwork`
```python
class AgeAssortativeNet(ss.DynamicNetwork):  # Was: ss.Network
    def step(self):
        self.end_pairs()
        self.add_pairs()
        return
```

### Verification of Fixed Network

With `DynamicNetwork`, contacts are now correctly generated:

**Call #1** (initialization, ti=None): All ages=0 → 100% "children" (discarded)
**Call #2** (first timestep): Correct UK ages → 6.2% children, 93.8% adults

**Contact patterns verified** (assortativity=0.9):
- Child-child: 5.6% (expected: ~6% × 0.9 = 5.4%) ✓
- Adult-adult: 84.4% (expected: ~94% × 0.9 = 85%) ✓
- Cross-age: 10.0% (expected: ~10%) ✓

## Results: Age-Assortativity Alone Insufficient

### Test Results (5 years, 5000 agents, adult rel_sus=0.5)

| Assortativity | Child % | Adult % | Error (pp) | Total Infections |
|---------------|---------|---------|------------|------------------|
| 0.50          | 5.1%    | 94.9%   | 80.9       | 466,006          |
| 0.70          | 4.8%    | 95.2%   | 81.2       | 424,308          |
| 0.80          | 4.6%    | 95.4%   | 81.4       | 404,942          |
| 0.90          | 4.1%    | 95.9%   | 81.9       | 390,870          |
| 0.95          | 3.6%    | 96.4%   | 82.4       | 392,779          |

**Target**: 86% in children, 14% in adults

### Key Finding: Higher Assortativity Makes Things Worse!

Counter-intuitively, **higher assortativity reduces child infection percentage**. Why?

## Mathematical Explanation

### The "Susceptible Mass" Problem

With UK demographics (6% children, 94% adults) and adult protection (rel_sus=0.5):

```
Susceptible mass of children = 0.06 × 1.0 = 0.06
Susceptible mass of adults   = 0.94 × 0.5 = 0.47

Adults have 8x more susceptible mass than children!
```

### Why High Assortativity Backfires

**Low assortativity (0.5)**:
- Children mix 50% with other children, 50% with adults
- Adult infections seed back into child population
- Children maintained by "spillover" from large adult reservoir

**High assortativity (0.9)**:
- Children mix 90% with other children, 10% with adults
- Child population becomes isolated cluster
- Small child group (6%) burns through infections quickly
- Large adult population (94%) sustains independent transmission
- Result: MORE adult infections as percentage

### The Underlying Problem

Even with 50% protection, adults numerically dominate:
- 94% of population
- 7 contacts/day each
- 0.5 susceptibility
- = 6.6 "susceptible-contact units" per adult

Children:
- 6% of population
- 7 contacts/day each
- 1.0 susceptibility
- = 7 "susceptible-contact units" per child

**But**: There are 15x more adults than children (94%/6% = 15.7)

So total:
- Adult susceptible-contacts: 94 × 6.6 = 620
- Child susceptible-contacts: 6 × 7 = 42
- **Adults dominate 94% of transmission capacity**

## What Would Work

### Option 1: Much Higher Adult Immunity

To achieve 86% infections in children with UK demographics requires:

```python
# Target: 86% in children, 14% in adults
# Child population: 6%, Adult population: 94%

# For equal infection rates per capita:
child_rate / adult_rate = (86/6) / (14/94) = 14.3 / 0.15 = 96

# This means adults need 96x less susceptibility than children
adult_rel_sus_needed = 1.0 / 96 = 0.010 (99% protection)
```

**Adult protection needs to be ~99%, not 50%!**

### Option 2: Extreme Age-Assortativity

Alternatively, could use extreme assortativity (>0.99) combined with:
- Much higher child contact rates (e.g., children 20 contacts/day vs adults 7)
- This creates separate epidemics in each age group
- Not realistic given user constraint: "adults actually have more total contacts than children"

### Option 3: Calibrate Adult Immunity

Make `adult_baseline_immunity` a calibration parameter:

```python
calib_pars = sc.objdict(
    adult_baseline_immunity=[0.95, 0.90, 0.99],  # Best, low, high
    # Other parameters...
)
```

This would let the calibration find the adult protection level (probably 95-99%) needed to match the observed 86% of infections in children.

## Recommendations

1. **Add adult_baseline_immunity as calibration parameter**
   - Range: [0.90, 0.99] (90-99% protection)
   - Expected best fit: ~0.95 (95% protection)

2. **Keep age-assortative networks with moderate assortativity**
   - Use assortativity = 0.5-0.7 (realistic mixing)
   - Don't use extreme values (0.9+) as they isolate child population

3. **Update `initialize_adult_immunity()` function**
   - Accept `adult_baseline_immunity` parameter
   - Set `disease.rel_sus[adults] = 1 - adult_baseline_immunity`

4. **Rerun calibration**
   - Let Optuna find optimal adult protection level
   - Should match both incidence and age distribution

## Implementation Notes

The current adult immunity initialization (in `calibrate_uk.py:initialize_adult_immunity()`) sets:
```python
disease.rel_sus[adult_uids] = 1.0 - homotypic_protection
```

This should instead use a dedicated adult baseline immunity parameter:
```python
disease.rel_sus[adult_uids] = 1.0 - adult_baseline_immunity
```

Where `adult_baseline_immunity` represents cumulative protection from childhood infections (~95%), distinct from `homotypic_immunity_efficacy` which is the protection gained from a single infection (~50%).
