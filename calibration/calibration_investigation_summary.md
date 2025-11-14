# UK Calibration Investigation Summary

## Issues Found and Fixed

### 1. ✅ Indentation Error in calibration.py:217
**Problem**: Missing code body in if statement  
**Fix**: Added `setattr(connector.pars, par, val)` and `break`

### 2. ✅ Prevalence Parameter Conversion in rotasim.py:315
**Problem**: `prevalence` not converted to `init_prev` format  
**Fix**: Added conversion handling both numeric and callable values

### 3. ✅ Missing adult_baseline_immunity Parameter in immunity.py
**Problem**: Parameter removed but code still referenced it  
**Fix**: Updated to use exponential saturation formula with `baseline_immunity_exponential_rate`

### 4. ✅ Matplotlib Fork Crashes in calibrate_uk.py:296-300
**Problem**: Plotting inside parallel workers caused macOS fork issues  
**Fix**: Commented out plotting code

### 5. ✅ Incorrect Immunity Calculation in immunity.py:481
**Problem**: Used `1.5 × age` giving 93-100% immunity for adults  
**Fix**: Changed to use `immunity_init_dist.rvs()` giving realistic 39-78% immunity

### 6. ✅ Reporting Rate Range Too Low
**Problem**: Range 0.01-0.05% was unrealistically low  
**Fix**: Increased to 0.5-10% range

### 7. ✅ Population Too Small for Low Incidence
**Problem**: 5,000 agents gave only 0.34 expected cases  
**Fix**: Increased to 50,000 agents (3.4 expected cases)

## Current Calibration Results (with all fixes)

**Metrics:**
- ✅ Calibration improves fit (GOF: 2.93 → 2.01)
- ✅ Incidence improved 42% (33.4 → 19.3 per 100k, target: 1.4)
- ✅ Age distribution GOF improved 31% (1.19 → 0.82)
- ⚠️ Incidence still 14× too high
- ⚠️ Adult cases 58% vs target 12%

**Age Distribution:**
| Age | Target | Actual | Status |
|-----|--------|--------|--------|
| <1y | 14% | 19% | ✓ Close |
| 1-2y | 28% | 10% | ✗ Low |
| 2-5y | 47% | 13% | ✗ Low |
| 5+y | 12% | 58% | ✗ Too high |

## Remaining Challenge

**Core Issue**: Too many adult cases (58% vs 12% target)

**Possible causes:**
1. Adult immunity initialization needs tuning (currently 5-15 infections)
2. `baseline_immunity_exponential_rate` may need adjustment (currently 0.1)
3. Age-specific transmission or susceptibility not modeled
4. Network structure may not reflect age mixing patterns

**Next steps to investigate:**
- Increase adult immunity exposure range (e.g., 10-20 infections)
- Increase `baseline_immunity_exponential_rate` to 0.15-0.2
- Consider age-assortative network (currently random mixing)
- Add maternal immunity protection for infants
