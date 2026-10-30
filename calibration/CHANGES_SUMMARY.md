# Summary of Changes for Maternal Immunity Calibration

## Date: 2025-10-24

## Changes Made

### 1. Added Maternal Immunity to immunity.py

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py`

**Added parameters** (lines 60-62):
```python
maternal_immunity_efficacy=0.9,  # Maximum protection from maternal antibodies at birth (90%)
maternal_immunity_half_life=ss.days(90),  # Half-life of maternal immunity decay (90 days = 3 months)
```

**Added maternal immunity calculation** (lines 290-301):
- Applies exponential decay of maternal antibodies with age
- Formula: `protection = efficacy * exp(-ln(2) * age / half_life)`
- Only applies to naive agents (those with no prior infections)
- At birth: 90% protection (configurable)
- At 3 months (1 half-life): 45% protection
- At 6 months (2 half-lives): 22.5% protection
- By 12 months: ~6% protection remaining

**Why this helps**:
- Concentrates first infections in 6-12 month age group
- Matches observed data showing 56% of cases in <1 year olds
- Allows calibration to sites with different levels of maternal protection

---

### 2. Updated Calibration System

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/calibration/calibration.py`

**Added to known_pars** (line 105):
```python
self.known_pars = [..., 'maternal_immunity_efficacy', 'maternal_immunity_half_life']
```

**Added parameter translation** (lines 195-208):
```python
elif par == 'maternal_immunity_efficacy':
    # Set on RotaImmunityConnector
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            connector.pars.maternal_immunity_efficacy = val

elif par == 'maternal_immunity_half_life':
    # Set on RotaImmunityConnector (in days)
    for connector in sim.connectors.values():
        if type(connector).__name__ == 'RotaImmunityConnector':
            connector.pars.maternal_immunity_half_life = val
```

---

### 3. Updated Test Calibration Script

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_improved_calibration.py`

**Fixed debug flag** (line 72):
- Changed from `debug=True` (only ran 2 trials)
- To `debug=False` (runs all 20 trials in parallel)

**Expanded calibration parameters** (lines 51-57):
```python
calib_pars = sc.objdict(
    reporting_rate=[0.002, 0.0001, 0.01],           # 0.01-1% surveillance capture
    maternal_immunity_efficacy=[0.85, 0.0, 0.95],   # 0-95% protection (0% = disabled)
    maternal_immunity_half_life=[90, 1, 120],       # 1-120 days half-life
    rel_beta=[1.0, 0.5, 3.0],                       # 0.5-3x transmission
    reassortment_rate=[0.10, 0.05, 0.15]            # 5-15% reassortment
)
```

**Key features**:
- Maternal immunity can be completely disabled (efficacy=0% or half_life=1 day)
- Allows flexibility for different sites (e.g., sites with neonatal infections)
- Expanded reporting_rate range (0.01-1%) for better fit
- Expanded rel_beta range (0.5-3x) for better fit

---

## Parameters Summary for Immunity

### Cross-Protection Parameters (in immunity.py)
These control protection from different strain types:

1. **`homotypic_immunity_efficacy = 0.9`** (90%)
   - Protection from same G,P strain (e.g., G1P8 → G1P8)

2. **`partial_heterotypic_immunity_efficacy = 0.5`** (50%)
   - Protection from strains with shared G OR P (e.g., G1P8 → G1P4)

3. **`complete_heterotypic_immunity_efficacy = 0.3`** (30%)
   - Protection from completely different strains (e.g., G1P8 → G2P4)

4. **`naive_immunity_efficacy = 0.0`** (0%)
   - Baseline for never-infected agents (before maternal immunity applied)

### Maternal Immunity Parameters (NEW - in immunity.py)
These control passive immunity from mother:

5. **`maternal_immunity_efficacy`** (default 0.9 = 90%)
   - Maximum protection at birth
   - **Calibratable**: [0.0, 0.95] allows turning off or maximizing

6. **`maternal_immunity_half_life`** (default 90 days)
   - How fast maternal antibodies decay
   - **Calibratable**: [1, 120] days allows minimal to extended protection

### Waning Parameters (in immunity.py)

7. **`immunity_waning_delay`** (default 0 days)
   - Delay before acquired immunity starts to decay
   - Not currently being calibrated

---

## Why Maternal Immunity is Critical

### Problem Before:
- Model produced 24% of cases in <1 year (TARGET: 56%)
- Cases spread evenly across ages (infants through adults)
- `rel_beta` couldn't fix this (affects all ages uniformly)

### Solution With Maternal Immunity:
- Infants 0-6 months: HIGH protection (60-90%)
- Infants 6-12 months: DECLINING protection (20-60%)
- Result: First infections concentrated in 6-12 month olds
- Matches observed data showing most cases in <1 year

### Flexibility for Different Sites:
- **High maternal immunity sites** (e.g., high breastfeeding):
  - efficacy ~90%, half_life ~90-120 days
  - Few neonatal infections, peak at 6-12 months

- **Low maternal immunity sites** (e.g., HIV-endemic, low breastfeeding):
  - efficacy ~0-50%, half_life ~30-60 days
  - More neonatal infections, earlier peak age

---

## Next Steps

### Ready to Run:
```bash
cd /Users/aliciakraay/PycharmProjects/rotasim/calibration
python test_improved_calibration.py
```

### What Will Happen:
1. Optuna will run 20 trials in parallel
2. Each trial tests different combinations of 5 parameters
3. Optimization minimizes GOF (lower = better fit)
4. Results will show:
   - Best parameter values found
   - Before/after age distribution comparison
   - Overall incidence fit
   - Whether maternal immunity improves age distribution match

### Expected Outcome:
- Maternal immunity parameters should find non-zero optimal values
- Age distribution should shift toward younger ages
- More cases concentrated in <1 year group
- Better match to target data (56% in <1y)

---

## Documentation Created:

1. **IMMUNITY_PARAMETERS_SUMMARY.md**: Detailed explanation of all immunity parameters
2. **CALIBRATION_ISSUES_ANALYSIS.md**: Analysis of the 3 issues (trials, GOF, age distribution)
3. **CHANGES_SUMMARY.md**: This file - summary of all changes made
