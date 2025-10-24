# Dual-Objective GOF Implementation

## Date: 2025-10-24

## Problem Statement

The previous calibration approach had a fundamental issue: the goodness-of-fit (GOF) function minimized a single combined metric that mixed **incidence magnitude** with **age distribution shape**. This allowed the `reporting_rate` parameter to dominate optimization by scaling everything down to minimize errors, resulting in massive underestimation (-84% error) while achieving a "good" GOF score.

## Solution: Separate Objectives

We implemented a dual-objective GOF function that separately fits:
1. **Overall incidence magnitude** (per 100k) → fitted by `reporting_rate`
2. **Age distribution shape** (proportions) → fitted by `maternal_immunity` and `rel_beta`

This prevents any single parameter from dominating the optimization by scaling.

---

## Changes Made

### 1. Updated Data Format

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/calibration/process_incidence.py`

**Old approach**: Single Excel tab (`Matlab_ageincidence`) with age-specific incidence rates

**New approach**: Two separate Excel tabs:
- `Matlab_incidence`: Single value for overall incidence (20.4 per 100k)
- `Matlab_agedistribution`: Proportions by age (52%, 38%, 5%, 4%)

**Changes to `process_data()`**:
```python
def process_data(filename=None, incidence_sheet=None, age_dist_sheet=None):
    """
    Extract and process experimental data

    Returns two separate values:
    - overall_incidence: float - Total incidence per 100k (for fitting reporting_rate)
    - age_distribution: dataframe - Proportion of cases by age (for fitting age distribution shape)
    """
    # Read overall incidence data
    incidence_data = sc.dataframe.read_excel(filename, sheet_name='Matlab_incidence')
    overall_incidence = incidence_data['Cases per 100k'].iloc[0]  # Single value

    # Read age distribution data
    age_dist_data = sc.dataframe.read_excel(filename, sheet_name='Matlab_agedistribution')

    # Process age mapping and return as proportions
    age_mapping = {'[0, 1)': 0, '[1, 2)': 1, '[2, 5)': 2, '[5, 125)': 5}
    ages = age_dist_data['Age'].replace(age_mapping)
    age_distribution = sc.dataframe(dict(ages=ages, proportion=age_dist_data['Proportion']))

    return overall_incidence, age_distribution
```

**Changes to `process_model()`**:
```python
def process_model(dat=None, popsize=None, verbose=False):
    """
    Extract and process data from the model

    Returns:
        overall_incidence: float - overall incidence per 100k
        age_distribution: dataframe - proportions by age
    """
    # ... [existing processing code] ...

    # Calculate age distribution (proportions)
    total_incidence = df['inci'].sum()
    df['proportion'] = df['inci'] / total_incidence

    # Return both overall incidence and age distribution
    overall_incidence = df['inci'].mean()  # Average across age groups
    age_distribution = sc.dataframe(dict(ages=df['ages'], proportion=df['proportion']))

    return overall_incidence, age_distribution
```

---

### 2. Updated Calibration System

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/calibration/calibration.py`

**Modified `__init__()`** to handle dual data format:
```python
def __init__(self, sim, data, calib_pars, ...):
    # data should be a tuple of (overall_incidence, age_distribution)
    if isinstance(data, tuple):
        self.overall_incidence, self.age_distribution = data
    else:
        # Backwards compatibility
        self.overall_incidence = data['inci'].mean()
        total = data['inci'].sum()
        self.age_distribution = sc.dataframe(dict(ages=data['ages'], proportion=data['inci']/total))
```

**Rewrote `compute_fit()`** for dual objectives:
```python
def compute_fit(self, sim, full=False):
    """
    Compute goodness-of-fit for both overall incidence and age distribution

    Strategy:
    - reporting_rate fits overall incidence magnitude
    - maternal_immunity and rel_beta fit age distribution shape

    Returns combined GOF that weights both objectives
    """
    # Get simulation results
    sim_overall_incidence, sim_age_distribution = self.sim_to_df(sim)

    # Handle case where simulation died out
    if sim_overall_incidence is None or sim_age_distribution is None or len(sim_age_distribution) == 0:
        penalty = 1e6
        if full:
            return penalty, penalty, penalty  # total, incidence_gof, age_dist_gof
        else:
            return penalty

    # 1. Compute GOF for overall incidence (single value)
    target_incidence = self.overall_incidence
    incidence_gof = abs(sim_overall_incidence - target_incidence) / (target_incidence + 1e-9)

    # 2. Compute GOF for age distribution (proportions)
    target_proportions = self.age_distribution.proportion.values
    sim_proportions = sim_age_distribution.proportion.values

    # Handle mismatched shapes
    if len(sim_proportions) != len(target_proportions):
        if len(sim_proportions) < len(target_proportions):
            padding = np.zeros(len(target_proportions) - len(sim_proportions))
            sim_proportions = np.concatenate([sim_proportions, padding])
        else:
            sim_proportions = sim_proportions[:len(target_proportions)]

    # Use sum of absolute differences for proportions (they sum to 1)
    age_dist_gof = np.abs(sim_proportions - target_proportions).sum()

    # Combined GOF: weight both objectives equally
    total_gof = incidence_gof + age_dist_gof

    if full:
        return total_gof, incidence_gof, age_dist_gof
    else:
        return total_gof
```

**Updated `sim_to_df()`** to return tuple:
```python
@staticmethod
def sim_to_df(sim):
    """
    Convert the sim output to data format

    Returns:
        overall_incidence: float - overall incidence per 100k
        age_distribution: dataframe - proportions by age
    """
    # ... [extract data from sim.analyzers] ...

    # Process the data using the process_incidence module
    overall_incidence, age_distribution = process_incidence.process_model(df)

    # Apply reporting rate if specified
    # This scales the OVERALL incidence but doesn't affect age distribution shape
    if hasattr(sim, '_reporting_rate') and sim._reporting_rate is not None:
        reporting_rate = sim._reporting_rate
        overall_incidence = overall_incidence * reporting_rate

    return overall_incidence, age_distribution
```

**Updated `check_fit()`** to display all GOF components:
```python
def check_fit(self, verbose=True):
    """ Run before and after simulations to validate the fit """
    if verbose: print('Checking fit...')
    before_pars = self.calib_to_sim_pars()
    self.before_sim = self.run_sim(sim_pars=before_pars)
    self.after_sim  = self.run_sim(sim_pars=self.best_pars)

    # Get simulation results (returns tuples)
    self.before_overall_incidence, self.before_age_distribution = self.sim_to_df(self.before_sim)
    self.after_overall_incidence, self.after_age_distribution = self.sim_to_df(self.after_sim)

    # For backwards compatibility, store dataframes
    self.before_df = self.before_age_distribution
    self.after_df = self.after_age_distribution

    # Get full GOF breakdown (total_gof, incidence_gof, age_dist_gof)
    self.before_fit, self.before_incidence_gof, self.before_age_gof = self.compute_fit(self.before_sim, full=True)
    self.after_fit, self.after_incidence_gof, self.after_age_gof = self.compute_fit(self.after_sim, full=True)

    if verbose:
        print(f'\nFit with original pars:')
        print(f'  Total GOF:       {self.before_fit:n}')
        print(f'  Incidence GOF:   {self.before_incidence_gof:n}')
        print(f'  Age dist GOF:    {self.before_age_gof:n}')
        print(f'\nFit with best-fit pars:')
        print(f'  Total GOF:       {self.after_fit:n}')
        print(f'  Incidence GOF:   {self.after_incidence_gof:n}')
        print(f'  Age dist GOF:    {self.after_age_gof:n}')

        if self.after_fit <= self.before_fit:
            print('\n✓ Calibration improved fit')
        else:
            print('\n✗ Calibration did not improve fit')
    return self.before_fit, self.after_fit
```

---

### 3. Updated Test Script

**File**: `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_improved_calibration.py`

**Changed data loading**:
```python
# Get both overall incidence and age distribution from data
overall_incidence, age_distribution = process_incidence.process_data()
print("\nCalibration target data:")
print(f"Overall incidence: {overall_incidence:.1f} per 100k")
print("\nAge distribution (proportions):")
print(age_distribution)
```

**Changed calibration initialization**:
```python
calib = Calibration(
    sim=sim,
    data=(overall_incidence, age_distribution),  # Pass as tuple
    calib_pars=calib_pars,
    total_trials=20,
    debug=False,
)
```

**Updated output display**:
```python
print("\n" + "="*60)
print("Comparing Overall Incidence:")
print("="*60)
print(f"Target:  {overall_incidence:.1f} per 100k")
print(f"Before:  {calib.before_overall_incidence:.1f} per 100k")
print(f"After:   {calib.after_overall_incidence:.1f} per 100k")
err_before_inci = (calib.before_overall_incidence - overall_incidence) / overall_incidence * 100
err_after_inci = (calib.after_overall_incidence - overall_incidence) / overall_incidence * 100
print(f"\nError before: {err_before_inci:+.1f}%")
print(f"Error after:  {err_after_inci:+.1f}%")

print("\n" + "="*60)
print("Comparing Age Distribution (proportions):")
print("="*60)
print(f"\n{'Age':<10} {'Target':<15} {'Before':<15} {'After':<15} {'Error Before':<20} {'Error After':<20}")
print("-"*100)

for i in range(len(age_distribution)):
    if i < len(calib.after_age_distribution):
        age = age_distribution.ages.iloc[i]
        target_prop = age_distribution.proportion.iloc[i] * 100  # Convert to percentage
        before_prop = calib.before_age_distribution.proportion.iloc[i] * 100
        after_prop = calib.after_age_distribution.proportion.iloc[i] * 100

        err_before = before_prop - target_prop  # Absolute difference in percentage points
        err_after = after_prop - target_prop

        print(f"{age:<10} {target_prop:<15.1f}% {before_prop:<15.1f}% {after_prop:<15.1f}% {err_before:<20.1f}pp {err_after:<20.1f}pp")
```

---

## How the Dual-Objective GOF Works

### Objective 1: Overall Incidence (Magnitude)
- **Target**: 20.4 per 100k
- **Metric**: Fractional error = `|sim_incidence - 20.4| / 20.4`
- **Controlled by**: `reporting_rate` (surveillance capture rate)
- **Interpretation**:
  - 0.0 = perfect match
  - 1.0 = 100% error (e.g., 40.8 or 10.2 per 100k)

### Objective 2: Age Distribution (Shape)
- **Target**: [52%, 38%, 5%, 4%] for ages [0, 1, 2, 5]
- **Metric**: Sum of absolute differences = `sum(|sim_props - target_props|)`
- **Controlled by**: `maternal_immunity_efficacy`, `maternal_immunity_half_life`, `rel_beta`
- **Interpretation**:
  - 0.0 = perfect match
  - 1.0 = completely opposite distribution (e.g., [4%, 5%, 38%, 52%])
  - Typical values: 0.1-0.5 for reasonable fits

### Combined GOF
- **Formula**: `total_gof = incidence_gof + age_dist_gof`
- **Equal weighting**: Both objectives contribute equally
- **Typical good fit**: total_gof < 0.5 (incidence_gof < 0.2, age_dist_gof < 0.3)

---

## Why This Fixes the Previous Problem

### Previous Issue:
- Single GOF mixed magnitude and shape
- `reporting_rate` could scale everything down to minimize combined error
- Result: GOF = 32.4 (looked good) but incidence was -84% off target

### New Approach:
- Incidence GOF explicitly measures magnitude error
- Age distribution GOF explicitly measures shape error
- `reporting_rate` can only improve incidence_gof (doesn't affect age_dist_gof)
- `maternal_immunity` and `rel_beta` primarily affect age_dist_gof
- Optimization must balance both objectives simultaneously

### Expected Outcome:
- Incidence error should be within ±20% of target (20.4 per 100k)
- Age distribution should match target shape (52%, 38%, 5%, 4%)
- Parameters should find meaningful values rather than extreme scaling

---

## Calibration Parameters

```python
calib_pars = sc.objdict(
    reporting_rate=[0.002, 0.0001, 0.01],           # 0.01-1% surveillance capture
    maternal_immunity_efficacy=[0.85, 0.0, 0.95],   # 0-95% protection at birth
    maternal_immunity_half_life=[90, 1, 120],       # 1-120 days half-life
    rel_beta=[1.0, 0.5, 3.0],                       # 0.5-3x transmission
    reassortment_rate=[0.10, 0.05, 0.15]            # 5-15% reassortment
)
```

### Parameter Roles:
1. **reporting_rate**: Scales overall incidence → fits **incidence_gof**
2. **maternal_immunity_efficacy**: Protects young infants → shifts age distribution younger → fits **age_dist_gof**
3. **maternal_immunity_half_life**: Controls duration of maternal protection → affects age distribution shape → fits **age_dist_gof**
4. **rel_beta**: Overall transmission intensity → affects both GOF metrics
5. **reassortment_rate**: Strain diversity (minor effect on both metrics)

---

## Testing

**Command to run**:
```bash
cd /Users/aliciakraay/PycharmProjects/rotasim/calibration
python test_improved_calibration.py
```

**What happens**:
1. Loads target data from new Excel tabs
2. Runs 20 calibration trials in parallel (12 workers)
3. Each trial tests different parameter combinations
4. Optuna minimizes combined GOF (incidence_gof + age_dist_gof)
5. Reports best parameters found
6. Shows before/after comparison for both objectives

**Expected output**:
- Best parameters that balance both objectives
- Incidence error within ±20% of 20.4 per 100k
- Age distribution closer to target [52%, 38%, 5%, 4%]
- Separate GOF metrics showing improvement in both objectives

---

## Files Modified

1. `/Users/aliciakraay/PycharmProjects/rotasim/calibration/process_incidence.py`
   - Updated `process_data()` to return (overall_incidence, age_distribution)
   - Updated `process_model()` to return (overall_incidence, age_distribution)

2. `/Users/aliciakraay/PycharmProjects/rotasim/calibration/calibration.py`
   - Updated `__init__()` to handle tuple data format
   - Rewrote `compute_fit()` for dual objectives
   - Updated `sim_to_df()` to return tuple
   - Updated `check_fit()` to display all GOF components

3. `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_improved_calibration.py`
   - Updated data loading
   - Updated calibration initialization
   - Updated output display

---

## Next Steps

1. **Wait for calibration to complete** (~5-10 minutes)
2. **Evaluate results**:
   - Check if incidence error is within ±20%
   - Check if age distribution matches target shape
   - Verify both GOF components improved
3. **If results are not satisfactory**:
   - Adjust cross-protection parameters (homotypic, partial heterotypic, complete heterotypic)
   - Expand parameter ranges
   - Increase number of trials for better optimization
4. **If results are good**:
   - Document best parameters
   - Test on different calibration sites
   - Implement site-specific parameter sets
