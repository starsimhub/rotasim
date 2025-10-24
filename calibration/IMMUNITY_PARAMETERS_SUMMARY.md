# Rotavirus Immunity Parameters Summary

## Current Immunity Parameters (from immunity.py:54-59)

### Cross-Protection Parameters (ADJUSTABLE for calibration)

1. **`homotypic_immunity_efficacy = 0.9`** (90% protection)
   - Protection from reinfection with the SAME G,P strain
   - E.g., G1P8 → G1P8 reinfection
   - Current: 90% reduction in susceptibility
   - **Could calibrate**: Range [0.7, 0.99] to tune same-strain protection

2. **`partial_heterotypic_immunity_efficacy = 0.5`** (50% protection)
   - Protection from strains with SHARED G OR P (but not both)
   - E.g., G1P8 → G1P4 (shared G) or G1P8 → G2P8 (shared P)
   - Current: 50% reduction in susceptibility
   - **Could calibrate**: Range [0.3, 0.7] to tune partial cross-protection

3. **`complete_heterotypic_immunity_efficacy = 0.3`** (30% protection)
   - Protection from strains with DIFFERENT G AND P
   - E.g., G1P8 → G2P4 (no shared antigens)
   - Current: 30% reduction in susceptibility
   - **Could calibrate**: Range [0.1, 0.5] to tune complete cross-protection

4. **`naive_immunity_efficacy = 0.0`** (0% protection)
   - Baseline immunity for agents who have NEVER been infected
   - Current: 0% (fully susceptible)
   - **Could add maternal immunity here** (see below)

### Immunity Waning Parameters (ADJUSTABLE)

5. **`immunity_waning_delay = ss.days(0)`** (0 days)
   - Time delay before immunity starts to decay
   - Current: Immunity starts decaying immediately after recovery
   - **Could calibrate**: Range [0, 180] days to add delay before waning

### Other Parameters (Less relevant for age distribution)

6. **`cotransmission_prob = 0.02`** (2%)
   - Probability of co-transmitting multiple strains during transmission
   - Less relevant for age distribution calibration

---

## Maternal Immunity (NOT YET IMPLEMENTED)

**What it is**: Passive immunity passed from mother to infant through:
- Transplacental IgG antibodies
- Breast milk antibodies

**Typical duration**: 3-6 months, declining over time

**Impact on age distribution**:
- Protects infants 0-6 months from infection
- As maternal immunity wanes, infants become susceptible
- This creates a peak of first infections at 6-12 months
- **KEY**: This is why real data shows 56% of cases in <1 year

**Implementation needed**:
- Add `maternal_immunity_efficacy` parameter (e.g., 0.8-0.95)
- Add `maternal_immunity_duration` parameter (e.g., 180 days = 6 months)
- Modify susceptibility based on age:
  - Age 0-6 months: High protection (80-95%)
  - Age 6+ months: Protection wanes exponentially
  - By 12 months: Minimal maternal protection remains

---

## Which Parameters Impact Age Distribution?

### STRONG impact on age distribution:
1. **Maternal immunity** (NOT YET IMPLEMENTED)
   - Would shift infections to older infants (6-12 months)
   - This is THE KEY parameter for matching observed age distribution

2. **Demographics** (birth_rate, death_rate)
   - Higher birth rate → more young susceptibles → more cases in young
   - Current: birth_rate=70/1000/year, death_rate=20/1000/year

3. **Age-specific transmission** (NOT YET IMPLEMENTED)
   - Could make infants more susceptible or have more contacts
   - Would directly shift infections to younger ages

### MODERATE impact:
4. **Cross-protection parameters** (homotypic, partial_hetero, complete_hetero)
   - Higher cross-protection → infections spread out over more time/age
   - Lower cross-protection → rapid sequential infections in infants
   - BUT: Effect is uniform across ages unless combined with other mechanisms

5. **Immunity waning delay**
   - Longer delay → agents stay protected longer → slower accumulation of immunity
   - Indirect effect on age distribution through immunity buildup

### WEAK impact:
6. **rel_beta** (overall transmission intensity)
   - Multiplies all transmission uniformly
   - Does NOT preferentially affect any age group
   - Only affects SPEED of infection spread, not age distribution

---

## Recommended Calibration Strategy

### Phase 1: Add Maternal Immunity (Required for age distribution)
Implement maternal immunity with parameters:
- `maternal_immunity_efficacy = 0.9` (range: [0.7, 0.95])
- `maternal_immunity_half_life = 90 days` (range: [60, 120] days)
- Efficacy decays exponentially: `efficacy = maternal_immunity_efficacy * exp(-age / half_life)`

### Phase 2: Calibrate Multiple Parameters Jointly
Calibrate these parameters together:
1. **reporting_rate**: [0.0001, 0.01] - for overall incidence magnitude
2. **maternal_immunity_efficacy**: [0.7, 0.95] - to shift cases to 6-12 months
3. **maternal_immunity_half_life**: [60, 120] days - to control protection duration
4. **complete_heterotypic_immunity_efficacy**: [0.1, 0.5] - to tune cross-protection
5. **reassortment_rate**: [0.05, 0.15] - for strain diversity

### Why This Will Work:
- Maternal immunity will concentrate first infections in 6-12 month olds
- This matches the observed 56% of cases in <1 year
- Cross-protection parameters tune how quickly subsequent infections occur
- reporting_rate scales to match surveillance data magnitude

---

## Current vs Target Age Distribution

**Current model (no maternal immunity)**:
- Age <1y: 24% of cases (TARGET: 56%)
- Age 1-2y: 42% of cases (TARGET: 42%) ✓
- Age 2-5y: 9% of cases (TARGET: 2%)
- Age 5+y: 25% of cases (TARGET: 0.1%)

**Problem**: Cases too evenly distributed across ages

**Solution**: Maternal immunity will:
1. Protect infants 0-6 months (reduce cases in very young)
2. As protection wanes at 6-12 months, infections spike
3. This creates the observed concentration in <1 year age group
