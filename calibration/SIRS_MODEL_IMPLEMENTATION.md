# SIRS Model Implementation Summary

## Changes Made

Successfully removed long-term immunity (LTI) mechanism and implemented simple SIRS (Susceptible-Infected-Recovered-Susceptible) model per user request.

### User's Requirement

> "Let's undo this long term immunity class. Instead, let's add a recovered class that all agents enter after infection (7 day infectious period) and stay for an average of 13 weeks before returning to being susceptible. Let's preserve infection counting."

## Files Modified

### 1. `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/rotavirus.py`

**Line 48-50:** Changed immunity waning duration from 7 days to 13 weeks (91 days)

```python
waning_rate_dist=ss.normal(
    loc=91, scale=14, unit="days"
),  # Duration of temporary immunity (13 weeks = 91 days mean, 2 weeks SD)
```

**Impact:** Agents now maintain temporary immunity for ~13 weeks after recovery before returning to fully susceptible state.

### 2. `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py`

**Line 323-324:** Disabled LTI susceptibility override

```python
# Override susceptibility for long-term immune agents (cannot be reinfected)
# DISABLED: Removed LTI mechanism per user request to implement simple SIRS model
# disease.rel_sus[self.long_term_immune[:]] = 0.0
```

**Lines 377-397:** Disabled probabilistic LTI assignment logic

```python
# DISABLED: Removed LTI mechanism per user request to implement simple SIRS model
# Probabilistically assign long-term immunity based on infection EPISODES (not individual strains)
# Only check for LTI when an infection episode completes
# if len(episode_complete_uids) > 0:
#     infection_counts = self.num_recovered_infections[episode_complete_uids]
#     ...
#     self.long_term_immune[newly_immune_uids] = True
```

**Impact:** Agents no longer develop permanent immunity. All immunity is temporary and wanes after ~13 weeks.

## Model Behavior After Changes

### Before (with LTI):
- Agents developed **permanent immunity** after 1-4 infection episodes
- 80-85% of population developed complete immunity (rel_sus = 0.0)
- Created herd immunity, disease went extinct after ~18 days
- Zero infections in years 2-10 of simulation

### After (SIRS model):
- Agents develop **temporary immunity** lasting ~13 weeks
- No permanent immunity
- Agents cycle: Susceptible → Infected (7 days) → Recovered (13 weeks) → Susceptible
- Disease should circulate endemically
- Infection counting preserved

## Components Preserved

1. **Infection counting:** `num_recovered_infections` continues to track infection episodes
2. **Episode counting:** Concurrent infections still count as one episode
3. **Cross-strain immunity:** Homotypic and heterotypic protection still functional during temporary immunity period
4. **State tracking:** `long_term_immune` and `long_term_immune_age` state arrays remain in code but are no longer assigned or used

## Testing

Created `/Users/aliciakraay/PycharmProjects/rotasim/calibration/test_sirs_model.py` to verify:

1. Disease circulates endemically (doesn't go extinct)
2. LTI count = 0 (no agents develop permanent immunity)
3. Agents cycle through S→I→R→S states
4. Infection counting still works
5. 13-week immunity period functions correctly

## Expected Outcome

With the simplified SIRS model:
- Disease should maintain endemic circulation over 10-year period
- Susceptible pool continuously replenished as immunity wanes
- R0 should remain > 1, preventing extinction
- Population achieves dynamic equilibrium with ongoing transmission

## References

- User request: Previous conversation summary
- Previous issue: `EPISODE_COUNTING_FIX_SUMMARY.md` (episode counting fix)
- Related files:
  - `rotavirus.py:138-166` - State transition logic (infected → recovered)
  - `immunity.py:329-397` - Recovery and immunity tracking
  - `immunity.py:299-324` - Susceptibility modulation (now only temporary immunity)
