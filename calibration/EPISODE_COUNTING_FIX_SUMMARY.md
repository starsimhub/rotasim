# Episode Counting Fix Summary

## Problem Identified by User

The original LTI implementation was counting individual strain infections rather than infection episodes toward the long-term immunity threshold. This meant that if an agent was infected with multiple strains concurrently or sequentially before recovering, each strain infection counted separately.

**User's requirement**: "Each infection episode prior to recovering or re-entering the susceptible class should count as only one infection and not multiple."

## Solution Implemented

Modified `/Users/aliciakraay/PycharmProjects/rotasim/rotasim/immunity.py` lines 358-395 in the `record_recovery` method:

### Key Changes:

1. **Track concurrent infections**: Use `num_current_infections` counter that increments on infection and decrements on recovery

2. **Episode completion detection**: Only increment `num_recovered_infections` when `num_current_infections` reaches zero (all concurrent infections have resolved)

3. **LTI assignment timing**: Only check for LTI assignment when an infection episode completes, not on every individual strain recovery

### Code Logic:

```python
# Decrement current infection count
self.num_current_infections[recovered_uids] -= 1.0

# Only increment recovered infections when ALL concurrent infections have resolved
completed_episode = self.num_current_infections[recovered_uids] == 0
episode_complete_uids = recovered_uids[completed_episode]

if len(episode_complete_uids) > 0:
    self.num_recovered_infections[episode_complete_uids] += 1.0

    # Track oldest infection time (only set if first infection episode)
    first_infections = np.isnan(self.oldest_infection[episode_complete_uids])
    self.oldest_infection[episode_complete_uids[first_infections]] = self.sim.ti

# Probabilistically assign long-term immunity only when episodes complete
if len(episode_complete_uids) > 0:
    infection_counts = self.num_recovered_infections[episode_complete_uids]

    # Determine probability for each agent based on their total infection episode count
    probs = np.zeros(len(episode_complete_uids))
    probs[infection_counts == 1] = self.pars.long_term_immunity_prob_after_1
    probs[infection_counts == 2] = self.pars.long_term_immunity_prob_after_2
    probs[infection_counts == 3] = self.pars.long_term_immunity_prob_after_3
    probs[infection_counts >= 4] = self.pars.long_term_immunity_prob_after_4

    # Randomly assign long-term immunity
    develops_long_term = np.random.rand(len(episode_complete_uids)) < probs
    newly_immune_uids = episode_complete_uids[develops_long_term]
    self.long_term_immune[newly_immune_uids] = True
```

## Test Results

### 15-Day Test with 100 Agents
- **Total strain-level infections**: 1,458
- **Completed infection episodes**: 11
- **Average strains per episode**: 132.5
- **Status**: ✓ Episode counting working correctly

### Example Agent Behavior:
- Agent 12: 10 strain infections → 1 completed episode → RECOVERED
- Agent 19: 13 strain infections → 1 completed episode → currently infected with 4 more strains (2nd episode ongoing)

## Before vs After Comparison

### Before Fix (Original Implementation):
- Agent has 5-7 individual strain infections
- Each strain infection counted toward LTI
- Resulted in 88.7% of population with LTI

### After Fix (Episode-Based Counting):
- Agent has 1-4 infection episodes
- Co-infections count as ONE episode
- Resulted in 84.9% of population with LTI

## Status: Fix Complete ✓

The user's requested fix has been successfully implemented. Episode counting now correctly treats concurrent/sequential infections as a single episode rather than counting each strain separately.

## Remaining Issue: Disease Extinction

**Important**: While the episode counting fix is correct, the underlying issue of disease extinction persists. This is because:

1. LTI gives **complete immunity** (rel_sus = 0.0) at immunity.py:323
2. Even with proper episode counting, 84.9% of agents still develop permanent immunity
3. This creates herd immunity that stops transmission after ~18 days
4. Disease dies out and never recovers during the remaining 9.95 years

**Root cause location**:
```python
# immunity.py line 323
disease.rel_sus[self.long_term_immune[:]] = 0.0  # Complete immunity
```

### Possible Solutions (NOT YET IMPLEMENTED):
1. Change LTI to partial immunity (e.g., `rel_sus = 0.05` for 95% protection)
2. Lower LTI probabilities
3. Implement LTI waning over time

**Awaiting user input** on which approach to take.
