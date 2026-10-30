# Starsim v4 notes from the Rotasim port

What moving Rotasim to Starsim 4 showed about Starsim. Nothing here blocks Rotasim.

## What worked well

- The migration guide was accurate, and the migration script had nothing to change, since Rotasim's Starsim 3 names were still valid.
- A disease written as code needed few changes: `ss.TimeArr.due()` replaced the hand-written time comparisons, and `ss.FloatArr(default=ss.normal(0, 1))` removed a whole mechanism for drawing a value once per agent at birth. That mechanism was also a bug (issue 37).
- The port is about 2.5 times faster for one strain (5.7 s against 14.1 s, best of three, 20,000 agents, 3 years). The model code was also rewritten, so how much of that is due to Starsim 4 has not been measured.

## Proposals

### 1. Strains that are carried together, with protection worked out in code

Rotasim is the pattern the migration guide says to replace ("several strains written as several diseases and connectors"), but it can't use `ss.Strains`:

- A disease written as code can only have strains carried one at a time (`define_strains()` raises `ss.CapabilityError`). Rotavirus co-infection is central to the model, since it is what makes reassortment possible.
- With declared strains, `ss.immunity(due_to=<attribute>, rel_sus=matrix)` is the closest fit, but Rotasim's protection isn't a matrix times an attribute. It is the maximum of a per-person term (from the number of infection episodes) and per-strain waning terms, times an efficacy from the best match, then compared with maternal immunity, which depends on age.
- `dur_inf` is a lognormal total with a fixed final phase, so the symptomatic phase isn't a single transition.

**Suggestion:** let a disease written as code carry several strains with `strains=ss.Strains(...)`, with `strain_sus(k)` (which already exists for one strain at a time) as the override for protection. Rotasim could then be one `ss.Infection` with arrays by strain, in place of one disease per strain and a connector. Results by strain would come from `res.by('strain')`.

### 2. Reassortment as a Starsim concept

Reassortment makes a new strain out of parts of the strains a host carries. `move=` on a transition can change the strains people carry, but only between named strains and for everyone carrying them. If strains were given as a product of two dimensions (G and P), a reassortment could be stated once, and it would work under every solver.

**Suggestion:** `ss.Strains` by segments, e.g. `ss.Strains(G=[1, 2], P=[4, 8])`, with immunity from matching by segment (`same=dict(G=0.5, P=0.5)`), and a `reassort` transition. Influenza's HA and NA have the same structure, so this isn't specific to rotavirus. It would also avoid creating every combination up front, which grows as G times P.

### 3. Half a step in a short phase

The move from Starsim 3 raised prevalence by 10% with no change of parameters (0.224 to 0.247), because each case is now infectious for half a step longer, and Rotasim's symptomatic phase lasts only about 5 steps. The migration guide does mention this, in the row on `ss.SIR`, but it reads as a small timing change. This is the Starsim 4 behavior being correct, not a bug, but a calibrated model has to be refit.

**Suggestion:** give it a row of its own in the behavior table of `migrate-starsim3`, saying that a phase of n steps changes by about 1/(2n), so calibrated models with short phases will need refitting.

## Downstream testing

Rotasim now has a `v4` branch, which `v4_design/deferred.md` (T13) and `v4_design/targets.md` list as missing, once this work is pushed. To add it to `test_downstream.yml`:

```yaml
- {model: Rotasim, repo: starsimhub/rotasim, ref: v4, dir: rotasim, install: pip install -e . -r tests/requirements.txt}
```
