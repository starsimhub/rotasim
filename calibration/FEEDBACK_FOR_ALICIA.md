# Notes for Alicia — MAL-ED Calibration Review (Dan, updated 2026-06-05)

Merged your `dec_calibration_akraay` into a working branch and reviewed it
alongside a set of diagnostic experiments (top-level `experiments/`). Your branch
is scientifically ahead in several places; ours adds a couple of fixes that matter
for correctness and for being able to run at scale. This note focuses on (A) what
we added that you'll want, (B) reconciliation where our findings meet yours, and
(C) what's still open.

Where you're already ahead (no action needed — just noting so we don't duplicate):
age-structured contacts / MixingPools (your exp 04), Bangladesh age pyramid
instead of UK, the Erlang / two-phase maternal-immunity model, and the headline
exp 05/06 result that **age-based symptom severity is required** (infection-number-
only can't make the 6–11mo peak). Our coverage check independently agrees with
this — see (B).

---

## A. Two fixes from our side worth pulling into the calibration

### A1. Memory-bounded reporter: `rs.MALEDTargets` (replaces InfectedStrainStats + process_model)

`calibrate_maled.py` logs every infection event via `InfectedStrainStats` and
post-processes with `process_model`. The event log is O(infections), so at high
prevalence it balloons — and across many parallel workers it OOMs (we hit this
hard: 64 workers drove a 314 GB machine to the wall).

`rs.MALEDTargets` (now in `rotasim/analyzers.py`) folds the detection + age-binning
into the analyzer's `step()` and keeps only per-bin case counters, per-bin person-
time, and one first-detected age per agent → **O(agents), flat in prevalence and
duration.** It is **validated to reproduce `process_model` exactly** under
deterministic detection and within Poisson noise under stochastic detection
(`experiments/03_coverage_check/validate_analyzer.py`). Recommend wiring the
calibration workers to it; removes the memory ceiling entirely.

### A2. Person-time denominator fix: `rs.PersonTimeByAge`

`process_incidence_maled.compute_person_months_steady_state` uses
`final_headcount × window_months` — a single end-of-sim snapshot × window length.
Because the population grows (births > deaths), that overstates the time-averaged
headcount. `rs.PersonTimeByAge` accumulates `count × dt` over the window (exact
regardless of growth). Two correctness fixes came with it and apply to both
analyzers: window membership uses calendar time (relvec, matching infection
`CollectionTime`), and person-time counts only living agents.

---

## B. Reconciliation — our denominator finding vs your exp 03

Both are correct; they cover different bins. **My exp 01** measured the snapshot
denominator bias as **age-dependent**: ~0% in `<6m`/`6-11m`, growing to **~16% in
`24-35m`** (older cohorts were born when the population was smaller, so the end-of-
sim snapshot most overstates them). **Your exp 03** found the denominator makes
~no difference to the `<6m` overshoot — consistent, because the bias is ~0 in that
bin. Net: the denominator fix matters for the **age gradient / older bins** (and so
for the immunity ladder `sus_after_3plus`), not for the `<6m` story. Worth stating
both scopes so we don't ship two contradictory one-liners.

Also: our exp 03 prior-predictive coverage **agrees with your exp 05/06**. With a
peaked (quadratic) age-symptom curve in the prior, a single draw (#314) reproduces
the Bangladesh IR shape — peak at 6–11mo, right magnitudes — at a realistic ~3.6%
prevalence. So the shape is reachable via age severity; no contradiction with your
exp 02 "0/250" (that was the infection-number model).

---

## C. Still open — and a structural change that "changes the game"

### C1. Recreate the MAL-ED study structure (in progress, our exp 04)

This is the big one. The current target pipeline diverges from MAL-ED's actual
design in ways that bias exactly the things we're fitting:

- **Cross-section vs birth cohort.** We score a steady-state slice; MAL-ED follows
  children 0–24mo from birth. → emulate a birth cohort (enroll at birth, follow to
  24mo, cohort-based person-time).
- **Constant `p_asymp_detect=0.4` hides an age effect.** MAL-ED collects
  asymptomatic surveillance stool **monthly to 12mo, then only quarterly
  (15/18/21/24mo)**. With ~13-day shedding that's ~13/30 ≈ 0.43 under 12mo but
  ~13/91 ≈ **0.14 after** — detection drops ~3× at one year. The constant 0.4
  overestimates detection in the 12–24mo range, biasing the older-age IR and the
  first-infection tail. → schedule-based detection.
- **Follow-up is 24mo, not 36** (we censor at 36).
- **Censoring + dropout.** The first-infection target drops the 44% censored
  children (events-only quartiles), biasing the median young. And the data has real
  early dropout (36% of censored kids leave before 20mo; 21 in the first 6mo), so
  the censoring isn't purely administrative. → compare age-at-first-**detection** by
  Kaplan-Meier (handles dropout), or better, forward-model the dropout times so the
  observation process matches end-to-end.

`experiments/04_maled_cohort_emulation/` builds this. Early signal: the
**repeat-infection fraction** is a brutally discriminating check — MAL-ED saw
repeats in only ~10% of children, while over-infecting draws force >90%. That
single number rules out the hyperendemic regime our prior is full of.

### C2. The prior over-infects

Exp 03: median endemic prevalence 0.23, 57% of draws >0.2 — vs a few % in reality.
Worth tightening `base_beta`'s upper bound (and/or the immunity floor) so calibration
doesn't spend its budget in the hyperendemic region. The ~10% repeat-infection
constraint (C1) is the cheapest way to enforce this.
