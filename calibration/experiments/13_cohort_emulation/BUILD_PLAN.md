# Exp 13 — Cohort/detection emulation: BUILD PLAN (in progress)

Adopt D. Klein's MAL-ED cohort observation in place of the steady-state `process_model`
path, with the TAC/EIA detection decisions from `../../DETECTION_MODEL_NOTES.md`, and add
the repeat-infection-fraction target. This file tracks what's built and what still needs
input — it is NOT a SUMMARY (no results yet).

## Why
The current pipeline scores a steady-state cross-section with constant detection. MAL-ED
is a birth cohort followed 0-24mo with age-varying surveillance and dropout. Matching the
observation process removes biases in the older-age IR, the age-at-first-detection tail,
and lets us add the **repeat-infection fraction** — the constraint that rules out the
hyperendemic regime Optuna wanders into.

## Status

DONE (committed):
- [x] `rs.MALEDCohort` analyzer ported into `rotasim/analyzers.py` — read-only observer
      (mixing-agnostic), infection-number symptoms, age-varying surveillance detection,
      KM age-at-first-detection, repeat-detected fraction. Detection structure matches
      DETECTION_MODEL_NOTES: symptomatic = `symp_collection(0.80) * eia_sensitivity(0.85)`
      (age-independent); asymptomatic = `p_surv(age) * eia` (monthly<12mo -> quarterly);
      no separate TAC-completeness factor (cancels in the TAC frame).

TODO (needs decisions / data — see Open questions):
- [ ] Provide the data's empirical per-child **censoring/exit-age distribution** -> feed
      `censoring_ages`.
- [ ] **Re-derive the targets consistently with this observation** (the current targets
      were built for the steady-state path): symptomatic IR by age, age-at-first-
      DETECTION (KM), and the repeat-detected fraction.
- [ ] Add `repeat_detected_frac` (and optionally `frac_ever_detected`) as targets in
      `process_incidence_maled.load_targets` + a GOF term.
- [ ] New objective `--fit-target cohort` (or extend `poisson`) combining Poisson IR +
      first-detection + repeat-fraction from `MALEDCohort.results_dict()`.
- [ ] Wire `_run_one_replicate` to use `MALEDCohort` when `--observation cohort`
      (replaces InfectedStrainStats + process_model + PersonTimeByAge), homogeneous mixing.
- [ ] Smoke-validate; confirm `MALEDCohort` reproduces `process_model` IR within Poisson
      noise under matched detection (sanity), then run.

## Open questions for Alicia (the "other steps")

1. **Censoring/dropout data.** Where is the MAL-ED Bangladesh per-child study-exit age
   distribution (months)? (CSV / RData object?) Needed for `censoring_ages`. Until then
   I'll placeholder with a 24-mo administrative exit + a nominal dropout, clearly flagged.

2. **Re-derived targets — the big one.** The cohort observation changes what "a case"
   and "first infection" mean (detected, KM-censored, among-followed cohort). Do you have
   cohort-consistent targets, or do we derive them from the data? Specifically:
   - symptomatic IR by age among the followed cohort,
   - age-at-first-**detection** (KM survival, handles dropout) — vs the current
     events-only first-infection quartiles,
   - repeat-detected fraction.

3. **Repeat-fraction definition + value.** Proposed (DK): among children with >=1 detected
   infection, the fraction with >=2 detected -> Bangladesh ~**0.43** (58/136). Confirm the
   definition and the value for your TAC dataset.

4. **Detection params (mostly settled, confirm):** `symp_collection=0.80` (diarrheal-stool
   collection completeness; sensitivity-check 0.9), `eia_sensitivity=0.85` (EIA positivity;
   set ~1.0 only if you switch the target to TAC positivity), `shed_days=13`.

5. **Person-time frame.** Cohort PT here is full in-follow-up child-time. Confirm that
   matches how your data PT is constructed (so the `symp_collection` deflation is applied
   once, not double-counted — see DETECTION_MODEL_NOTES).

## Not launching tonight
This needs the re-derived targets (#2) before the objective is meaningful, so it is built
but NOT run. Resume once #1-#3 are answered.
