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

1. **Censoring/dropout data — RESOLVED.** Source: `maled_data/first_infection_bangladesh.csv`
   (from `CoxDat`, event_observed = `!is.na(DateRota)`). 265 children, 44% censored.
   `censoring_ages` = `age_event_months` where `event_observed==0` (116 ages; median exit
   23.9 mo; filter one <=0 edge row). No R re-run needed.

2. **First-detection target — RESOLVED (re-derive as KM).** Same CSV gives
   `(age_event_months, event_observed)` -> KM age-at-first-DETECTION. NOTE: the current
   target (events-only quartiles Q25/med/Q75 = 5.13/7.98/11.24) drops the 44% censored and
   biases the median young; the cohort objective should use the censoring-aware KM curve
   (median sits higher). Symptomatic IR-by-age target: the existing
   `ir_by_age_symp_bangladesh.csv` is a per-age rate (cases/PT), comparable cohort vs
   cross-section, so reuse as-is (revisit only if detection-convention shifts it).

3. **Repeat-fraction definition + value — pending confirm.** Proposed (DK): among children
   with >=1 detected infection, fraction with >=2 detected -> Bangladesh ~**0.43**
   (58/136). Using 0.43 as a placeholder target; CONFIRM value + definition. (Not in the
   first_infection CSV -- needs a per-child detected-count extraction, or just the scalar.)

4. **Detection params (settled, confirm):** `symp_collection=0.80` (sensitivity-check 0.9),
   `eia_sensitivity=0.85` (EIA positivity), `shed_days=13`.

5. **Person-time frame.** Cohort PT here is full in-follow-up child-time. Confirm matches
   the data PT frame so `symp_collection` isn't double-counted (see DETECTION_MODEL_NOTES).

## Remaining build (turnkey once #3 confirmed)
- Wire `censoring_ages` from `first_infection_bangladesh.csv` (event_observed==0, >0).
- Re-derive KM first-detection target (manual KM; no lifelines dep).
- Add `repeat_detected_frac` target (0.43) + GOF term.
- `--observation cohort` path in `_run_one_replicate` (MALEDCohort, homogeneous mixing) +
  a `cohort` objective: Poisson IR + KM-first-detection + repeat-fraction. Smoke, launch.

## Not launching tonight
This needs the re-derived targets (#2) before the objective is meaningful, so it is built
but NOT run. Resume once #1-#3 are answered.
