# Exp 04 — MAL-ED Cohort Emulation + Kaplan-Meier Censoring

**Question.** Does reproducing MAL-ED's *actual study structure* — a birth cohort
followed 0–24 months under its real surveillance schedule, with age-at-first-
infection compared by Kaplan-Meier (properly handling censoring) — change the
fit picture versus the current steady-state, constant-detection, events-only
approach? Specifically: with faithful emulation, is the model still over-
infecting, and is the age-at-first-infection target reachable?

Motivated by exp 03 (IR shape reachable but the prior over-infects; first-
infection is the discriminating target but its censoring is unfixed) and by what
the MAL-ED design actually is.

**What MAL-ED is (from Mohan et al. 2017, MAL-ED rotavirus paper).**
- Enroll <=17 days of age; follow to 24 months.
- Asymptomatic surveillance stool: **monthly to 12mo, then 15/18/21/24mo** (quarterly).
- Diarrheal stool on every diarrheal episode (>=3 loose/24h), ~79% captured.
- Rotavirus detected by EIA (antigen).
- ~46% of infections symptomatic; only ~10% of children had a repeat detected infection.

**Four fixes over the current method.**
1. **Birth cohort, not cross-section.** Enroll agents at birth into a cohort and
   follow each from birth to 24 months of age. Person-time and events are cohort-
   based (proper accounting), replacing the steady-state snapshot.
2. **Discrete, age-varying surveillance detection.** Replace constant
   `p_asymp_detect=0.4` with the real schedule: an asymptomatic infection is
   detected only if a scheduled surveillance visit (monthly <=12mo, then
   15/18/21/24mo) falls within its ~13-day shedding window — so detection is
   ~0.4 under 12mo and ~0.14 after. Symptomatic infections detected at diarrhea
   onset (x ~0.79 capture x EIA sensitivity).
3. **24-month follow-up** (not 36).
4. **Kaplan-Meier** for age-at-first-detected-infection on BOTH model and data
   (data has `event_observed`; censor at 24mo / end of follow-up). Compare KM
   survival curves and KM-derived quartiles — apples to apples.

**Plan.** A new `MALEDCohort` analyzer: tags agents born after a burn-in as the
enrolled cohort, tracks per-child infection/shedding events to age 24mo, applies
the surveillance detection model, and records (age-at-first-detection,
event_observed) per child plus cohort person-time by age bin. Outputs KM survival
+ IR-by-age. Run a handful of parameter sets (incl. exp 03's best draw #314 and a
hyperendemic draw) on `capybara`; compare to MAL-ED Bangladesh KM + IR. Also report
the **repeat-infection fraction** as an independent check against the observed ~10%.

**Success criteria.**
- Good: under faithful emulation, a plausible-prevalence parameter set matches the
  MAL-ED KM curve (and ~10% repeat fraction) — the targets are jointly reachable and
  we can proceed to calibrate against the KM target.
- Informative failure: the model can match IR-by-age but not the KM curve (or forces
  >>10% repeats) → confirms the over-infection is the binding structural problem, and
  tells us which mechanism (FOI, immunity, detection) to open up.
- Watch: whether the monthly→quarterly detection drop alone reshapes the 12–24mo IR
  enough to change earlier conclusions.
