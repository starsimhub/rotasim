# Exp 13 — Cohort/detection emulation calibration (infection-number model)

**Question.** All prior experiments scored a steady-state cross-section with constant
detection. MAL-ED is a birth cohort followed 0-24mo with age-varying surveillance and
~44% censoring. Does calibrating the infection-number model under the *actual* MAL-ED
observation process — birth-cohort enrollment + dropout, age-varying detection
(monthly->quarterly), KM age-at-first-DETECTION, and the **repeat-infection-fraction**
constraint — give a coherent fit, and does the repeat-fraction constraint pin down the
transmission level (ruling out the hyperendemic regime)? Builds on the detection
decisions in `../../DETECTION_MODEL_NOTES.md` and D. Klein's cohort approach. See
`BUILD_PLAN.md` for the implementation record.

**Plan.** `calibrate_maled.py --site bangladesh --symptom-model infection_number
--observation cohort --fit-target cohort --maternal-model {erlang|titer}
--n-trials 40 --n-reps 20`, Bangladesh demographics, homogeneous mixing,
covaguest. Observation = `rs.MALEDCohort` (infection-number symptoms; symptomatic
detection `0.80*0.85` age-independent; asymptomatic = surveillance schedule * 0.85;
dropout from the data's per-child exit ages). Objective `cohort` =
`w_inc * Poisson-deviance(symptomatic IR)` + `w_first * GOF_first(KM first-DETECTION)` +
`w_repeat * GOF_repeat`, with GOF_repeat = z-score^2 vs the 0.403 target (binomial SE
0.040 -> a 1-SE miss = 1.0, so the repeat term is on a chi-square scale and bites: a
0.1 miss ~ 6, a 0.2 miss ~ 25). Default weights 1/1/1.

Targets (Bangladesh, cohort-consistent):
- symptomatic IR by age (existing `ir_by_age_symp_bangladesh.csv`),
- age-at-first-DETECTION KM: Q25/med/Q75 = **6.90 / 12.12 / 30.36 mo** (not the
  events-only 5.13/7.98/11.24 — KM keeps the 44% censored),
- repeat-detected fraction = **0.403** (among children with >=1 detected, fraction >=2).

**Success criteria.** A fit that matches the symptomatic-IR shape, a sensible KM
first-detection curve, AND the ~0.40 repeat fraction at a realistic (non-hyperendemic)
transmission level. The key test: does adding the repeat-fraction constraint force a
lower `base_beta` / pin the level where IR-shape-only fitting wandered into
over-infection? Run under both maternal models (erlang and titer) to see which the
cohort observation prefers.

**Status:** built + smoke-validated (see BUILD_PLAN.md), NOT yet launched. Open before
launch: (1) confirm the objective weights (default 1/1/1 — the se-normalized repeat term
is already well-scaled; tune `--w-first`/`--w-repeat` if first-detection or repeat needs
more pull); (2) the KM Q75 (30.36) is a heavy-censoring extrapolation — decide whether to
target full KM quartiles or just median + repeat-fraction.
