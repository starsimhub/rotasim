# Exp 04 — MAL-ED Cohort Emulation + Censoring — SUMMARY

**Question.** Does faithfully reproducing the MAL-ED study structure — a birth
cohort followed 0–24mo under the real surveillance regime, with individual data-
driven dropout — change the fit picture vs the steady-state / constant-detection /
events-only approach? Is the model still over-infecting, and are the targets
jointly reachable?

**Setup.** New `MALEDCohort` analyzer: enroll a birth cohort (sim years 5–7.5),
follow each child to their individually-drawn study-exit age (sampled from MAL-ED
Bangladesh's empirical censoring ages → matched dropout), apply surveillance
detection (symptomatic→diarrheal stool at capture×EIA; asymptomatic→monthly-<12mo /
quarterly-after schedule), record age-at-first-detection + cohort IR + repeat
fraction. 480 prior draws (same prior+seed as exp 03), 20k agents, on `capybara`
(435s, 0 fail). KM kept as a *descriptive* curve (matched censoring makes it
non-corrective — see methods note).

## Result

**Under faithful emulation, the model cannot jointly reproduce the MAL-ED data,
and the binding constraint is the repeat-infection fraction.**

- **The prior overwhelmingly over-infects.** Repeat-infection fraction: prior
  median **0.89** (5–95% [0.00, 0.93]) vs MAL-ED's **~0.10**. KM curves crash to
  zero (everyone detected in the first months) for nearly all draws; the data
  declines gradually to S(24)≈0.37 (63% detected by 24mo).
- **0/480 draws** match the repeat fraction (~10% ±10pp) AND the detection fraction
  (±15pp) together.
- **Best joint draw #212** (low β=0.076): detection fraction roughly right
  (model 0.56 vs data 0.63), but **misses the 6-11m peak** — IR_symp
  [2.45, 1.96, 3.15] (flat) vs data [1.91, **5.37**, 2.35] — and still has **2×
  too many repeats** (0.22 vs 0.10).

**Interpretation.** Exp 03's "IR shape is reachable" was real but **manufactured by
the age-symptom curve**, not backed by realistic infection dynamics. Once the
cohort structure and the repeat-infection constraint are imposed, the tension is
exposed: matching the 6-11m incidence peak requires enough infection that the model
produces far more reinfection than the observed ~10%. MAL-ED's pattern (sharp infant
peak + only 10% repeats) implies **strong, durable protection after first
infection** — which the current immunity ladder (`sus_after_1/2/3+`) does not
deliver. This is the concrete missing mechanism.

## Figures

![Cohort fit](figures/cohort_fit.png)

## Observations

- **The repeat-infection fraction is the discriminating target** the IR-shape
  coverage (exp 03) missed — exactly as predicted there. It should be an explicit
  calibration target going forward; it alone rules out the hyperendemic regime the
  prior is full of.
- Reconciles with and extends Alicia's exp 02 (structural tension) and exp 05/06
  (age-severity needed): even *with* a peaked age-symptom curve in the prior, the
  joint fit fails — so age-severity is necessary but not sufficient; durable
  acquired immunity is also required.
- The four structural fixes (cohort vs cross-section; monthly→quarterly detection;
  24mo follow-up; matched dropout) all now in one observation model. The
  monthly→quarterly detection drop and the dropout both reduce later-age detection,
  making the over-infection gap even starker than exp 02/03 implied.

## Methods note — is KM needed?

No (now). KM was needed when model and data had mismatched censoring. With dropout
simulated individually and drawn from the data's own censoring distribution, both
sides share the same (non-informative) censoring, so observed-to-observed comparison
is valid and KM is **descriptive only** — the survival curve is just the most
complete single summary (timing + cumulative fraction detected). Hand-rolled (no
dependency); `lifelines` was dropped.

## Next

- **Open up durable post-infection immunity** (stronger/longer protection after
  first infection, or a long-term-immune fraction) and re-test whether the model can
  then hit the 6-11m peak AND ~10% repeats together.
- **Add repeat-infection fraction (and the KM curve) as explicit calibration
  targets** alongside IR-by-age.
- Tighten the prior away from the hyperendemic region (`base_beta` upper bound).
- PINS to revisit: EIA sensitivity (0.85), shedding/detectable window (13d). These
  shift quantities but not the qualitative 0.89-vs-0.10 gap.
