# Calibration — MAL-ED rotavirus (pre-vaccine immunity-structure selection)

Calibration work for RotaABM/rotasim lives here. The model is in the parent repo.

## Intake

**Model.** RotaABM/rotasim — agent-based, Starsim/Python (parent repo
starsimhub/rotasim, branch dec_calibration_akraay). ~3.8 min/sim at 100k
agents × 10 yr. Calibrated with custom Optuna (spawn-pool, point fit, no
uncertainty) on the covaguest VM. Single-strain for current work.

**Question.** Immediate: get a good PRE-VACCINE fit to MAL-ED age-stratified
incidence and do model selection on the immunity/symptom structure — whether
an age term is needed beyond the canonical infection-number-only structure.
Current structure: age modulates SYMPTOMS (P(symptomatic|infected));
infection/susceptibility is infection-number (sus_after_*) + maternal immunity
(the only age-effect on infection). Downstream, OUT of current scope: add
vaccination → achieved-VE-by-age (does LMIC age-distribution of infection give
different achieved VE at identical underlying efficacy); ultimately strain
persistence × genetic diversity × achievable VE. VE/strain deferred until the
pre-vaccine fit is solid.

**Data.** MAL-ED Bangladesh birth cohort (chosen: NO vaccine rollout → clean
natural baseline). Two joint targets currently: symptomatic incidence in 4 age
bins, and age-at-first-DETECTED-infection quartiles. Detection model: symptomatic
infections assumed ~100% captured; asymptomatic infections detected only via
monthly stool (partial — `p_asymp_detect`). Source RotaDat.RData → per-site CSVs.
Counts sparse (1 case in 24–35 mo). All-infection incidence by age (IRdat_new)
also available but would require a reporting-rate parameter, so not currently used.
Vaccinated MAL-ED sites available later as a contrast.

**Constraints.** covaguest Azure spot VM (120 cores/448 GB/no swap, eviction-
prone; plain-python launches). Solo (Alicia Kraay). No hard deadline. Env:
conda (uv declined). Point fits now; posterior/uncertainty likely needed
downstream for the VE claim (deferred).

**Core open problem.** The two joint targets — symptomatic-IR-by-age and
age-at-first-infection — cannot be fit simultaneously (a Pareto tension):
fits good on one are poor on the other. Symptom-model variants tried
(age-logistic; infection-number per-infection probs; age+infection linear and
categorical-offset combined) and maternal-waning variants (single/two-phase,
Erlang) have not resolved it. This looks structural, not under-identification.

**Suggestions surfaced at intake (for calibration-workflow to sequence):**
(1) prior-predictive coverage check on the MAL-ED age curve — does the model
produce BOTH targets at once under ANY parameters? (distinguishes structural
mismatch from search failure). (2) count-based likelihood (Poisson/Beta-Binomial)
for the sparse age bins instead of squared-log-IR + LOG_EPS. (3) all-infection
IR-by-age remains an option if more reinfection-by-age constraint is needed, but
weigh its reporting-rate cost against the age-at-first-infection target it largely
duplicates.
