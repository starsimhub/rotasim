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

## Deferred / backlog
- **Age-specific (under-5) mortality** — currently demographics use a uniform
  scalar death rate (Bangladesh 6/1000/y) and no emigration. If we add age-specific
  mortality later: Bangladesh **U5MR ≈ 44.4 per 1000 live births (2012)**. Deferred
  pending exp 03 (model verification) — early results suggest the demographic
  representation is not what is breaking the age-incidence fit, so this is low
  priority unless verification implicates it.
- **Downstream VE-gap work — age-of-infection targets + the age-symptom channel.**
  The eventual question is whether the LMIC age-distribution of infection produces a
  lower *achieved* VE at identical underlying per-infection efficacy. Empirical
  age-at-(severe, ~first)-infection targets for parameterizing/validating two settings
  in the ABM, from a pre-vaccine hospitalization review (PMC6736387):
  - LMIC / very-high child mortality: **median 38 weeks (~8.7 mo)**, IQR 25–58 wk.
  - High-income / low mortality: **median 65 weeks (~15.0 mo)**, IQR 40–107 wk
    (huge within-stratum spread: France 35 wk → Ukraine 101 wk; IQRs overlap).
  Back-of-envelope (NOT a substitute for running the ABM): applying the Lewnard et al.
  *J Infect Dis* natural-log-age coefficient for P(RVGE|infection) — **28%/log-month
  for primary infections** (the right one, since pre-vaccine hospitalizations are
  ~all first infections; β = ln(0.72) = −0.33) — the 38→65 wk shift is only a 1.71×
  ratio = 0.54 log-units, giving a **~16% lower** per-infection symptomatic risk in the
  older/high-income setting (~32% with the steeper 51%/log-month Vellore secondary
  coef). So the single age-symptom curve plausibly carries a substantial *fraction*
  (~⅓–⅔) of the observed ~50% high-vs-low-income VE gap, but not all of it — and
  "lower P(RVGE|infection)" ≠ "lower VE" (the channel must propagate through vaccine
  take, maternal-antibody timing, and force of infection; the ABM is what bridges it).
  Note our symptom model uses a *logistic* link with its own fitted betas, whereas
  Lewnard PLoS Comput Biol (pcbi.1007014) uses a *log* link with age & age² each
  z-scored to unit variance — so Lewnard's published betas are not drop-in seeds.
  Deferred until the pre-vaccine symptom-structure selection (exp 07–09) is settled.
