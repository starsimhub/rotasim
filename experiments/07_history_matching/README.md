# Exp 07 — History Matching (Wave 1) on MAL-ED Bangladesh

**Question.** Use the IDM `history_matching` package to find the Not-Ruled-Out-Yet
(NROY) region of the ~12-parameter space consistent with the MAL-ED Bangladesh
targets — replacing exp 06's crude 1-replicate log-SSE ranking with emulator-based
waves that account for observational *and* model (stochastic) uncertainty. Coverage
is established (exp 03/06: targets jointly reachable), so HM should narrow rather
than collapse.

**Prerequisite — replicate variance (and a winner's-curse check).** Before any wave,
run the exp-06 best region at ~10 seeds to (a) confirm the good fit reproduces (not a
1-draw fluke) and (b) measure the model's stochastic SD per observable. That SD is
the *model* component of each target's variance; it also sets the seed count for the
later trajectory-selection step.

**Targets (mean, std) — proper uncertainties.** Each observable registered separately
(HM layers consistency checks; no scalarised loss). Std = observational (sampling) ⊕
model (replicate) variance:
- symptomatic IR per bin: mean = data IR; observational SD ≈ √cases / PT × 100
  (Poisson on the case counts: <6m 27, 6-11m 74, 12-23m 59 cases). The 24-35m bin
  (1 case) is too noisy to constrain — exclude or down-weight.
- repeat-infection fraction (among ever-detected): mean ≈ 0.43 (Bangladesh
  58/136); SD from the binomial on 136.
- age-at-first-detection: a KM-derived summary (e.g., median + IQR) with its SD.

**Parameters (bounds only — no prior).** ~12, box-bounded:
- log-transform `base_beta`, `titer_median` (span orders of magnitude).
- **Monotone ladders need reparameterization** (HM uses axis-aligned boxes): replace
  `sus_after_1/2/3+` with `sus_after_1 ∈ (0,1)` + ratios `r2,r3 ∈ (0,1)`
  (`sus_2 = sus_1·r2`, `sus_3 = sus_2·r3`); same for `p_symp_1/2/3+`. This makes the
  monotone constraint a box.
- linear: `young_reservoir`, `adult_contacts`, `infant_exposure`, titer
  `gsd`/`half_life`/`hill_slope`, `maternal_efficacy`.

**Wave plan.** Bayes Linear emulator (fall back to GPR if R²<0.8). Emulate 1 feature/
wave, macro→fine:
- **Wave 1:** a macro, easy-to-emulate feature — overall symptomatic incidence
  (sum over infant bins) or `frac_ever_detected`. Implausibility threshold 4.0 (early,
  conservative), ~1500 samples.
- **Waves 2–3:** the per-bin IR shape / 6-11m peak ratio.
- **Later:** first-detection summary, repeat fraction.

**Sim function.** Reuse exp 06's cohort machinery (`MALEDCohort` + MixingPools + titer
maternal): `params_df → DataFrame` of observables (`ir_<6m`, `ir_6-11m`, `ir_12-23m`,
`repeat_frac`, `frac_ever_detected`, `first_inf_median`). Run on capybara/zebra.

**Success criteria.**
- Good: NROY narrows over waves to a stable, plausible region; emulator R²>0.8 per
  emulated feature; the exp-06 best region sits inside NROY.
- Watch: NROY collapse in wave 1 (threshold too tight or a target mis-specified —
  recheck coverage/target SDs); poor emulator fit (switch feature or GPR).
- Then (exp 08): the Bayesian step — trajectory selection on NROY for the posterior.
