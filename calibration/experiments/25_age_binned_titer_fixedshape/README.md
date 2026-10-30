# Exp 25 — binned age-symptom + titer (fixed shape): is the quadratic over-constraining age?

**Question.** Exp 23 (age) uses a *quadratic* age-symptom logistic
(`P(symp) = logistic(beta0 + beta1·(a−12) + beta2·(a−12)²)`). That parabola forces a single
symmetric peak and may be **over-constraining** the age-symptom relationship — possibly part of
why age+titer struggled to fit (extinction, loose NROY) and why first-infection timing missed.
This experiment replaces the parabola with a **non-parametric** symptom model: a free
`P(symptomatic)` per age bin (**<6, 6–11, ≥12 mo**, 3 free params), nothing tying them to a
curve. We then ask: do the fitted bin probabilities *fall on* a quadratic, or does freeing them
fit better / identify more cleanly? Same **fixed-shape titer maternal** as
[exp 23](../23_age_titer_fixedshape/) / [exp 24](../24_infnum_titer_fixedshape/), so this is a
clean swap of *only* the age-symptom functional form vs exp 23.

**Note — runs on the corrected maternal.** This is the first titer experiment after **PR #38**
(maternal-titer persistence fix) was merged. Exp 23/24 ran on the buggy redraw-every-step
maternal; exp 25 has per-infant titer heterogeneity working, which on its own shifts
first-infection later (+~2.4 mo in a single-seed check) — so exp 25 is *not* directly comparable
to exp 23's numbers until exp 23 is re-run on the fix too.

**Plan.** Shared HM driver, only the symptom form differs:
```
PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python python hm_calibrate.py \
    --model age_binned --maternal titer --fix-titer-shape --all-targets --max-iter 10
```
40k agents, cohort observation, Bayes-linear emulators, CycleFeatureSelection over all 5 targets.
Fits transmission + sus-ladder + `maternal_efficacy` + the 3 bin probs
(`p_symp_age_0_6`, `p_symp_age_6_11`, `p_symp_age_12plus`); titer shape held at
`FIXED_TITER_SHAPE`. Then `trajectory_select.py --model age_binned --maternal titer
--fix-titer-shape` → posterior (overdispersed reweight; emulator-MCMC as backup). Optionally set
`early_stop_extinct` to ~3.5× the extinct draws and afford more samples/wave.

**Success criteria.** (1) NROY converges (contrast exp 23 age-quadratic's 0.190 and age+titer-free
0.257) with manageable extinction. (2) The fitted bin posterior tells us whether the quadratic
over-constrains: if the bins land on a smooth parabola, the quadratic was fine; if they don't
(e.g. flat-then-drop, or a non-symmetric shape) **and** the binned model fits the targets better
or identifies more cleanly, the parabola was the limiter. (3) If binned age finally hits
first-infection timing + IR-by-age together, it becomes a candidate age member for the exp-20 VE
comparison.

**Pipeline (built 2026-06-15, smoke-validated).** `symptom_model='age_binned'` added to
`MALEDCohort._symp_prob` (rotasim/analyzers.py); `SYMPTOM_BOUNDS['age_binned']` + `SYMPTOM_MODEL`
+ `untransform` + `--model age_binned` choices wired in hm_calibrate / trajectory_select /
mcmc_nroy / validate_mcmc; binned params threaded through `calibrate_maled._run_one_replicate`.
Smoke (bins 0.10/0.80/0.40) → symptomatic IR [0.26, 6.30, 2.55], confirming the bins drive the
per-age symptomatic incidence as intended.
