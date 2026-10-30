# Exp 07 — Calibrate the peaked age-symptom model to MAL-ED (quantitative fit)

**Question.** Exp 05 showed *qualitatively* that a peaked (quadratic) age-symptom
curve — mild `<6m`, worst at 6–11 mo, milder after — plus strong Erlang maternal
immunity under homogeneous mixing reproduces the MAL-ED Bangladesh age-incidence
shape, and exp 06 showed the infection-number-only model cannot. The researcher now
wants the *quantitative* fit: calibrate this model to the actual targets and report
GOF and fitted parameters, so it can be compared head-to-head with the calibrated
infection-number model (exp 08). See `../05_nonlinear_age_symptom/SUMMARY.md` and
`../06_infection_number_compare/SUMMARY.md`.

**Plan.** Custom Optuna/TPE point calibration via `calibrate_maled.py`
(`--symptom-model age_only --maternal-n-stages 6 --fit-target joint`), Bangladesh
demographics (birth 19 / death 6, pyramid `[2.5, 2.5, 7.5, 87.5]%`), homogeneous
mixing (`RandomNet`, n_contacts=7), on the covaguest VM at 100k agents. Joint
objective = squared-log IR over the 4 age bins + normalized first-infection-quartile
difference. Search ranges are deliberately **wide and unbiased** (base_beta 0.05–0.5
log; monotone susceptibility; maternal efficacy 0.5–0.99, mean duration 30–300 d;
age betas beta0 ∈ [−5,2], beta1 ∈ [−1,1], beta2 ∈ [−0.5,0.5]). To keep TPE from
sliding into exp 02's monotone-declining basin, **enqueue one peaked seed**
(sharp_peak_9mo betas `[-1.3, -0.06, -0.01]`, base_beta 0.2, sus 0.6/0.4/0.25,
maternal eff 0.95 / mean 200 d) — a seed, not a constraint, so the optimizer is free
to leave it. 40 trials × 20 replicates, n_jobs=1.

**Success criteria.** A calibrated fit that reproduces the shape (low `<6m`, peak
6–11 mo, decline through 24–35 mo) AND a reasonable age-at-first-infection, with a
good joint GOF and biologically plausible fitted parameters (esp. maternal duration
vs. the ~3–7 mo Pitzer/Lopman literature). A poor fit, or fitted parameters pinned at
prior edges, would mean the peaked structure needs more than this calibration can
give — informative either way, and the direct contrast to exp 08 is the deliverable.
