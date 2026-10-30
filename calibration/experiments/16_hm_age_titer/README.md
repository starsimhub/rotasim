# Exp 16 — History-matching posterior for the AGE-symptom + titer model (the re-test)

**Question.** Two things at once. (1) The collaborator's point: the age+titer model fit
*poorly* under Optuna (exp 12/14: `<6m` crushed, cosine ~0.91) — but that may be an
*optimization artifact* (Optuna struggled on these surfaces). Does age+titer fit well under
**history matching** (which explores the space differently and yields the NROY region)? If
yes, the clean *same-maternal* matched pair (age+titer vs infnum+titer, exp 17) is
recoverable. (2) Produce a **posterior** (not a point fit) so the downstream VE comparison
is over VE *distributions*. See `../12_age_titer/`, `../14_age_cohort/`, `../../VE_HM_PLAN.md`.

**Plan.** `historymatching` v2.0.1 (now installed in the `rota-hm` env on covaguest),
reusing D. Klein's exp-07 `hm.HistoryMatching(...)` pattern. Symptom model = `age_only`
(age betas), maternal = titer, **homogeneous mixing** (no reservoir — fewer params than
DK's setup), **cohort observation** (`MALEDCohort`) so the targets carry their
uncertainties. Box-bounded params (~9): `log_base_beta`; `sus_after_1`, `sus_r2`, `sus_r3`
(ratios for monotonicity); titer block (`log_titer_median`, `gsd`, `half_life`, `hill`,
`maternal_efficacy`); age betas `beta0/1/2`. Targets (mean, SD): symptomatic IR in the
reliable age bins (Poisson SD on counts), repeat-detected fraction 0.403 (binomial SE),
age-at-first-DETECTION median (KM). Bayes-Linear emulator, auto feature selection
(1/wave, cooldown 2), implausibility 3.0, ~1–2k sims/wave, ~6 waves -> NROY ->
trajectory-selection posterior. tmux + HM's native resume (covaguest is spot). The HM
simulator reuses `calibrate_maled._run_one_replicate` (cohort path) so the sim is the
validated one. Validate wave 1 (emulator R²) before scaling.

**Success criteria.** A good age+titer NROY/fit (peak in 6–11mo, `<6m` not crushed,
repeat-fraction ~0.40, sensible KM median) => Optuna's poor exp-12 fit WAS an artifact;
the same-maternal matched pair is recoverable, and we get an age-model VE posterior.
A poor/empty NROY (collapses, `<6m` still crushed) => the symptom/maternal entanglement is
real (titer over-suppresses the age model's `<6m`), and we fall back to age+Erlang as the
age member of the (maternal-confounded) structural pair.
