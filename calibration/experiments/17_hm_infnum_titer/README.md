# Exp 17 — History-matching posterior for the INFECTION-NUMBER + titer model

**Question.** The matched partner to exp 16. The infection-number + titer model already
fits under Optuna (exp 11, cosine 0.974; exp 13 under the cohort observation). Produce its
**HM posterior** so the VE comparison is over VE *distributions* (do the two models'
distributions overlap?), and so the repeat-fraction target properly constrains the
susceptibility ladder (exp 11's `sus_after_1 = 0.93` was under-identified — fit without the
repeat-fraction). Same observation/objective/method as exp 16 — only the symptom model
differs (the matched pair). See `../11_titer_maternal_infnum/`, `../16_hm_age_titer/`,
`../../VE_HM_PLAN.md`.

**Plan.** Identical HM setup to exp 16 (same `hm.HistoryMatching` driver, homogeneous
mixing, cohort observation, same targets + uncertainties, Bayes-Linear, auto feature
selection, ~6 waves -> trajectory-selection posterior, tmux + resume), via the same
parameterized run script with `--model infnum`. The only differences from exp 16: symptom
model = `infection_number`, and the symptom params are `p_symp_1`, `p_r2`, `p_r3` (ratios)
instead of the age betas. ~9 box params: `log_base_beta`; `sus_after_1/r2/r3`; titer block;
`p_symp_1/p_r2/p_r3`. Can run in parallel with exp 16 on covaguest (independent;
spot-resilient) once the pipeline is validated on exp 16's wave 1.

**Success criteria.** A converged infnum+titer NROY/posterior matching the targets
(it should — the model fits), with the repeat-fraction now pinning `sus_after_k` (expect
`sus_after_1` to come down from Optuna's 0.93). This is the infection-number member of the
matched pair; combined with exp 16 it gives the two HM posteriors for the rigorous VE
comparison (redo the exp-15 toy with VE distributions, which may shift the central estimate,
not just add bands — since the toy rested on the under-identified Optuna sus ladder).
