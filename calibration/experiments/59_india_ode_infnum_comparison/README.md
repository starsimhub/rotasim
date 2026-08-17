# Exp 59 — India Vellore: recode the ODE cohort model for infnum, compare to age_binned

**Question.** All ODE-pipeline work so far (exp54-58) has used `age_binned`
(p_symp keyed by age bin), because that's what the ABM HM arc converged on
as India's working symptom structure. But `infnum` (p_symp keyed by
infection order) was the structure originally tried for Vellore in
exp31-35, before the arc moved to age_binned — and that comparison predates
the freed-p_symp, neonatal-priming, and survival-vote decisions made since.
Now that a full 12-parameter global MLE fit costs ~5 minutes instead of
days (exp57's BDF fix) and exp58 will have established whether age_binned's
ODE optimum is even stable, this experiment asks: does `age_binned` still
win over `infnum` once both are fit through the same fast, deterministic
ODE pipeline under current (2026-08) decisions?

**Why this is a modest recode, not a rewrite.** Checked directly against
`cohort_model.py` and `ode_model_age.py`: both already track infection
**order** as a first-class state axis (`S_j`/`IS_j`/`IA_j`/`R_j` for
j=0..3), because the susceptibility structure (`sus_after_1`, `sus_r2`,
`sus_r3`) is already order-indexed. `p_symp` is looked up by AGE in exactly
two places: (1) `cohort_model.rhs_cohort`, via `p_det_of_age_months`,
called once per timestep outside the per-order loop; (2) `run.py`'s
`model_predict`, which selects `p_symp_age[label]` per age bin when
computing `ir_symp` from the age-structured equilibrium state. Both already
have the order index available in scope (`j` in the cohort model's loop;
per-order equilibrium fractions in the age model's state vector) — infnum
only requires re-keying these two lookups by order instead of age, not new
compartments or a new ODE structure.

**Plan.**
1. Add `p_symp_of_order(j, p_symp_order)` alongside the existing
   `p_symp_of_age_months` in `cohort_model.py` (keep both — age_binned
   stays available for comparison, controlled by a `model` flag mirroring
   `hm_calibrate.py`'s `--model age_binned`/`infnum` convention). Move the
   `pdet` computation inside `rhs_cohort`'s per-order loop when
   `model=='infnum'`.
2. Update `model_predict`'s IR-by-age calculation to pull p_symp by the
   equilibrium state's per-order fractions when `model=='infnum'`, instead
   of the per-age-bin lookup.
3. Update `PARAM_NAMES`/`BOUNDS` for the infnum variant: replace
   `p_symp_age_0_6/6_11/12plus` with `p_symp_order_1/2/3plus` (3 params,
   same count as age_binned's, so the two models stay directly comparable
   on parameter budget).
4. Refit via `differential_evolution`, same settings as exp57/58 (BDF
   solver, `workers=-1`, `maxiter=60`, `popsize=15`). Given exp58 will have
   already shown whether a single seed is trustworthy for age_binned, run
   infnum across the same number of seeds exp58 used, not just one.
5. Compare: best logL, per-target fit, and (if exp58 finds a stable
   optimum for age_binned) whether infnum's best logL is meaningfully
   different from age_binned's, not just whether individual targets look
   better or worse.

**Success criteria.** If infnum's best achievable logL is close to
age_binned's (like exp57 vs exp52's 1-log-unit gap), that would say the
recent structural-ceiling finding isn't specific to age_binned's
functional form — a stronger, more general claim than exp57 alone
supports. If infnum does meaningfully better or worse, that's new evidence
bearing directly on India's symptom-model selection, which has been
carried forward from exp39 without being re-tested under the current
(freed-p_symp, corrected-anchor, survival-vote) decisions.

**Status:** not yet started — waiting on exp58's stability results before
launching, so the age_binned side of this comparison rests on a confirmed
(not single-seed) optimum.
