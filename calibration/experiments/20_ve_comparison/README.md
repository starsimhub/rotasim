# Exp 20 — Vaccine-impact (VE) comparison over the two model posteriors

**Question.** The payoff. Two structurally-different models both fit the MAL-ED pre-vaccine
data (age+titer = exp 18 posterior; infnum+titer = exp 19 posterior). Apply the *same*
hypothetical vaccine to each and compare the **VE distributions**: do they overlap (VE is
robust to the unidentifiable symptom structure) or separate (VE depends on structure)? The
exp-15 *toy* showed a large point-estimate divergence (age ~0.25 vs infnum ~0.62 at response
0.75); this is the rigorous version — VE *distributions* propagated from the HM→trajectory
posteriors, so the divergence is tested against parameter uncertainty (esp. age's wide
posterior). See [`../15_vaccine_toy/`](../15_vaccine_toy/), [`../18_age_posterior/`](../18_age_posterior/),
[`../19_infnum_posterior/`](../19_infnum_posterior/), `../../VE_HM_PLAN.md`.

**Vaccine mechanism** (A. Kraay's VIMC SI; reused from exp 15 `VaccinePrime`): 2 doses at 2 &
4 months; each seroconversion (prob = `response`) advances one infection-equivalent
(`num_recovered_infections += 1`, up to +2). That counter feeds BOTH acquisition (sus_after_k)
and — in the infnum model only — symptom-given-infection (p_symp); in the age model symptoms
are age-driven, so the vaccine only reduces acquisition. That structural asymmetry is what we
are measuring.

**Plan (Phase B — strictly downstream of the locked posteriors).**
1. For each model, load `../1{8,9}_*/outputs/posterior.csv`, untransform, thin to N_VE draws.
2. Per draw: run baseline (no vaccine) vs vaccinated at the **same seed** (common random
   numbers → a *paired* VE estimate, variance-reduced; the Phase-B seed discipline). Reuse the
   exp-15 cohort sim + `SympIRObserver` (symptomatic IR overall and by age ≤36mo).
3. VE_draw = 1 − IR_symp(vax)/IR_symp(novax), overall and per age bin.
4. Aggregate → VE distribution per model (median + 95% CrI); overlay age vs infnum.

**Design choices (defaults; revisit before launch).**
- **Metric:** overall symptomatic-IR VE *and* by-age VE (total effect, whole-population — same
  as exp 15). By-age speaks to the LMIC-vs-HIC achieved-VE question.
- **Response:** fixed at **0.75** (exp-15 central) so the model-vs-model VE difference reflects
  structure + parameter uncertainty, not vaccine assumptions. Can sweep 0.6/0.75/0.9 later.
- **N_VE:** **800** draws thinned from each posterior (≈ the informative content once ESS is
  known; 800×2 sims ≈ ~1 h/model at 118 workers). If a posterior's ESS ≪ 800, use the unique
  draws and weight by posterior multiplicity.
- **N_agents:** 40k (consistent with the fit), `WINDOW=(5,10)`, Bangladesh demographics.

**Success criteria.** Two VE distributions (age, infnum) with credible intervals. The headline
is their **overlap**: clear separation ⇒ VE depends on the unidentifiable symptom structure (a
structural-uncertainty finding); substantial overlap ⇒ VE is robust. Either is publishable. A
secondary read is the **by-age VE shape** (does the age model predict more age-skewed achieved
VE?), the bridge to the LMIC-vs-HIC question.
