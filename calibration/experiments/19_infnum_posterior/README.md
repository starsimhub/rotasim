# Exp 19 — Trajectory-selection posterior for the INFECTION-NUMBER + titer model

**Question.** The matched partner to exp 18. Turn the exp-17 infnum+titer **NROY** into a
**posterior** via trajectory selection (importance resampling), for the VE-distribution
comparison. infnum's NROY is tight and well-identified (exp 17: 0.023, still converging;
repeat-fraction pinned `sus_after_1`≈0.60), so we expect a tighter posterior and higher ESS
than age. Matched partner: [`../18_age_posterior/`](../18_age_posterior/). Consumed by the VE
comparison (exp 20).

**Plan.** Identical to exp 18, via the same driver with `--model infnum`:
1. **Draw** N≈5000 NROY samples from the exp-17 HM checkpoint, cached to `outputs/nroy_draw.csv`.
2. **Simulate** each at 40k agents, cohort observation, one fixed seed per draw (`BASE + idx`).
3. **Score** with the same composite log-likelihood — Poisson(IR bins) + Binomial(repeat) +
   censored-survival(first-inf); **no ever-detected channel**; extinct → −∞.
4. **Importance-resample** → `outputs/posterior.csv`; report ESS. Streamed/resumable; pinned env;
   on a 120-core VM.

The only difference from exp 18 is the symptom model (infection-number `p_symp_1/p_r2/p_r3`
instead of the age betas) — same observation, likelihood, seeds, and method, so the two
posteriors are directly comparable.

**Success criteria.** A posterior over the infnum box params + posterior-predictive covering the
five targets, with a healthy ESS (expected higher than age, since the NROY is tighter). Combined
with exp 18 it gives the two posteriors for the rigorous VE comparison (exp 20) — do the VE
distributions overlap?
