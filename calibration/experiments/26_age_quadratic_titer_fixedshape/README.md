# Exp 26 — quadratic age-symptom + titer (fixed shape) on the CORRECTED maternal

**Question.** Disambiguate exp 25. The binned age-symptom model fit cleanly (NROY 4.7%, reweight
ESS 61.5, all 5 targets) where the quadratic (exp 23) collapsed (ESS 5). But exp 25 changed *two*
things vs exp 23 — binned symptoms **and** the corrected maternal (PR #38: per-infant titer now
persists instead of being redrawn every step). And the binned fitted shape was
**quadratic-compatible** (0.48/0.57/0.31, a 6–11mo peak). So which fix mattered — the functional
form, or the maternal bug? This re-runs the **quadratic** model with *only* the maternal corrected,
making it directly comparable to exp 23.

**Hypothesis check.** The binned-equivalent quadratic is **β0=0.113, β1=−0.082, β2=−0.011** — well
inside (and near the centre of) the standard quadratic prior box
(`SYMPTOM_BOUNDS['age']` = beta0 [−5,2], beta1 [−1,1], beta2 [−0.5,0.5]). So exp 23's failure was
*not* a bounds problem; that good region was always reachable. (Note: a single marginal-median
point overshoots ~5× for *both* the quadratic and binned models — the fit is a posterior-average
property, so only the full HM + reweight settles it, not a point eval.)

**Plan.** Same pipeline/bounds as exp 23, only the maternal is now correct:
```
hm_calibrate.py --model age --maternal titer --fix-titer-shape --all-targets \
    --out-dir experiments/26_age_quadratic_titer_fixedshape/outputs/hm \
    --early-stop --n-samples 5000 --max-iter 8 --resume
```
Then `reweight_overdispersed.py --model age --exp-dir 26_age_quadratic_titer_fixedshape
--phi 3 --rho 0.10`.

**Success criteria / interpretation.**
- If the quadratic now **converges tight** (NROY → ~0.05, like binned) and **reweights to a usable
  ESS** → the **maternal bug**, not the functional form, was the limiter. exp 23's failure was the
  bug. Prefer the quadratic (3 smooth params) as the age member.
- If it **stays loose** (~0.19, like exp 23) and **collapses** under reweight → the parabola
  genuinely **over-constrains** age, and exp 25 (binned) is the age member.

Cross-refs: [`../25_age_binned_titer_fixedshape/`](../25_age_binned_titer_fixedshape/) (binned, ESS
61.5), [`../23_age_titer_fixedshape/`](../23_age_titer_fixedshape/) (quadratic on buggy maternal,
ESS 5).
</content>
