# Exp 39 — India Vellore: age_binned with biweekly-fixed p_symp

**Date:** 2026-07-14 (run) / 2026-08-11 (closed).

**Question.** Following the exp31-35 arc (infnum under Vellore's cohort data hit a
Pareto tension between symptomatic-IR-by-age and age-at-first-infection; the
biweekly Vellore birth cohort suggested `age_binned`'s p_symp bins should be fixed
directly from data rather than fitted), does `age_binned` with p_symp fixed at the
biweekly-derived values (<6m 0.381, 6-11m 0.407, 12-23m 0.189, 24-35m 0.122),
titer maternal, neonatal priming (`NEO_PRIME=1`), and FOI anchored via all-infection
IR (`USE_IR_ALL`, auto-on for `MALED_SITE=india`) fit the Vellore MAL-ED cohort?
6 HM waves, 1500 samples/wave, all-targets, extinction penalty
(`EXT_PENALTY=1`) placed first in the likelihood cycle.

**Result.** Best-mixing India HM run to date: ESS = 9.15/3000 (438/3000 finite
logL), vs exp31-35's ESS≈1 point estimates. Posterior-weighted fit is close on
two of five targets but still misses the young end of the age curve: IR&lt;6m
0.59 (target 0.40), IR 6-11m **1.39 (target 1.71, still undershooting the peak)**,
IR 12-23m 0.60 (target 0.61, good), repeat_frac 0.116 (target 0.138), Q25
first-infection 17.5mo (target 15.1mo).

![Exp 39 vs 40 — India Vellore cohort fit vs targets, posterior-weighted](figures/exp39_vs_exp40_comparison.png)

## Observations

1. **ESS improved by ~9x over the exp31-35 arc** (9.15 vs ≈1) — fixing p_symp from
   external biweekly data plus anchoring FOI via all-infection IR meaningfully
   reduced degeneracy, without changing the qualitative miss.
2. **The residual miss is the same shape as exp31's**: &lt;6m over-predicted,
   6-11m under-predicted (the model can't get the peak sharp enough), first
   infection detected too late. Fixing p_symp did not resolve this — it's on the
   FOI/susceptibility side, not the symptom side.
3. **12-23m and repeat_frac are both close** — the model is not uniformly wrong,
   only wrong at the young end, which is where neonatal-priming mechanics and
   maternal protection interact most.
4. This run is the reference posterior used downstream for VE forward-prediction
   (exp41) and VE-scored reweighting (exp42) — both draw on the HM checkpoint at
   `outputs/hm/maled_age_binned_titer_fixedagepsymp/`.

## Next

- `../40_india_age_inf_extpen/SUMMARY.md` — does adding an infection-order term
  on top of age-binned symptoms do better?
- `../41_india_ve_validation/SUMMARY.md` — does this posterior reproduce
  plausible vaccine impact?
- `../42_india_ve_scored_ts/SUMMARY.md` — can VE-plausibility be added as a
  scoring target to steer this posterior without a structural change?
