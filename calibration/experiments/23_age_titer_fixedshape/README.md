# Exp 23 — age+titer with the titer SHAPE fixed (mechanism test for the age+titer pathology)

**Question.** age+titer is pathological for the posterior step (loose NROY, 64–83% extinction,
ESS collapse, real-sim validation failure — exp 16/18). The hypothesis (see the maternal-model
discussion): the titer maternal is *redundant* with the age-symptom curve — both are flexible
`<6m`/peak shapers — so together they make a degenerate, weakly-identified space. The
maternal-by-age figure showed the *fitted* titer (infnum) and Erlang (age) protection curves
**coincide**, so it's not the maternal level/shape that differs — it's titer's extra
*flexibility*. This test removes that flexibility: hold the titer SHAPE params
(`median/gsd/half_life/hill`) fixed at the identified curve (`FIXED_TITER_SHAPE`, the infnum
posterior medians) and fit only `maternal_efficacy` + transmission + age betas.

**Plan.** `hm_calibrate.py --model age --maternal titer --fix-titer-shape --all-targets`
(cycle over all 5 targets/wave; 8 waves; 40k agents; cohort obs). Run on covaguest in parallel
with exp 21 (age+Erlang) on raccoon. See [`../16_hm_age_titer/`](../16_hm_age_titer/),
[`../18_age_posterior/`](../18_age_posterior/), [`../21_hm_age_erlang/`](../21_hm_age_erlang/).

**Success criteria.** If age+titer-fixed-shape now yields a **tight, low-extinction, identifiable
NROY** (like age+Erlang is expected to), that confirms the pathology was titer's redundant
*flexibility* (mechanism (b)), not the maternal model itself — and validates the general
strategy of *fixing maternal immunity to the data-identified curve before fitting*. If it's
*still* loose/degenerate, the age model's identifiability problem is deeper than the maternal
parameterization.
