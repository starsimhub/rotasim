# Exp 44 (infnum sibling) — fractional neonatal order-crediting under infnum

**Date:** 2026-08-11 (run) / 2026-08-12 (closed).

**Question.** exp43 showed `age_binned` is structurally immune to any fix that
works through the infection-order counter. Does `infnum` — order-sensitive by
construction — benefit from a FITTED fractional order-credit
(`neonatal_order_effect`, 0-1) for neonatally-primed children, landing on an
untested interior point between exp31's two already-tried endpoints (no
priming, and `hm_neoprime`'s full deterministic credit)?

**Result.** No improvement, and the joint fit got worse. Weighted vs exp39's
`age_binned` baseline: IR &lt;6m 0.60 (unchanged, target 0.40); IR 6-11m
**1.14 (worse than exp39's 1.39, target 1.71 — the core peak-undershoot got
bigger, not smaller)**; IR 12-23m 0.60 (unchanged); repeat_frac 0.10 (worse,
target 0.138); Q25 17.2mo (~unchanged, target 15.1). ESS collapsed to
**2.98/3000** (372 finite), a third of exp39's 9.15 — a harder-to-satisfy
joint constraint, not an easier one.

![Exp 44 — order-crediting does not resolve the <6m/6-11m tension](figures/exp39_vs_exp44_siblings.png)

## Observations

1. **The fitted `neonatal_order_effect` posterior concentrates well above
   zero** — median 0.71, mean 0.62, 10-90% range [0.30, 0.75] — closer to
   exp31's full-credit finding than to a "no protection against symptomatic
   infection" prior (the hypothesis AK's separate conversation with a browser
   Claude instance raised). Given the ESS this posterior is drawn from is
   itself only 2.98, treat this as suggestive, not a settled answer — but note
   it does NOT support fixing the parameter at 0.
2. **Adding a free dimension to an already order-sensitive model made
   identifiability worse, not better.** infnum + `order_effect` (10 free
   params) has under a third of `age_binned`'s (9 params) ESS, despite testing
   the exact mechanism (order-crediting) that should, in principle, have
   leverage on the &lt;6m/6-11m pair that `age_binned` structurally lacks. Having
   leverage on a target and actually improving the fit are different things.
3. **This is now the third distinct India symptom-model structure
   (age_binned, infnum, age_and_infection — see sibling) and third neonatal-
   priming design (symptom-flag-only [exp31], real-detected-event [exp43],
   fractional order-credit [this experiment]) that has failed to close the
   &lt;6m/6-11m gap.** That breadth of negative results is itself evidence:
   this doesn't look like a missing single-mechanism problem anymore.

## Next

See `../44_india_order_effect_age_inf/SUMMARY.md` for the sibling result
(worse — degenerate ESS≈2). Given the accumulated negative results across
model structure AND priming design, the two live candidates going forward are
(1) a genuine two-strain representation (the persistent community strain and
wild-type strains as separate, weakly-cross-protective populations — not
patchable within a single-strain model no matter how the symptom/order
mechanics are tuned), or (2) treating this as a data-adequacy problem (44
cases jointly constraining 9-14 parameters) rather than a structural one, and
leaning on larger-N India data (TN surveillance) instead of continuing to
refine against the sparse Vellore cohort. Not yet decided between these.
