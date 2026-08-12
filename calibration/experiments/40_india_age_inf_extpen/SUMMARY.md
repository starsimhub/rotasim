# Exp 40 — India Vellore: age_and_infection with extinction penalty

**Date:** 2026-07-14 (run) / 2026-08-11 (closed).

**Question.** Exp 33 tried `age_and_infection` (age-binned offset + infection-order
symptom modifier, combining both structures) for India but only ran 3 waves with
no extinction penalty and came back with ESS=1 (under-identified, not a clean
result). Does 6 waves + an extinction penalty (`EXT_PENALTY=1`, same treatment
that helped exp39/exp35) let `age_and_infection` mix well enough to judge fairly
against `age_binned` (exp39)?

**Result.** No — still degenerate. ESS = 1.25/3000 (442/3000 finite logL),
barely better than exp33's ESS≈1 and far below exp39's 9.15. The (unreliable,
because ESS≈1) weighted fit is also worse than exp39 on 4 of 5 targets: IR&lt;6m
1.12 (target 0.40, badly overshooting), IR 6-11m 1.55 (target 1.71, closer than
exp39 but not meaningfully — see caveat below), IR 12-23m 1.32 (target 0.61,
badly overshooting), repeat_frac 0.103 (target 0.138), Q25 20.0mo (target
15.1mo, worse than exp39's 17.5).

![Exp 39 vs 40 — India Vellore cohort fit vs targets, posterior-weighted](../39_india_age_binned_fixed/figures/exp39_vs_exp40_comparison.png)

## Observations

1. **`age_and_infection` loses the India model comparison**, same pattern as the
   UK result in exp28 (`age_and_infection` nests `infnum`, has more free
   parameters, and is under-identified rather than genuinely well-fit — its
   better-looking G²/logL there was flagged as not a clean win either).
2. **ESS=1.25 means these numbers are close to a single best-fit point**, not a
   real posterior average — treat the "closer on 6-11m" reading with real
   skepticism rather than as a genuine advantage.
3. **`age_binned` (exp39) is the working India structure going forward.** No
   further tuning of `age_and_infection` is planned unless a future structural
   change (e.g. decoupled neonatal priming, higher FOI) changes the picture.

## Next

- `age_binned` (exp39) carries forward into VE validation:
  `../41_india_ve_validation/SUMMARY.md`.

## CORRECTION (2026-08-11)

**This experiment did not actually test `age_and_infection`.** Found while scoping
exp44: `MALEDCohort` (the cohort observer used for every India experiment) never had
a `beta3` parameter and its `_symp_prob` only branched on `'age_only'`/`'age_binned'`
— anything else, including `'age_and_infection'`, silently fell through to a plain
order-based `p_symp[order]` lookup with `p_symp` defaulting to `[1.0, 1.0, 1.0]`
(since `age_and_infection`'s `untransform()` branch never populates
`p_symp_1/2/3plus`). So this run actually simulated **every infection as 100%
symptomatic**, unrelated to the intended age+order quadratic-logit model — that
also parsimoniously explains the across-the-board overshoot above (IR&lt;6m 1.12,
IR12-23m 1.32 — consistent with symptomatic IR collapsing toward the *all-infection*
IR once nothing suppresses it). `Surveillance` (used for UK, exp28) implemented
`age_and_infection` correctly the whole time — `MALEDCohort` was simply never
updated to match. Fixed in `rotasim/analyzers.py` (`MALEDCohort.__init__` +
`_symp_prob` now mirror `Surveillance`'s formula) and `calibrate_maled.py`
(`beta3` now passed through). The observations above describing "ESS=1.25,
under-identified" are still true of *what actually ran*, but the conclusion
"`age_and_infection` loses the India comparison" is **not established** —
`age_and_infection` has not yet been validly tested against the Vellore cohort.
