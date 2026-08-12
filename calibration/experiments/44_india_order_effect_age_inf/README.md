# Exp 44 (age_and_infection sibling) — India Vellore: fractional neonatal order-crediting under age_and_infection

**Question.** Same as `../44_india_order_effect_infnum/README.md`, but under
`age_and_infection` (quadratic age logit + a linear infection-order slope
`beta3*min(order,5)`) instead of plain `infnum`. Found while scoping this
experiment: `MALEDCohort` (the cohort observer used for every India experiment)
never actually implemented `age_and_infection` — no `beta3` parameter, and
`_symp_prob` silently fell through to a plain order-based lookup with `p_symp`
defaulting to `[1.0, 1.0, 1.0]` (i.e. every infection scored 100% symptomatic).
That invalidates exp40's "age_and_infection loses the comparison" conclusion —
see the correction note on `../40_india_age_inf_extpen/SUMMARY.md`. Fixed in
`rotasim/analyzers.py` + `calibrate_maled.py` (mirrors `Surveillance`'s already-
correct implementation, used for UK/exp28) and smoke-tested. This is therefore
the FIRST valid test of `age_and_infection` against the Vellore cohort, combined
with the same fractional neonatal order-credit as the infnum sibling — since
`age_and_infection` is order-sensitive too (nests `infnum`-like behavior via
`beta3`), the credit mechanism applies here as well.

**Design.** Same `NeonatalPriming`/`order_effect` mechanism as the infnum
sibling (undetected, probabilistic order-credit only; `p_neo`/`age_weeks`/
`sus_effect` fixed).

**Run:** `--model age_and_infection --maternal titer --all-targets`,
`MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1`, 6 waves x 1500 samples,
`--early-stop`, on zebra. No `--fix-*` mechanism exists yet for `beta0-3`
(unlike `--fix-psymp`/`--fix-age-psymp` for the other models) — all four left
free. That's 14 total free parameters (vs the infnum sibling's 10), so a
higher risk of the same ESS collapse seen in exp40's (invalid) run — watch
this, since more parameters alone doesn't mean a better answer.

**Parameters (free, 14):** `log_base_beta, sus_after_1, sus_r2, sus_r3,
log_titer_median, titer_gsd, titer_half_life_days, hill_slope,
maternal_efficacy, beta0, beta1, beta2, beta3, neonatal_order_effect`.

**Success criteria:** ESS not collapsed to ~1 (would indicate this parameter
count is simply too high for 44 cases, independent of whether the structure is
right); does the fit beat exp39's age_binned baseline on the &lt;6m/6-11m pair
specifically (the one thing age_binned structurally cannot do); does
`neonatal_order_effect` concentrate away from 0/1.

**Sibling:** `../44_india_order_effect_infnum/README.md`.
