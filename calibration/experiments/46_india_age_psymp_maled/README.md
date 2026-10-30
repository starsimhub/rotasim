# Exp 46 — India Vellore: age_binned refit with MAL-ED-derived (not slum-derived) p_symp

**Question.** exp39's `age_binned` fit uses `FIXED_AGE_PSYMP` — P(symp|infected)
by age bin sourced from the Vellore biweekly **slum** cohort (Lewnard et al.):
&lt;6m 0.381, 6-11m 0.407, 12-23m 0.189. AK recalculated these fractions
**directly from MAL-ED India** (not the slum cohort) with the asymptomatic-
ascertainment correction applied (~50% under-detection at monthly stool
frequency — the same correction identified in exp35 but never carried into
`FIXED_AGE_PSYMP`) and got materially different numbers: &lt;6m 0.172, 6-11m
0.511, 12-23m 0.444 — close to a **flip** of the slum-derived &lt;6m/12-23m
values. This directly targets the persistent tension every India experiment
since exp31 has hit: exp39's model overshoots &lt;6m and undershoots the 6-11m
peak. A lower &lt;6m and higher 6-11m symptom probability point exactly the
right direction to fix both without touching FOI or immunity at all.

**Design.** Added `FIXED_AGE_PSYMP_MALED` alongside the existing
`FIXED_AGE_PSYMP` in `hm_calibrate.py`, selected via `AGE_PSYMP_SOURCE=maled`
(env var, default `slum` — exp39 unaffected/reproducible). No other change:
same model (`age_binned`), same free parameters (9: `log_base_beta,
sus_after_1, sus_r2, sus_r3, log_titer_median, titer_gsd,
titer_half_life_days, hill_slope, maternal_efficacy`), same
`--fix-age-psymp`, same `--all-targets`, same `NEO_PRIME=1`/`EXT_PENALTY=1`.
This isolates the p_symp source as the ONLY variable versus exp39 — a clean
apples-to-apples comparison.

**Run:** `--model age_binned --maternal titer --fix-age-psymp --all-targets`,
`MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 AGE_PSYMP_SOURCE=maled`, 6 waves
x 1500 samples, `--early-stop`, on zebra.

**Success criteria (vs exp39: ESS 9.15, IR&lt;6m 0.59/6-11m 1.39/12-23m 0.60,
repeat 0.116, Q25 17.5; targets 0.40/1.71/0.61/0.138/15.1):** does IR&lt;6m come
down toward 0.40 and IR6-11m come up toward 1.71 — the two persistent misses —
without a corresponding ESS collapse (which would indicate this just trades
one tension for another, as every other exp39 variant has so far). 12-23m is
expected to move too (0.444 is much higher than the slum-derived 0.189) — not
obviously in a good direction given exp39 already fit that bin well, worth
watching but secondary to the primary &lt;6m/6-11m question.
