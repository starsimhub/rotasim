# Exp 47 — India Vellore: age_binned with p_symp free, bounded to the slum↔MAL-ED bracket

**Question.** exp46 (MAL-ED-derived p_symp) overshot exp39's (slum-derived
p_symp) misses on both &lt;6m and 6-11m, but in the *correct direction*, and
both targets sit BETWEEN the two anchor values for each bin. Does letting
`p_symp_age_0_6/6_11/12plus` vary FREELY within the interval spanned by the
two anchors (rather than fixing at either endpoint) let HM find an interior
point that satisfies both targets simultaneously, with less ESS damage than
either fixed extreme?

**Design.** New `AGE_PSYMP_INTERP=1` env var (added to `hm_calibrate.py`)
sets `p_symp_age_0_6/6_11/12plus` bounds to the slum↔MAL-ED bracket (with a
small margin) instead of the uninformed (0,1) default:
`p_symp_age_0_6: (0.15, 0.40)` [slum 0.381, MAL-ED 0.172],
`p_symp_age_6_11: (0.35, 0.55)` [slum 0.407, MAL-ED 0.511],
`p_symp_age_12plus: (0.15, 0.50)` [slum 0.189, MAL-ED 0.444].
No `--fix-age-psymp` flag (p_symp genuinely free within these bounds).

**Confirmed before running (AK, 2026-08-13):** `p_symp` has NO feedback into
transmission dynamics at all — `MALEDCohort` is a read-only observer;
disease-level transmission reduction (90% during a fixed 8-day post-infection
tail) is unconditional on every infection, independent of the case-
classification `p_symp` governs. So this experiment can only reshuffle which
bin's infections get counted as symptomatic cases — it will NOT move
`repeat_frac` or the extinction rate, which are governed entirely by
`base_beta`/`sus_after_*`/maternal params. Expect those to land wherever the
transmission-side parameters do, roughly independent of this experiment's
result.

**Run:** `--model age_binned --maternal titer --all-targets` (no
`--fix-age-psymp`), `MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1
AGE_PSYMP_INTERP=1`, 6 waves x 1500 samples, `--early-stop`, on zebra.

**Parameters (free, 12):** the same 9 as exp39/46
(`log_base_beta, sus_after_1, sus_r2, sus_r3, log_titer_median, titer_gsd,
titer_half_life_days, hill_slope, maternal_efficacy`) plus
`p_symp_age_0_6, p_symp_age_6_11, p_symp_age_12plus` (bounded, not fixed).
3 more free dims than exp39 -- watch ESS given the pattern so far (every
added free dimension has hurt ESS, from 9.15 down to 1.2-3.0). Narrower
bounds than a fully free search should limit the damage, but this is a real
risk, not a given.

**Success criteria:** IR &lt;6m and 6-11m both land closer to target than
EITHER exp39 or exp46 individually; ESS not catastrophically worse than
exp46's already-low 1.26 (a further collapse would suggest the interior
isn't better-behaved, just differently thin).
