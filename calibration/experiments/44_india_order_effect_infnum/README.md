# Exp 44 (infnum sibling) — India Vellore: fractional neonatal order-crediting under infnum

**Question.** exp43 showed `age_binned` is structurally immune to any fix that
works through the infection-order counter (its symptom probability depends only
on age). Since neonatal priming's only remaining lever (after dropping the
ungrounded detection-probability discount AK flagged) is an order-credit, it can
only matter under an order-sensitive symptom model. exp31 already tried `infnum`
+ a full, deterministic order-credit (`hm_neoprime`) and it helped (best of that
arc) but did not resolve the &lt;6m overshoot / 6-11m undershoot tension. This asks:
does letting the credit be a FITTED FRACTION (`neonatal_order_effect`, 0-1) —
"what if the neonatal infection didn't count as much toward the counter" — find
an untested interior point better than either endpoint (0 = no-priming baseline,
1 = exp31's full-credit)?

**Design.** `NeonatalPriming` reverted to UNDETECTED (exp43's real-detected-event
mechanism caused a Q25 overshoot in the opposite direction, independent of
symptom model — dropped, see exp43 SUMMARY). Its only effect now: a per-child
probabilistic credit (sized by `order_effect`) toward the symptom-order lookup
for that child's next real infection. `p_neo=0.5`, `age_weeks=2.0`,
`sus_effect=0.0` all stay literature-fixed (unidentifiable from this data).
`neonatal_order_effect` is the one NEW free parameter, bounded [0,1].

**Run:** `--model infnum --fix-psymp --maternal titer --all-targets`,
`MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1`, 6 waves x 1500 samples,
`--early-stop`, on zebra. `--fix-psymp` uses the same biweekly-derived
`p_symp_1=0.407` etc. as exp31/32 — isolates `order_effect` + FOI/immunity as
the new dimensions, rather than re-opening the psymp-freedom question exp35
already explored separately.

**Parameters (free, 10):** `log_base_beta, sus_after_1, sus_r2, sus_r3,
log_titer_median, titer_gsd, titer_half_life_days, hill_slope,
maternal_efficacy, neonatal_order_effect`.

**Success criteria (vs exp39 age_binned: ESS 9.15, IR&lt;6m 0.59/6-11m 1.39/12-23m
0.60, repeat 0.116, Q25 17.5; vs exp31 hm_neoprime's full-credit infnum result:
&lt;6m 0.57-0.98, 6-11m undershoot persisted regardless):** does the fitted
`order_effect` posterior concentrate away from both 0 and 1 (data actually
prefers an interior value), and does that interior value narrow the &lt;6m/6-11m
gap beyond what either endpoint achieved?

**Sibling:** `../44_india_order_effect_age_inf/README.md` — same question under
`age_and_infection` (age-sensitive AND order-sensitive; only correctly
implemented in `MALEDCohort` as of today, see exp40's correction note).
