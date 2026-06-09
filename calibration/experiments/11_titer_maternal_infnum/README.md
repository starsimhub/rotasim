# Exp 11 — Does titer maternal *alone* let infection-number make the peak?

**Question.** The decisive test of the model-selection fork. Your exp 06/08/09 found the
infection-number symptom model **cannot** reproduce the MAL-ED 6-11mo symptomatic-IR
peak (it stays `<6m`-highest / flat) — but those used **Erlang** maternal immunity. D.
Klein's exp 06 reproduced a *sharp* peak with the **same infection-number symptoms** by
switching to an **IBM titer-based maternal** model — but bundled with structured mixing
(low young-reservoir + a 5+ "persistence" reservoir) and a cohort/detection observation.
This experiment isolates the maternal model: run infection-number + **titer maternal**
under the **same homogeneous mixing and `process_model` observation as exp 09/10** (no
reservoir levers, no cohort/detection emulation, no EIA sampling — so it's apples-to-
apples with the Erlang baselines). Does titer maternal *by itself* flip the result? See
`../06_infection_number_compare/SUMMARY.md`, `../09_shape_aware_likelihood/SUMMARY.md`,
`../10_denominator_rerun/SUMMARY.md`.

**Plan.** `calibrate_maled.py --symptom-model infection_number --maternal-model titer
--fit-target poisson --n-trials 40 --n-reps 20`, Bangladesh demographics, homogeneous
mixing (`RandomNet`, n_contacts=7), corrected `PersonTimeByAge` denominator,
covaguest. The titer params are **fitted** (median 4-60 IC50 log, gsd 1.3-3.5, half-life
25-70 d, Hill slope 1.5-8 — priors matching DK exp-06); infection-number `p_symp_1/2/3+`
fitted as before; susceptibility ladder + base_beta as before. Seed the search near DK's
exp-06 titer best (median 41 / half-life 68d / gsd 2.6 / Hill 6.2) so the sharp-protection
basin has a foothold. Study `rota_maled_bangladesh_infnum_titer_poisson`. Compare the
best fit + shape scorecard against (a) the MAL-ED target and (b) the exp 09/10
infection-number (Erlang) fits, which were `<6m`-high / flat.

**Success criteria.** *Titer flips it:* the infection-number + titer fit reproduces the
shape (low `<6m`, 6-11mo peak, decline; shape cosine ~0.99, peak in the right bin) — then
the maternal model, not an age-symptom curve, carries the peak (rising limb = maternal
waning, falling limb = per-infection severity), and the conclusion reframes to "monotone
symptoms + titer maternal works *and* generalizes across the FOI gradient." *Titer
doesn't flip it:* the fit stays `<6m`-high / flat like the Erlang version → titer maternal
alone is insufficient and DK's peak needed the reservoir mixing too, making your
age-symptom curve the more parsimonious, better-justified model. Either outcome is
decision-grade for the model selection.
