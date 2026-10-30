# Exp 14 — Age-symptom + titer maternal under the cohort observation (Set 2 partner)

**Question.** The age-symptom partner for exp 13 (infnum+titer+cohort), to form the
cohort-matched pair (same maternal = titer, same observation = cohort, differ only in
symptom model). exp 12 showed age+titer fits poorly under `process_model` (strong titer
crushes the age model's already-mild `<6m` to 0.21). Can the age model fit under titer at
all — i.e., can TPE find a *weak*-titer region where the age curve does the work and titer
barely suppresses `<6m` — under the cohort observation + repeat-fraction constraint? If
yes, exp 13 + exp 14 are a clean matched pair for the VE comparison. If no, the two
symptom models need opposite maternal strengths (age=gentle, infnum=sharp) and the honest
pair is {age+Erlang (exp 10)} vs {infnum+titer (exp 11/13)}. See `../12_age_titer/`,
`../13_cohort_emulation/`.

**Plan.** `calibrate_maled.py --site bangladesh --symptom-model age_only
--maternal-model titer --observation cohort --fit-target cohort --n-trials 40
--n-reps 20`, Bangladesh, homogeneous mixing. Seeded with a WEAK titer (efficacy 0.5,
median 4, half-life 25d, Hill 1.5) + the exp-10 peaked age curve, so titer is near-off
and the age curve leads — the configuration most likely to avoid the `<6m` over-crush.
Study `rota_maled_bangladesh_titer_cohort`.

**Success criteria.** A good age-model fit under titer (cosine ~0.99, `<6m` not crushed,
peak in 6-11m, repeat fraction near 0.40) → exp 13 + exp 14 = clean same-maternal cohort
pair → use for the VE comparison. A poor fit (`<6m` crushed again, like exp 12) → confirms
symptom/maternal entanglement; fall back to {age+Erlang} vs {infnum+titer} as the
structural pair. Either outcome resolves the matched-pair question.
