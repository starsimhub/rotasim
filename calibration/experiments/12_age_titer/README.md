# Exp 12 — Peaked age-symptom model + titer maternal (completes the maternal x symptom 2x2)

**Question.** The maternal x symptom grid so far: age+Erlang (exp 09/10, the strong
baseline), infnum+Erlang (exp 09/10, fails the peak), infnum+titer (exp 11, testing
whether titer flips it). The missing cell is **age + titer**. Does swapping the peaked
age-symptom model's maternal component from Erlang to D. Klein's titer model change or
*sharpen* the (already good) peaked fit? Same homogeneous mixing + `process_model`
observation as exp 09-11, so it's apples-to-apples. See `../10_denominator_rerun/SUMMARY.md`,
`../11_titer_maternal_infnum/README.md`.

**Plan.** `calibrate_maled.py --symptom-model age_only --maternal-model titer
--fit-target poisson --n-trials 40 --n-reps 20`, Bangladesh demographics, homogeneous
mixing (`RandomNet`), corrected `PersonTimeByAge` denominator, covaguest. Age logistic
betas + titer params (median/gsd/half-life/Hill) all fitted. Seeded with the exp-09/10
peaked betas + DK-titer config so the basin has a foothold. Study
`rota_maled_bangladesh_titer_poisson`. Compare the best fit + shape scorecard against the
exp 10 peaked+Erlang fit and the MAL-ED target.

**Success criteria.** This is a completeness / robustness run, so both outcomes are
informative. If age+titer fits as well or better than age+Erlang (cosine ~0.99, peak in
6-11mo bin, level matched) -> the peaked conclusion is robust to the maternal model, and
titer may tighten the `<6m` trough. If it fits worse -> the Erlang maternal was doing
useful work for the peaked model. Low-stakes; mainly fills the grid cell and uses idle
compute.
