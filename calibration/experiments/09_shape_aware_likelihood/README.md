# Exp 09 — Re-calibrate both models under a shape-aware (Poisson + multinomial) likelihood

**Question.** Exp 07/08 exposed a problem with the current objective: the squared-log
joint GOF is a per-bin *level* penalty, so it nearly tied the two models on the scalar
(peaked-age 3.62 vs infection-number 4.80) even though their *shapes* are very
different (normalized-profile L1 0.13 vs 0.45). The peaked model's error was almost
all level overshoot (~3x) with near-perfect shape; the infection-number model's was a
genuine shape error. So the objective penalized the right model for level while letting
the wrong model's flat shape slide. Does an objective that explicitly scores *shape*
(a) sharpen the model comparison to decisive, and (b) pull the peaked model's level
overshoot down toward the data? See `../07_calibrate_peaked_age/SUMMARY.md` and
`../08_calibrate_infection_number/SUMMARY.md`.

**Plan.** Replace the squared-log-IR incidence term with a **count-based likelihood on
the age-bin case counts**: per-bin Poisson, expected count = `model_IR_bin x observed
person-time_bin`, observed = MAL-ED case counts. Per-bin Poisson factorizes exactly
into a **scale** term (total-count Poisson) x a **shape** term (multinomial over the
age-bins given model-predicted proportions), which (i) directly penalizes putting cases
in the wrong age bin, (ii) weights the sparse 24-35 mo bin (1 case) correctly instead
of letting the log + LOG_EPS over-weight it, and (iii) lets us optionally up-weight the
shape (multinomial) component if level and shape need explicit balancing. The
age-at-first-infection quartile term is kept as a separate added component, as now.
**Decision: this first pass uses pure per-bin Poisson (no free shape-weight)** -- the
cleanest, parameter-free version. If it does not separate the models or fix the level
overshoot, a tunable up-weight on the multinomial shape term is the planned fallback (a
follow-up experiment). Implement this as a new `--fit-target` option in
`calibrate_maled.py` /
`process_incidence_maled.py` (the model and per-rep storage are unchanged), then
re-run **both** models under it -- peaked-age (`age_only`, with the exp-05 peaked seed
re-enqueued) and infection-number -- at 40 trials x 20 reps, Bangladesh demographics,
homogeneous mixing, Erlang-6 maternal, on the covaguest VM. Compare with the same
scorecard + overlay as exp 08, plus the fitted incidence *levels* (did the overshoot
shrink?).

**Success criteria.** Expected: the multinomial term penalizes the infection-number
model's flat young-end shape hard, widening the gap to decisive; and the Poisson scale
term pulls the peaked model's ~3x level overshoot down toward the observed magnitudes
while preserving its correct shape. That would give a clean, level-matched peaked-age
fit and an unambiguous model-selection result. The informative failure modes: if the
infection-number model *also* fits well under the count likelihood, the two mechanisms
are more degenerate than exp 08 suggested and we lean harder on first-infection +
parsimony; or if the peaked model's level still won't come down, the overshoot is
structural (e.g. detection / person-time mismatch) rather than an objective artifact,
which would redirect the next experiment to the observation model.
