# Exp 02 — Joint identifiability on the optimized fits: does the achievable frontier exclude the data corner?

**Question.** `01_coverage_check` (see `../01_coverage_check/SUMMARY.md`) was the
wrong instrument — random wide-prior draws can't reach calibrated quality, so
they can't settle whether the model can *jointly* fit symptomatic-IR-by-age and
age-at-first-infection. This experiment asks the question with the right
instrument: the **optimized** fits we already have. Across all completed TPE
studies, is there a fundamental tradeoff such that **no** parameterization
reproduces the data's shape on both targets at once — specifically the 6–11 mo
symptomatic peak, the near-zero 24–35 mo bin, and the correct first-infection
median? If the achievable frontier excludes the data corner, the Pareto tension
is structural (rigorously, on optimized fits); if some fit reaches the corner,
the earlier tension was a per-study local-optimum artifact.

**Plan.** No new simulations (or at most a small targeted confirmation). Pull
every completed trial from the existing study DBs — the Erlang studies
(`offsets` and `infnum_erlang6`) carry per-rep IR-by-age + first-infection in
`user_attrs`; the single-phase and two-phase baselines come from `user_attrs`
or the calibration logs. For each trial compute (a) `gof_inc` and `gof_first`,
and (b) the interpretable target features: 6–11 mo IR, 24–35 mo IR, and
first-infection median. Plot the **achievable Pareto frontier** two ways —
(1) `(gof_inc, gof_first)` with the data corner at the origin, and (2)
interpretable feature space (e.g. simulated 6–11 mo peak vs simulated
first-infection median) with the MAL-ED data point and a tolerance box — pooled
across all studies, colored by model family. Assess whether any optimized fit
lands near the data corner / inside the tolerance box.

**Success criteria.** *Frontier excludes the data corner* (no optimized fit
simultaneously near the 6–11 mo peak, the low oldest bin, and the right
first-infection median) → the tension is **structural**, established on the
fits the optimizer actually found; next experiment (03) tests a structural
change — age-modulated protection against *infection* (the original
hypothesis). *Some fit reaches the corner* → the tension was a search/
local-optimum artifact, not structural; next experiment re-runs that region
with a count-based likelihood and tighter search. Either way this replaces
exp 01's inconclusive verdict with a decision-grade one.
