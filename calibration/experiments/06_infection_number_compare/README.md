# Exp 06 — Infection-number-only symptoms (same setup as exp 05): can it make the peak?

**Question.** exp 05 showed a peaked age-symptom curve + strong maternal +
homogeneous mixing reproduces the MAL-ED shape. The project's central question is
whether age-based symptom severity is actually *needed* over the canonical
infection-number model. This experiment runs the **infection-number-only** symptom
model under the **identical** setup (homogeneous mixing, strong Erlang maternal) and
asks: can it produce the low-`<6m` / 6–11 mo-peak / declining shape, or not? See
`../05_nonlinear_age_symptom/SUMMARY.md`.

**Plan.** Reuse `_run_one_replicate` exactly as in exp 05 but with
`symptom_model='infection_number'` (per-infection probabilities `p_symp_1/2/3plus`,
no age term), same strong maternal (eff 0.95, Erlang n=6, mean 200 d) and homogeneous
mixing. Forward-run a couple of `p_symp` sets (first-infection-mostly-symptomatic;
graded) × `base_beta`. Plot symptomatic IR by age vs data; assess the shape. No
calibration — this is the matched feasibility comparison to exp 05.

**Success criteria.** This is a *comparison*, so either outcome is informative. If
infection-number-only also reproduces the peak shape → the two mechanisms are
degenerate at the shape level and we lean on calibration + the held-out
first-infection target to distinguish them. If it does *not* (expected from exp 04 —
first infections cluster but the symptomatic profile flattens/over-weights early and
late bins) → age-based symptom severity is genuinely needed, the project's headline
result. Either way, both models get calibrated next for the quantitative comparison.
