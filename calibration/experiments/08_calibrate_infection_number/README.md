# Exp 08 — Calibrate the infection-number-only model to MAL-ED (comparison baseline)

**Question.** This is the matched comparison to exp 07. Exp 06 showed
*qualitatively* that the canonical infection-number-only symptom model cannot make
the 6–11 mo peak (its profile is always `<6m`-highest and declining). Calibrating it
to the same targets, under the same setup as the peaked-age model, puts a *number* on
how badly it fits — the quantitative model-selection gap the researcher wants
alongside exp 07. See `../06_infection_number_compare/SUMMARY.md` and
`../07_calibrate_peaked_age/README.md`.

**Plan.** Same calibration machinery, demographics, mixing, compute, and joint
objective as exp 07 — only the symptom model differs:
`calibrate_maled.py --symptom-model infection_number --maternal-n-stages 6
--fit-target joint`, Bangladesh demographics, homogeneous mixing, covaguest VM, 100k
agents, 40 trials × 20 replicates, n_jobs=1. The open symptom parameters here are the
per-infection probabilities (`p_symp_1/2/3plus`) with no age term; everything else
(base_beta, monotone susceptibility, maternal efficacy/duration) shares exp 07's wide
unbiased ranges. No seed enqueued — this model has no peak to seed toward.

**Success criteria.** This is a *comparison baseline*, so a poor fit is the expected
and informative outcome: we expect the best infection-number fit to still get the
shape wrong (`<6m` too high, no 6–11 mo peak) and so a worse joint GOF than exp 07. If
instead it fits the shape well, the two mechanisms would be degenerate at the data
level and we'd lean on the held-out first-infection target and parsimony to choose —
also informative. The deliverable is the head-to-head GOF + best-fit-vs-data overlay
against exp 07.
