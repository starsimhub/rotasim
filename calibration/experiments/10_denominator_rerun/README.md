# Exp 10 — Re-run exp 09 under the corrected person-time denominator (+ infra fixes)

**Question.** Exp 09 used the end-of-sim person-time *snapshot* (`final_headcount x
window`) as the IR denominator, which overstates older cohorts by up to ~16% in a
growing population (D. Klein's exp 01; ~0 in `<6m`, growing with age). We've adopted
his `PersonTimeByAge` accumulator (count x dt over living agents, exact under growth)
and the starsim#1343 memory-leak mitigation. Does the exp-09 result — peaked age model
wins decisively under the shape-aware Poisson objective; level, shape, and
age-at-first-infection all matched — **hold under the corrected denominator**? See
`../09_shape_aware_likelihood/SUMMARY.md`.

**Plan.** Identical to exp 09 — both models (`age_only` peaked, `infection_number`),
Erlang-6 maternal, `--fit-target poisson`, 40 trials x 20 reps, Bangladesh
demographics, homogeneous mixing, covaguest — the *only* change is the corrected
denominator (`PersonTimeByAge`) plus the memory fixes (`sim.shrink` + worker
recycling). Fresh study DBs (`*_ptfix.db`) so old- and new-denominator trials never
mix. Peaked seed ladder re-enqueued (same as exp 09). Reporting still via
`process_model` (DK's `MALEDTargets` reporter is ported but supports `age_only` only,
so it can't yet serve the infection-number arm — deferred). Compare the new fits and
GOF head-to-head with exp 09.

**Success criteria.** The expected, reassuring outcome: the corrected denominator
shifts the older-bin (`12-23`, `24-35 mo`) model IR up modestly and may nudge the
`sus_after_3plus` rung, but the **headline holds** — peaked still beats
infection-number decisively (qualitative peak-location difference), and the peaked
best-fit still matches level / shape / first-infection. The informative failure: if the
conclusion flips or the peaked fit degrades materially, the denominator was doing
hidden work and we re-examine the immunity ladder. Either way exp 09 (old denominator)
remains the committed baseline for the comparison.
