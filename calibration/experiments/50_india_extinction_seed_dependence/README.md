# Exp 50 — India Vellore: is extinction (or time-to-extinction) seed-dependent?

**Question.** Every India HM run (exp39-49) classifies a parameter point as
"extinct" from a single simulation seed. A quick 8-seed check on one
parameter set (posterior row 0, fixed params, seeds 1000-1007) already
showed all 8 seeds go extinct, but time-to-extinction ranged from **1.86 to
9.59 years** — a huge spread for the *identical* parameters. That single
check also revealed something more fundamental: the model's initial
seeding (4% of the population, uniform across ALL ages 0-100, well-mixed
network) triggers a near-total synchronized outbreak (~90% of the
population infected by day 12) before any steady-state dynamics take over —
"extinction" is really about whether a residual transmission chain survives
years after that initial sweep, not a low endemic equilibrium fizzling out.

This experiment asks properly: across a genuinely diverse set of
parameter points that were classified extinct in exp39's posterior, how
much does the extinction outcome (and time-to-extinction, for those that do
go extinct) vary across independent seeds? If a substantial fraction of
"extinct" parameter points are actually only extinct on SOME seeds (not
all), that's direct, decisive evidence that single-seed classification is
misclassifying genuinely viable parameter regions — strengthening the case
for exp48's classifier-based fix (or an even more direct multi-seed
extinction score).

**Design.** 10 parameter sets selected from exp39's NROY pool
(`nroy_draw.csv`), all classified extinct in the original single-seed
evaluation, chosen to span nearly the full `log_base_beta` range explored
(from ~0.05 to ~1.03 on the natural scale) — not just 10 near-identical
points. Original NROY indices: `[2320, 1720, 852, 1454, 2287, 2306, 2538,
1841, 104, 1414]`. For EACH of these 10 parameter sets, run **10 fresh
seeds** (independent of the original NROY seed) — 100 simulations total,
40k agents each, `age_binned` + `--fix-age-psymp` (matching exp39's exact
model configuration, the only variable is: same params, different seeds).

For each of the 100 runs, records: extinct (bool, `ir_sum<=0`), time of
last active infection (years), time and size of the initial peak, using
the new `n_infected_series` field added to `_run_one_replicate`'s
cohort-branch output (population-level `n_infected` time series, one entry
per simulated day — diagnostic only, ignored by the GOF/HM path).

**Success criteria / what we're looking for:** for each of the 10 parameter
sets, the fraction of its 10 seeds that go extinct. If that fraction varies
widely across parameter sets (some near 0/10, some 10/10) rather than being
uniformly high, that's evidence some "extinct" classifications in exp39-49
were wrong calls driven by bad luck on a single seed. Also report the
spread of time-to-extinction among seeds that do die out, for each
parameter set, to characterize how sensitive that timing is.
