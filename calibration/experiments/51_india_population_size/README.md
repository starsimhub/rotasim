# Exp 51 — India Vellore: does population size alone rescue extinction? (CCS test)

**Question.** exp50 showed 100/100 sims (10 parameter sets spanning a 20x
`base_beta` range) go deterministically extinct within ~2 years at the
40,000-agent population size used throughout this entire India arc (and
every Bangladesh HM experiment, exp16-19, at the identical N — confirmed
via `hm_calibrate.py`'s `N_AGENTS = 40_000`, unchanged since the file's
first commit). AK's hypothesis: this looks like classic **critical
community size** (CCS) behavior from the measles literature — below some
population threshold, a disease can't sustain transmission through the
trough after an initial wave, regardless of transmissibility. Does scaling
up population size alone (parameters held fixed) change the outcome?

**Design.** Two parameter sets from exp50, chosen for a clean test:
- `orig_idx=2320` (`base_beta=0.050`, the lowest beta tested — the mildest
  initial burn (35% attack rate) and thus the most "SIR-like"/most likely
  candidate to benefit from a bigger susceptible reservoir).
- `orig_idx=1720` (`base_beta=0.076`, the point with the longest observed
  survival in exp50 — median 1.12y, max 2.11y — the most "marginal" of the
  10 tested).

For each, run at **N = 100,000 / 200,000 / 400,000** (in addition to the
already-available N=40,000 baseline from exp50, 100% extinct), **10 seeds
each** — 60 new simulations total. Same model otherwise (`age_binned`,
`--fix-age-psymp`, titer maternal) — population size is the only new
variable.

**Note on interpretation:** these two points were originally selected in
exp50 by spreading across `base_beta` alone, not by proximity to the real
(multi-dimensional) viable corridor — so a null result here (still 100%
extinct even at 400k) wouldn't rule out CCS, it would just mean these two
specific parameter combinations are far enough from viable on other
dimensions too. A positive result (extinction rate drops with N) would be
clean, sufficient evidence for a population-size effect on its own.

**Run:** on zebra, same `hm_calibrate.py`/`calibrate_maled.py` pipeline,
`n_agents` overridden per batch instead of the usual fixed 40,000.

**Success criteria:** does extinction rate (out of 10 seeds) drop as N
increases from 40k -> 100k -> 200k -> 400k, for either or both parameter
sets? Does time-to-extinction (for runs that still go extinct) increase
with N even if the binary outcome doesn't flip? Either would support a CCS
interpretation; no change across a 10x population range would argue against
population size being the binding constraint for these specific points.
