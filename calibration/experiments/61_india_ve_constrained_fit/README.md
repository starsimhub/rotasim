# Exp 61 — India Vellore: joint natural-history + direct-VE fit (age_binned)

**Question.** exp42 tried adding VE as an explicit HM scoring target on top
of the ABM's cohort-fit posterior and found it only relocated the tension
(pulling toward VE-plausibility cost the cohort fit on IR6-11m and Q25,
exp39's two already-weakest dimensions) rather than resolving it — but that
was importance-reweighting a FIXED, already-degenerate NROY pool under a
noisy stochastic ABM (ESS dropped further, 9.15→4.60). exp60 confirmed the
underlying reason a fixed reweight can't fix this: `sus_r2`/`sus_r3` (which
direct VE runs through) are genuinely ridge-degenerate against the 0-35mo
cohort data alone. Now that the ODE pipeline can jointly OPTIMIZE (not just
reweight a fixed sample) both natural history and VE in the same
deterministic search, in ~5 minutes instead of days, does a region exist
that satisfies both simultaneously — succeeding where exp42's ABM-based
reweighting failed — or is the trade-off real regardless of method?

**Design.** Add one term to exp57/58's exact 12-parameter composite
log-likelihood: a Gaussian direct-VE term,
`ll += -0.5*((VE_model(take=0.74) - 0.59) / 0.054)**2`,
computed via exp60's cohort-model vaccine mechanism (Rotavac 3-dose,
6/10/14wk, direct/low-coverage design) using the SAME `foi_eq` the natural-
history side of the likelihood already computes each evaluation — no
separate pool, no reweighting, genuinely joint.

- **Target 0.59, sigma 0.054**: Nair et al.'s 6-11m test-negative VE point
  estimate with its 95% CI (47-68%) converted to a Gaussian sigma
  (`(0.68-0.47)/(2*1.96)`). This is the same target exp41/60 used (direct/
  individual VE, test-negative design — confirmed with AK this is
  coverage-independent as an estimand, unlike the population-impact target
  exp42 used).
- **take=0.74 fixed, not a free parameter**: keeps this a clean, direct
  extension of exp57/58's exact 12-param space (no bounds/identifiability
  questions for a 13th parameter to sort out first) and matches the take
  value exp42 and other prior India VE work already used.
- **6 seeds**, same as exp58/59, to check the joint optimum is itself stable
  before reading anything into it.

**Plan.**
1. New `run.py`/`run_one_seed.py` (mirrors exp58's structure): combine
   exp57's `composite_logL` with exp60's vaccine-cohort machinery, add the
   VE Gaussian term, re-run `differential_evolution` × 6 seeds.
2. At each seed's joint-best point, report BOTH the joint logL and the
   natural-history-only component (recomputed without the VE term) —
   the number that answers "how much, if any, natural-history fit is
   sacrificed to satisfy VE" — directly comparable to exp58's -284.22 to
   -284.67 range.
3. Check whether the 6 seeds' joint-best points cluster in `sus_r2`/`sus_r3`
   space (a real narrowing of the ridge) or scatter across it the way the
   unconstrained fit's *population* does (VE picked one point but didn't
   actually constrain the family) — this is the actual test of whether VE
   is a useful extra identifiability constraint, not just whether one
   run happens to find a compromise point.

**Success criteria.** Best case: joint-best natural-history component stays
close to -284.2 to -284.7 (no real sacrifice) AND `sus_r2`/`sus_r3` cluster
tighter across seeds than exp58's unconstrained ridge did — VE successfully
narrows the ridge for free. Failure mode matching exp42: natural-history
component degrades noticeably to hit VE, or the "joint-best" points still
scatter across sus_r2/sus_r3 as much as before (meaning many ridge points
are compatible with the VE target too — it's not actually identifying,
just satisfiable everywhere).

**Status:** design drafted, not yet run — confirming target/sigma/take
choices with AK before spending zebra time on 6 seeds.
