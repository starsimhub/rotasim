# Exp 63 — India Vellore: population impact vs take, across coverage and FOI scenarios

**Question.** exp62 fixed coverage at 66.4% (NFHS-5) and found population
impact rises steadily with take, with the ridge (`sus_r3`, unconstrained by
either the natural-history or VE data) creating real but bounded uncertainty
in the achieved number. Two follow-on questions, both motivated by the EDGE
team's interest in whether a higher-`take` next-generation vaccine would be
worth developing: (1) does the take→impact "slope" — how much a `take`
improvement is actually worth — change at different coverage levels (their
lever is `take`; coverage is a health-system lever largely outside their
control, but interacts with it)? (2) separately, how would impact shift
under a lower-FOI counterfactual (also outside EDGE's control, but a
plausible future if WASH/other conditions improve)?

**Design.**
- **Take grid**: finer than exp62's 5 points — `[0.5, 0.55, ..., 0.95]`
  (10 points), for a clean interpolated threshold estimate (see below), not
  just a visual slope.
- **Coverage levels**: 0.40 / 0.664 / 0.85 — the low/mid/high convention
  already established in this project's India VE forward-prediction work
  (project memory: "SWEEP coverage... low ~0.40 (early ramp) / mid
  ~0.55-0.66 (NFHS-5) / high ~0.85 (2023/BCG ceiling)"), not new numbers.
- **FOI scenarios**: each ridge draw's own fitted `base_beta` scaled by
  `{1.0, 0.85, 0.7}` (up to a 30% FOI reduction) — a plausible envelope for
  "FOI drifts down for reasons unrelated to this vaccine," not a specific
  prediction.
- **Ridge**: same 6 confirmed-stable age_binned draws as exp58/60/61/62.
- **Total**: 6 ridge × 10 take × 3 coverage × 3 FOI = 540 vaccinated sims +
  18 unvaccinated baselines (one per ridge×FOI combination, coverage/take
  don't affect it) = 558 runs. At ~7.8s/sim serially that's over an hour;
  run via a local `multiprocessing.Pool` (one pool for the whole grid, not
  the repeated-pool-creation pattern that caused exp58's file-descriptor
  bug) to bring it down to single-digit minutes — no need for zebra at this
  size.

**New deliverable per AK (2026-08-19): a 75%-population-VE reference line
and threshold-take calculation.** For every (coverage, FOI-scenario) panel,
draw a dashed horizontal line at 75% population-impact VE, and for each
ridge draw, interpolate across the take grid to find the take value where
population-wide VE crosses 75% (reporting "never reaches 75% in this take
range" if it doesn't). Report that threshold take as a function of
coverage, FOI scenario, and — since take grid is shared across ridge draws
but each draw has a different `sus_r3` — **as a function of `sus_r3`**,
directly quantifying how much the "take you'd need" moves depending on
where the true immunity-waning parameter sits.

**Success criteria.** This directly answers the EDGE-relevant question:
under today's coverage/FOI, what take does a next-gen vaccine need to hit
a 75% population-impact target, and how much does that requirement move
if coverage or FOI also improve — and separately, how much does the
*answer itself* move depending on the unresolved `sus_r3` ridge (i.e., is
"you need take=X" a confident statement or a wide range)?

**Status:** not yet implemented.
