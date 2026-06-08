# Exp 01 — Demographics Equilibrium Check — SUMMARY

**Question.** Does the modelled age distribution reach demographic equilibrium
during the burn-in, is that equilibrium plausible for Bangladesh/Pakistan, and
are the person-time denominators used in the calibration correct?

## Result

Two findings, one benign and one a real bug.

**(1) Equilibration is fast and the equilibrium is correct — the UK
initialization is not a problem for infant bins.** Despite initializing from
`uk_age_data.csv` (~94% aged 5+), the population re-equilibrates to the correct
infant fraction within ~1 year — well inside the 5-year burn-in. Equilibrium
<6m fractions match demographic theory:

| Site | model <6m (yr 5-10) | theory (birth_rate × 0.5y) |
|---|---|---|
| Bangladesh | ~0.77–1.01% | 0.95% |
| Pakistan | ~1.17–1.46% | 1.35% |

So burn-in adequacy (exp 02) is **not** a concern for the <36m targets. (The
<36m fraction is still slowly drifting up at year 10 — would matter only for
older-child targets.)

**(2) BUG: person-time denominators are inflated, age-dependently, up to
16%.** `process_incidence_maled.py::compute_person_months_steady_state` uses
`final_headcount × window_months`. Because the population grows (births >
deaths), the end-of-sim snapshot overstates the time-averaged headcount.
Direct measurement (snapshot-PT ÷ true accumulated-PT):

| Site | <6m | 6-11m | 12-23m | 24-35m |
|---|---|---|---|---|
| Bangladesh | 0.99 | 1.00 | 1.05 | 1.16 |
| Pakistan | 0.99 | 1.04 | 1.08 | 1.16 |

The bias concentrates in older infant bins, so it **flattens the modelled
age-incidence gradient** — biasing not just `base_beta` but the immunity ladder
(`sus_after_3plus`). Systematic, not noise.

**Fix built and validated.** Added `rs.PersonTimeByAge` analyzer
(`rotasim/analyzers.py`) that accumulates `count × dt` over the calibration
window — exact regardless of population growth. `validate_fix.py` confirms it
recovers the unbiased denominators. PR-able.

## Figures

![Age distribution trajectory](figures/age_distribution_trajectory.png)

## Observations

- The MAL-ED study followed children ~24 months from birth (BD 23.1mo avg, PK
  25.2mo); the calibration computes IR over a 60-month window. Cohort-vs-cross-
  section mismatch — pinned for a future "mini-MAL-ED in sim" experiment.
- Censoring in the first-infection data is large (44% BD, 64% PK) — raises the
  priority of exp 05.

## Next

- **Exp 02 (burn-in):** mostly answered here for infant bins (fast
  equilibration) — will do a quick confirmation and document, then move on.
- The person-time fix should be wired into the calibration driver (worker adds
  `PersonTimeByAge`, `process_model` reads its `person_months`) — a follow-up
  beyond these diagnostics. Feedback captured in
  `calibration/FEEDBACK_FOR_ALICIA.md`.
