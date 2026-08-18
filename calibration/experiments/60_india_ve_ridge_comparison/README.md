# Exp 60 — India Vellore: direct-VE sensitivity across age_binned's fitted ridge

**Question.** exp58 found that `log_base_beta`/`sus_r2`/`sus_r3` sit on a
compensating ridge — many different combinations fit the 0-35mo natural-history
data (MAL-ED IR-by-age, repeat_frac, Q25) equally well. Does forward-predicted
**direct vaccine efficacy**, at a FIXED true per-dose take, vary meaningfully
across that ridge? If different (beta, sus_r2, sus_r3) triples that are
observationally indistinguishable on natural history give very different
achieved VE, that's an identifiability problem specifically for VE-forward-
prediction — a sharper, more consequential version of the 36m+
compartment-fraction sensitivity exp58 already found for ABM initialization.

**Vaccine mechanism — mirrors the ABM's `VaccinePrime` exactly (found in
`calibrate_maled.py:101`).** A `coverage` fraction of the cohort receives each
dose at fixed ages; among those, a `take` (`response_prob` in the ABM)
fraction seroconvert and get `num_recovered_infections += 1` — i.e. a
successful dose is mechanically identical to a prior infection, advancing
susceptibility order by one (`moa='infection_blocking'`). This maps directly
onto the cohort ODE's existing order axis: at each dose age, transfer a
`coverage × take` fraction of the cohort from order `j` to `j+1` (same
transformation as the existing `R_j → S_{j+1}` recovery transition), applied
proportionally across all phase/detected-count sub-compartments so as not to
disturb the detection bookkeeping. Implemented by solving `cohort_model.py`'s
ODE in segments between dose ages and applying the transfer at each boundary,
rather than as a continuous rate.

**Design choice (per AK): direct VE only, low coverage** — isolates the
individual-level mechanism (order-jump interacting with `sus_after_1/r2/r3`)
from herd effects, matching exp41's direct-VE design and giving the sharpest
test of the ridge-sensitivity hypothesis. Dose schedule: Rotavac 3-dose
(6/10/14 weeks), matching exp41/AK's India-specific VE work.

**New piece of code needed (small, one addition):** the cohort model
currently only tracks *total* detected-infection count (`d`, used for
`repeat_frac`/the survival curve), not specifically *symptomatic*-detected
count — but VE needs to compare symptomatic case rates between arms (matching
Nair et al.'s test-negative VE, which is against RVGE specifically). Adding
one scalar accumulator state to `cohort_model.py`,
`d(cum_symp)/dt = Σ_j [foi·σ_j·S_j] · p_symp(age or order, j) · DET_SYMP`,
using exactly the same per-order incidence terms `rhs_cohort` already
computes each step — not a new mechanism, just one more output tapped off
the existing dynamics.

**Plan.**
1. Extend `cohort_model.py` with the `cum_symp` accumulator and a
   dose-schedule-aware `simulate_cohort_vaccinated` (or a `doses` argument to
   the existing function) that solves in segments and applies the order-jump
   transfer at each dose boundary.
2. For each of exp58's top-10 pooled high-likelihood draws (same draws used
   in exp58's equilibrium re-check, so this reuses an already-validated,
   confirmed-stable set): run the cohort forward twice from the same
   `foi_eq` — once unvaccinated, once with doses (coverage=0.05, to match
   exp41's direct-effect isolation) — at a small sweep of `take` values
   (e.g. 0.6 / 0.74 / 0.9, spanning the UK/India take estimates already used
   elsewhere in this project).
3. Compute direct VE = 1 − (vaccinated `cum_symp` rate) / (unvaccinated
   `cum_symp` rate) in the 6-11m window (the real-world anchor: Nair et al.
   52-59%, AK's ecologic estimate 52.4%), across all 10 draws × 3 take
   values.
4. Report the spread of achieved VE across the 10 draws at each fixed take —
   the quantity that answers the actual question — not just each draw's
   point estimate.

**Success criteria.** If achieved VE varies only a little across the 10
ridge draws at fixed take (say, within a few percentage points), the ridge
is a natural-history curiosity that doesn't propagate to the decision-
relevant quantity. If it varies a lot (e.g., spans from clearly-too-low to
clearly-too-high relative to the real 52-59% anchor depending on which ridge
point is used), that's an important, actionable finding: natural-history
model selection alone isn't sufficient to pin down VE forward-predictions
for this population, and the VE-scoring approach exp42 tried (but which
degraded the cohort fit) may need revisiting with the much cheaper ODE
pipeline instead of the expensive ABM.
