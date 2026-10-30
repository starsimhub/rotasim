# Exp 62 — India Vellore: population-level vaccine impact vs take, across the ridge

**Question.** exp60/61 measured *direct* (individual-level) VE and found it's
sensitive mainly to `sus_r2` (the order-1→2 transition, the one Rotavac's
3-dose schedule directly acts on in young infants). But `sus_r3` (order-2→3+)
governs the susceptibility of the ~80-95% of the population that sits in the
older, repeat-exposed compartment at equilibrium (exp55/56) — the population
that actually carries ongoing transmission. **Population-level impact**
(herd effects, how much vaccination suppresses transmission at realistic
coverage, whether higher take buys meaningfully more impact as it climbs
toward 95%) should therefore depend heavily on `sus_r3` — a parameter
exp61 could *not* constrain (if anything it got less constrained under the
VE-joint fit). This experiment tests that directly: how much does predicted
population impact vary across take (up to 95%) and across the ridge, and
is `sus_r3` really the dominant source of that variation?

**Why this needs new mechanism, not a reuse of exp60/61.** exp60/61
deliberately used the *birth-cohort* model at low/isolated coverage
specifically to avoid herd feedback — a single cohort's vaccination status
has negligible effect on the population's shared FOI, so `foi_eq` could be
computed once (unvaccinated) and reused for both arms. Population impact is
the opposite question: it requires sustained vaccination to change the
population's OWN equilibrium FOI. That means extending the *age-structured
population* ODE (`ode_model_age.py`, exp55/56/57's `simulate_age`), not the
birth-cohort model.

**Mechanism — revised per AK (2026-08-19): resolve the actual dose schedule,
not one lumped boundary jump.** A single "coverage × take advances one order"
transformation applied once at the `<6m → 6-11m` boundary would miss the real
protection ramp-up (or lack of it) *during* the 0-6 month window — and
`<6m` incidence is itself one of this arc's key targets, so getting the
within-window dynamics right matters here specifically. Also, per AK's
prior VIMC modeling work: per-dose coverage cascades (each successive dose
reaches a smaller fraction than the last — you can't get dose 2 without
dose 1), not one flat number applied identically to all 3 doses.

Concretely:
- **Split `<6m` into 4 sub-bins at the real dose ages**: [0,6wk), [6,10wk),
  [10,14wk), [14wk,26wk). The other four bins (6-11m, 12-23m, 24-35m, 36m+)
  are unchanged.
- **Add a "doses-received-so-far" axis (0/1/2/3) to just these 4 sub-bins**
  — collapsed (summed away) once population flows into 6-11m, since only
  the cumulative immunity-order effect matters after that, not dose
  history itself.
- **Per-dose coverage**, AK's independence rule: given a target full
  (3-dose) coverage `C3` (e.g. NFHS-5 Tamil Nadu's 66.4%), `C2 = sqrt(C3)`,
  `C1 = sqrt(C2)`. At the age boundary for dose *k*, only the sub-population
  currently at dose-level *k-1* is eligible (dose-2 eligibility requires
  having received dose 1, not just being the right age); the conditional
  fraction receiving it is `C1` (dose 1, applied to everyone), `C2/C1`
  (dose 2, applied only to dose-1 recipients), `C3/C2` (dose 3, applied
  only to dose-2 recipients). Of those receiving a dose, `take` fraction
  also advance one susceptibility order (seroconvert, mirroring exp60/61's
  `_apply_vaccine_dose`); `(1-take)` receive the dose but get no
  immunological credit — same non-maternal-compartment caveat as
  exp60/61 (the order-jump only touches S/IS/IA/R, not the M chain).
- Run to a new 60-year equilibrium under sustained vaccination (same
  procedure as `simulate_age` today, with this modified fine-bin/dose-axis
  structure for `<6m`) and compare symptomatic incidence-by-age against the
  unvaccinated equilibrium.

**Design.**
- **Take axis**: sweep 0.6 (India's established LMIC Rotarix/Rotavac
  anchor — see exp30/41, NOT a value borrowed from UK) up to 0.95, in
  reasonable steps (e.g. 0.6, 0.7, 0.8, 0.9, 0.95).
- **Coverage**: `C3` fixed at a realistic value for the primary sweep — 66%
  (NFHS-5 Tamil Nadu dose-3 coverage, already used in the India VE arc,
  see project memory), with `C1`/`C2` derived via the cascade rule above —
  not swept in this first pass, to keep the design focused on the
  take/ridge question AK asked about. Flag coverage as a natural follow-on
  sweep if this run shows something worth exploring further.
- **Ridge**: run every (take) point across exp58's 6 confirmed-stable
  age_binned seeds (not just one point), to get the uncertainty band the
  direct-VE work already showed matters.
- **Outcome**: population-level (total-effect, herd-inclusive) symptomatic
  IR-by-age at the new vaccinated equilibrium vs the unvaccinated
  equilibrium, summarized as population-impact VE per age bin, across
  take × ridge draw.

**Success criteria.** If population impact's spread across the 6 ridge
draws is comparable to or wider than direct VE's (exp60's 29-point spread),
that confirms `sus_r3` (or the ridge generally) matters at least as much for
population-level questions as it did for individual-level ones — the more
consequential, decision-relevant version of exp58's finding, since
population impact is what a real vaccination-program decision would
actually be based on. If it's narrower, that's equally worth knowing (would
mean the herd/equilibrium averaging washes out some of the individual-level
ridge sensitivity).

**Status:** README revised (2026-08-19) to resolve the actual dose schedule
per AK's feedback; building now. AK will review this README again before
the run is launched.
