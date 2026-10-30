# Exp 04 — Age-structured contacts: can a peaked force of infection reproduce the data shape?

**Question.** `03_model_verification` showed (no bug; clean) that homogeneous
mixing produces a *flat* infection-age profile, which cannot be reshaped into the
data's sharp 6–11 mo symptomatic peak by the symptom/maternal mechanisms exhausted
in `02_joint_identifiability` (0/250 fits). This experiment replaces homogeneous
mixing with an **age-structured contact** pattern — infants preferentially exposed
by slightly older children — to concentrate the force of infection, and the
first-infection peak, into 6–11 mo, then tests whether it reproduces the full
MAL-ED profile that homogeneous mixing could not. See
`../03_model_verification/SUMMARY.md`.

**Plan.**
1. **Implement a *parsimonious* age-structured contact structure** (NOT a free
   contact matrix — that reintroduces the overfitting risk we've guarded against).
   A small number of age bands with mixing controlled by ~1–3 parameters (e.g.
   within-group contact rate plus one infant↔young-child coupling / assortativity
   term), replacing `ss.RandomNet(n_contacts=7)`. Exact starsim network mechanism
   TBD — will confirm the available age-mixing/contact-matrix option before coding.
2. **Feasibility / coverage check FIRST** (workflow step 3, before any calibration):
   can the age-structured model, at *some* parameters, produce a **peaked**
   all-infection (and symptomatic) incidence at 6–11 mo — the thing homogeneous
   mixing demonstrably could not? Prior-predictive over the contact + existing
   params. Do not calibrate until this passes.
3. **If feasible, calibrate and re-check the full profile.** Compare a calibrated
   age-structured model's full-profile fit (all 4 IR bins + first-infection median)
   against the exp-02 baseline — does it get into the data corner that 5 families
   missed?

**Success criteria.** *Feasibility* — the age-structured model produces a 6–11 mo
incidence peak (which homogeneous mixing could not). *Fit* — a calibrated
age-structured model reproduces the full MAL-ED profile materially better than
exp 02's 0/250, with **few** added contact parameters (parsimony; no overfitting).
*Failure* — even age-structured contacts can't make the peak, or only with
implausibly many parameters → revisit symptom-curve flexibility / likelihood
(exp 03's other open thread) rather than mixing. This keeps the age-vs-infection-
number immunity question uncontaminated (additive age-on-susceptibility was
rejected earlier as confounding).
