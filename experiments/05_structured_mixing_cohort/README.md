# Exp 05 — Generalizing Mechanism: Infection-Number Severity + Maternal + Structured Mixing (Bangladesh)

**Question.** Can we jointly fit MAL-ED **Bangladesh** — the 6-11mo symptomatic IR
peak, the age-at-first-detection (KM) timing, AND the ~10% repeat-infection fraction
— using a symptom mechanism *chosen to generalize across the FOI gradient*, rather
than a Bangladesh-tuned age-symptom curve? Fit one site now; the cross-site (UK
low-FOI) check is banked for after a single-site fit works.

**Why this mechanism (the key design decision).** The age of rotavirus disease is
not fixed — it tracks force of infection (UK low-FOI peaks at 2-5yr; Bangladesh
high-FOI peaks in year 1; global review: 38→65 wk across the mortality gradient). A
free *age*-symptom curve tuned to Bangladesh's 6-11mo peak fits one site but cannot
transfer — almost certainly the original "structural age issue." The mechanism that
generalizes is an **intrinsic per-infection conditional**, where the cross-site age
shift emerges from FOI-driven *timing*, not per-site re-tuning:

- **Symptom severity by infection-number (B):** first infection most likely
  symptomatic, later ones milder (acquired immunity reduces severity) —
  `p_symp_1 ≥ p_symp_2 ≥ p_symp_3+`. A *first* infection is symptomatic at any age,
  which is why low-FOI settings still get symptomatic disease at 2-5yr (UK's 47% of
  cases there) — natural under (B), awkward under sharp age-severity (A).
- **Strong Erlang maternal immunity:** sets the *floor* age by delaying the first
  infection past ~6mo (resolves Alicia's exp 06, where (B)-alone peaked at <6mo
  because first infections landed too early).
- **Age-structured low-infant-exposure mixing** (Alicia's 3-group `MixingPools`):
  controls infant FOI → the *timing* of that first infection (high reservoir among
  toddlers, low spillover to infants).

The 6-11mo peak then emerges from timing (maternal floor + infant FOI) × severity-on-
first-infection — a combination that should slide later automatically at lower FOI.

**Plan.**
- Network: `ss.MixingPools`, 3 groups (infants 0-1 / young 1-5 / rest 5+); levers
  `young_reservoir` (high), `infant_exposure` (low, key), `cross_contacts` (bg).
- Symptom model: **infection-number** in `MALEDCohort` (track per-child infection
  order; `p_symp_1/2/3+`). Replaces the age-symptom logistic.
- Immunity: Erlang maternal (`efficacy`, `mean_duration`, `n_stages=6` sharp) +
  acquired ladder `sus_after_1/2/3+`.
- Observation: cohort + schedule-based detection + individual data-driven dropout.
- Targets (joint): symptomatic IR-by-age (6-11mo peak), KM first-detection curve,
  repeat-infection fraction (~10%).
- Pop: 40k agents. Compute: ~1000 draws on `capybara`. First a small **β/contact
  range-check pilot** (the MixingPools FOI scale differs from RandomNet).

**Success criteria.**
- Good: a parameter region jointly hits the 6-11mo IR peak, the KM timing, AND
  repeat ~10% (±few pp) at plausible prevalence — a single-site fit exists with a
  generalizing mechanism. Then (next experiment) verify it slides to UK's later peak
  at UK-FOI.
- Informative failure: even (B)+maternal+structured mixing can't hit ~10% repeats
  while making the peak → the residual is acquired-immunity *durability* (how much
  reinfection is suppressed), and exp 06 opens that up.
- Watch: degeneracy between `infant_exposure`, maternal duration, and `p_symp_*`;
  whether 40k pins the repeat fraction at low FOI.
