# Exp 66 — Bangladesh: direct-VE ridge analysis, both models

**Question.** Mirrors India's exp60: does direct/individual VE vary
substantially across each model's 6 equally-good fitted draws (exp64
age_binned, exp65 infnum)? Bangladesh's ridge is broader than India's on
`log_base_beta` specifically (found while reviewing exp64/65: 7x range for
infnum, 3.3x for age_binned, not just the expected `sus_r2`/`sus_r3`
degeneracy) — so this is checking whether that translates into even bigger
VE swings than India's 29-point spread, for both structures.

**Design.** Reuses exp60's exact vaccine mechanism (order-jump dose model,
Rotavac 3-dose at 6/10/14 weeks, `coverage=1.0` within the simulated
vaccinated arm to isolate direct/individual effect) for `age_binned`
(`age_binned_cohort_model.py`, unchanged copy of exp60's) and a matching
infnum version (`infnum_cohort_model.py`, order-keyed p_symp + the same
dose/cum_symp mechanism, already built and validated during the earlier
quick UK-vs-India check). Per AK: Rotavac's real schedule and take values
carried over unchanged from India (Bangladesh hasn't introduced the
vaccine, so there's no real-world Bangladesh schedule/coverage to use
instead). Take sweep: 0.6/0.74/0.9 (the same three values used throughout
India's VE work, for direct comparability). Uses each model's own 6 real
fitted seed draws directly (not a synthetic sus_r2/sus_r3-only grid) — so
the full multi-parameter ridge found in exp64/65 (including `log_base_beta`)
is automatically reflected, not just the two susceptibility ratios.

**Success criteria.** Compare each model's direct-VE spread (at fixed
take) across its own 6 draws to India's exp60 finding (39.8-68.5% at
take=0.74, a 28.7-point spread). Also worth comparing age_binned vs infnum
directly at the same take, mirroring the historical exp20 comparison
(which found infnum's VE ~1.7x higher than age's) — now via direct
optimization instead of ABM/HM reweighting.
