# Exp 37 — UK HM (infnum): VE + case-shape joint calibration; Bangladesh-constrained immunity

**Question.** Exp 28's shape-only calibration is unidentifiable: high-beta/high-sus_after_1
and low-beta/lower-sus_after_1 produce the same case age distribution. Exp 36 showed
`first_inf_median_months` cannot serve as an FOI anchor (metric structurally unreachable at
any viable beta). IgA seroprevalence (Hungerford 2025, 20 U/ml) is non-monotonic across
age groups, indicating it measures transient recent-infection IgA, not cumulative "ever
infected", and cannot be used as a direct ABM target without IgA kinetics modelling.

This experiment adds **vaccine efficacy (VE ≈ 0.74)** as the identifiability-breaking target.
VE directly distinguishes the two degenerate regimes: sus_after_1 ≈ 0.75 (Bangladesh-like)
→ VE ≈ 74%; sus_after_1 ≈ 0.975 (exp28 artifact) → VE ≈ 10%.

**Plan.**
- Two sims per parameter set: no-vaccine run (case shape vs pre-vaccine UK data) + vaccine
  run (VE vs UK Rotarix ~74%). VE and shape are therefore independent targets from separate
  runs. Vaccine config fixed at UK programme values: 2 doses at 2+4mo, 90% coverage,
  response_prob=0.85 (infection-blocking).
- Bangladesh sus_after_1/2/3 constrained to exp27 (infnum, corrected maternal) posterior
  medians ±50%: sus_after_1=[0.376, 0.99], sus_r2=[0.308, 0.923], sus_r3=[0.330, 0.989].
  ±50% allows genuine cross-setting variation while strongly excluding the sus_after→1 artifact.
- Same infnum symptom model and fixed titer shape as exp28/36.
- Initialization override: init_prevalence=0.005, flat age distribution — prevents extinction
  in the low-beta region (fix for exp36's main failure mode).
- 40k agents, 8 free parameters (log_base_beta + 3 sus_after + maternal_efficacy + 3 p_symp).
- 3 HM waves × 2000 samples on zebra.

**Success criteria.**
- Non-empty NROY (ESS > 5) satisfying case shape and VE jointly.
- NROY VE median: 0.65–0.85.
- sus_after_1 constrained well below 0.90 (excluding the degenerate reinfection regime).
- No extinction within NROY (all valid samples produce >= some threshold of total cases).
