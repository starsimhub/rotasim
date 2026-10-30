# Exp 65 — Bangladesh: infnum ODE direct-fit + seed-stability check

**Question.** Repeat India's exp59 for Bangladesh under `infnum` (p_symp
keyed by infection order, not age): fit the ODE+cohort reduction directly
against Bangladesh's real MAL-ED composite likelihood, across the same 6
seeds as exp64, producing the infnum counterpart to exp64's age_binned
pool. Runs alongside exp64, not in competition with it — Bangladesh's
corrected model-selection record (exp20/28) is genuine non-identifiability
between the two structures, so both are carried forward per AK.

**Design.** Identical to exp64 except the symptom-model structure:
`p_symp_order1/order2/order3plus` (cascading ratios, same convention as
`sus_r2`/`sus_r3`) instead of age-bin lookups, using exp59's infnum
cohort_model.py unchanged. Same site (`SITE='bangladesh'`, demographics
19/6), same freed titer shape, same 12-parameter budget, same BOUNDS/BDF
solver/timeout guard/6 seeds/subprocess-per-seed pattern.

**Success criteria.** A stable pool of 6 fitted draws, directly comparable
to exp64's age_binned pool on logL (per exp20/28's precedent, expect infnum
to show equal-or-better ESS/logL behavior than age_binned for Bangladesh,
unlike India where age_binned won decisively) — the second input needed
for exp66 (Bangladesh direct-VE ridge analysis, both models).
