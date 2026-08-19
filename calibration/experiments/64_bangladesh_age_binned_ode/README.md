# Exp 64 — Bangladesh: age_binned ODE direct-fit + seed-stability check

**Question.** Repeat India's exp57+58 for Bangladesh under `age_binned`:
fit the ODE+cohort reduction directly against Bangladesh's real MAL-ED
composite likelihood via `differential_evolution`, across 6 seeds, to get
a confirmed-stable pool of high-likelihood draws — the Bangladesh
counterpart to exp58's India ridge. Unlike India, this is NOT a
model-selection step (exp20/28's corrected record: Bangladesh is genuinely
non-identifiable between `age_binned` and `infnum`) — both models will be
fit (this experiment + exp65) and carried forward in parallel, mirroring
this project's own historical "clean same-maternal pair" design (exp20)
but on the new, much faster ODE pipeline.

**Design — reuses exp57/58's exact code and method, changing only the
site:**
- `SITE='bangladesh'` (targets already exist: `maled_data/{first_infection,
  ir_by_age_symp,ir_by_age_all,pt_by_age}_bangladesh.csv`,
  `REPEAT_FRAC['bangladesh']` = 0.403/149).
- Demographics: birth=19, death=6 per 1000 (`SITE_DEMOGRAPHICS['bangladesh']`),
  vs India's 16/7.
- **Titer maternal shape freed** (not fixed at Bangladesh's old ABM-pipeline
  values, median=20/gsd=2.3/half_life=50/hill=4.7) — per AK: comparability
  between sites matters more here than matching the old ABM convention.
  Bangladesh's historical fixed-shape fits (exp25/27) had `base_beta`
  medians of 0.138-0.153, comfortably inside exp57's existing bounds
  (0.05-1.5) — no bounds changes needed.
- Same 12-parameter budget, same BOUNDS, same BDF solver, same per-eval
  timeout guard, same 6 seeds (20260817, 1-5), same subprocess-per-seed
  orchestrator pattern (avoids exp58's open-file-descriptor bug).

**Success criteria.** A stable pool of 6 fitted draws (logL spread
comparable to India's ~0.5 units would indicate a similarly well-behaved
optimum) — the input needed for exp66 (Bangladesh direct-VE ridge
analysis, both models).
