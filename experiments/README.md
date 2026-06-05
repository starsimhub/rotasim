# MAL-ED Calibration — Diagnostic Experiments

Work on branch `dec_calibration_akraay_dk` (safe local copy of Alicia's `dec_calibration_akraay`).

**Goal.** A series of quick diagnostic experiments to assess structural issues in the MAL-ED calibration before resuming parameter search. Five questions, in dependency order: demographics → burn-in → timestep → network → censoring.

**Context.** Alicia's calibration fits rotasim (Starsim ABM) to MAL-ED Bangladesh and Pakistan age-stratified rotavirus incidence (IR by age, first-infection quartiles). The model uses 100k agents, dt=1 day, 10-year runs with a 5-year burn-in. Identified structural concerns: UK age distribution used to initialize a South Asian sim, random network with no age structure, and a first-infection GOF that ignores right-censoring.

**Note.** This calibration work lives inside the model repo by choice (branch off Alicia's work). A dedicated calibration repo would be cleaner for the long run.

## Experiments

| # | Question | Status |
|---|---|---|
| 01 | Is the modelled age distribution at demographic equilibrium during the calibration window? | open |
| 02 | Is 5 years of burn-in sufficient, or does the transient extend into the science window? | pending |
| 03 | Does dt=2 days give meaningfully different results from dt=1 for these age-bin targets? | pending |
| 04 | Does a MixingPools age-structured network change infant incidence vs RandomNet? | pending |
| 05 | How large is the censoring bias in the first-infection GOF? Does a KM-based target fix it? | pending |
