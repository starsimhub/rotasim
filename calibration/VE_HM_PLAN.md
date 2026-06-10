# Plan sketch — HM posteriors → two-model vaccine-impact comparison

Forward plan for the VE phase. Goal: two structurally-different, data-consistent models
(age-symptom and infection-number), each as a **posterior** (not a point fit), then add
vaccination and compare VE **distributions** — per setting and across the LMIC↔high-income
FOI gradient. Divergence ⇒ VE depends on unidentifiable structure; agreement ⇒ robust.
Status: sketch only, nothing built. Reuses D. Klein's exp-07 HM infrastructure.

## Why HM, not the Optuna point fits
A point fit gives ONE VE number per model — can't tell a structural VE difference from
parameter noise. HM → posterior gives a VE **distribution** per model, so the comparison
becomes "do the two VE distributions overlap?" (the rigorous form of the question). HM
also natively consumes target *uncertainties* (Poisson on IR counts, binomial on the
repeat/ever-detected fractions, censored-survival on first-detection), which the cohort
observation provides. Use the SAME method for both models (fair comparison).

## Stage 1 — HM posterior per model (reuse DK exp 07: run_wave.py + trajectory_selection.py)
`historymatching` v2 (EmulatorBank), box-bounded params (log base_beta/titer_median;
monotone ladders as ratios), NaN-on-extinction, auto feature selection, checkpoint/
spot-resume (run_hm_full.sh). Two runs, one per model:
- **Model A — age-symptom:** params = base_beta, sus ladder (r2,r3), maternal **Erlang**
  (efficacy + mean_duration), age betas (beta0/1/2); symptom_model=age_only.
- **Model B — infection-number:** params = base_beta, sus ladder, maternal **titer**
  (efficacy/median/gsd/half_life/hill), p_symp (ratios); symptom_model=infection_number.
- Maternal differs by model BY NECESSITY (entanglement: age wants Erlang, infnum needs
  titer — exp 12/14). So this is a structural pair (symptom+maternal differ), not
  symptom-isolated. Both fit the SAME data/targets.
- Pipeline per model: HM waves → converged NROY → trajectory selection (importance-
  resample on the composite likelihood) → posterior parameter sets.
- Output: ~hundreds–thousands of posterior draws per model.

## Stage 2 — Vaccine layer
- Add RotaVax / RotaVaxProg (already in rotasim/interventions.py).
- For each posterior draw (per model): forward-sim WITH vs WITHOUT vaccine →
  achieved VE = 1 − (vaccinated symptomatic incidence / unvaccinated), in the target age window.
- **Settings dimension (the point):** parameterize LMIC vs high-income so simulated
  age-at-first-infection matches the empirical gradient (~38 wk LMIC vs ~65 wk
  high-income; PMC6736387) — via FOI/base_beta (± demographics). Hold the **underlying
  per-dose efficacy IDENTICAL across settings** — the whole hypothesis is that the same
  efficacy yields different *achieved* VE because of the age-distribution of infection.
- → VE distribution per (model × setting).

## Stage 3 — Compare
- Per setting: do the two models' VE distributions overlap? (structural uncertainty in VE)
- Across settings: does the **LMIC↔high-income VE gap** differ between the two models?
  (headline — does the age-of-infection→VE-gap prediction depend on symptom structure?)
- Plots: VE posterior-predictive per model×setting; VE-gap distributions; overlap.

## Effort / compute (rough)
- Per model: ~6 waves × ~2k sims + ~10k trajectory-selection sims ≈ ~22k sims; ×2 models;
  + the VE forward runs (posterior draws × settings × ±vaccine). DK ran single-model HM at
  40k agents / 118 workers in ~hours–day on covaguest. Whole phase ≈ multi-day of compute.
- Spot-resilient (DK's checkpoint/resume loop) — important given the evictions.

## Open decisions (resolve before building)
1. **Environment:** `historymatching` package install. DK uses `uv` (you declined uv for
   the calibration). Either install historymatching into the conda env, or run the HM
   piece under DK's uv setup. → quick check needed.
2. **Reuse mechanism:** cherry-pick DK's exp-07 scripts (run_wave/trajectory_selection)
   onto this branch and parameterize per model, vs merge his branch. → cherry-pick + adapt.
3. **Mixing:** homogeneous (our exp 11/13 worked; fewer params) vs DK's reservoir. → start
   homogeneous (drops young_reservoir/adult_contacts/infant_exposure from the box).
4. **Observation:** cohort (gives HM the target uncertainties + pins transmission via
   repeat-fraction) vs process_model. → cohort.
5. **Maternal per model:** age→Erlang, infnum→titer (entangled; settled by exp 10–14).
6. **Settings parameterization:** how LMIC vs high-income is defined (base_beta/FOI?
   demographics? contact structure?) to hit the age-of-infection gradient — needs a
   small design step + the high-income age/IR targets (currently only Bangladesh).
7. **Vaccine config:** RotaVax dose schedule + the (constant-across-settings) underlying
   efficacy to use.

## Suggested first concrete step
Confirm the environment (historymatching in conda or via uv), cherry-pick DK's run_wave +
trajectory_selection, and do a **re-identification / smoke HM** on ONE model (infnum+titer,
homogeneous, cohort) to validate the pipeline end-to-end before scaling to both models +
the VE layer. (This is calibration-workflow step 7 — re-identify before the real run.)
