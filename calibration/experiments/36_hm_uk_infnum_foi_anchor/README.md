# Exp 36 — UK HM (infnum): FOI anchor via first-infection timing target

**Question.** Exp 28 (UK infnum) calibrated only to the shape of the surveillance case age
distribution (proportions in 6 bins, <5 years). That shape target does not constrain the
absolute force of infection: a high-beta / high-p_symp_1 / reinfection-driven solution matches
the 12-23 month case peak *without* having first infections at the right age. The resulting
NROY posterior has median first-infection age of 1.7 months at the calibrated base_beta — far
below the biological expectation for the UK (~15 months). Vaccination at 2+4 months then arrives
after most first infections, giving a predicted VE ~10% rather than the observed ~74% (exp 30).

This experiment adds a first-infection timing target derived from Hasso-Agopsowicz et al.
(PMC6736387): HIC median = 65 weeks (15.0 months), IQR = 40-107 weeks (9.2-24.7 months). The
target is anchored at the HIC-wide data median (not the UK projection, which was extrapolated
from other low-mortality countries). The surveillance case-shape target is retained to constrain
the symptom structure; the timing target constrains the FOI.

**Plan.**
- Same ABM setup as exp 28: infnum, --fix-titer-shape, UK demographics, 8k agents.
- New feature in the HM: `first_inf_median_months` with obs = (15.0, 3.5 mo).
  SD = 3.5 mo → 3σ NROY boundary at [4.5, 25.5] mo, slightly wider than the Hasso-Agopsowicz
  HIC IQR. Draws at 1.7m are 3.8σ from target → excluded; draws at 9 months are 1.7σ → included.
- `Surveillance.step()` now tracks all order-1 infections throughout the entire 10-year run
  (not just the calibration window) and reports `first_inf_median_months` via `results_dict()`.
- 3 waves × 1500 samples on zebra.
- `--all-targets`: cycle over all 7 features (6 bin proportions + timing) once per wave.
- Follow with trajectory selection (n=3000) using the exp36 NROY.

**Success criteria.**
- Non-empty NROY with first_inf_median in ~[9, 24] months.
- Shape target still plausibly fitted (no major degradation in 12-23 month proportion).
- Downstream: exp 37 vaccine validation using exp 36 NROY gives VE consistent with ~74% (exp 30).

**Code changes (committed with this experiment):**
- `rotasim/analyzers.py` — `Surveillance`: track `_first_inf_ages_m` throughout full run,
  expose `first_inf_median_months` in `results_dict()`.
- `calibration/calibrate_maled.py` — pass `first_inf_median_months` through surveillance branch.
- `calibration/hm_calibrate_uk.py` — `obs_cols()`, `make_observations()`, `make_simulator()`
  updated to include `first_inf_median_months` with obs = (15.0, 3.5).

**Note on exp 28 backward compatibility.** Exp 28 NROY was produced before this change; its
draws are not consistent with the new timing target. Downstream analyses (exp 30) used exp 28
draws but coincidentally selected the right-FOI regime via top-K filtering. Exp 36 replaces
exp 28 as the canonical UK infnum posterior for VE prediction purposes.
