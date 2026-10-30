# Exp 67 — Bangladesh: VE-constrained joint fit, both models

**Question.** exp66 found Bangladesh's direct-VE ridge is as wide as
India's (~25-40 points across takes, for both `age_binned` and `infnum`).
India's exp61 showed a real trial-based VE anchor (Nair et al.) can narrow
that ridge by adding it as a second joint-likelihood target alongside the
0-35mo natural-history fit. Does the same approach work for Bangladesh,
using a real Bangladesh trial VE estimate as the anchor, for both models?

**VE anchor.** PROVIDE (Rogawski et al. 2018, *J Infect Dis* 217(6):861-8,
reporting on the original Dhaka PROVIDE RCT), traditional (uncorrected —
per AK, NOT the natural-immunity-adjusted version) per-protocol estimate:
**severe RVD (Vesikari ≥11), postvaccination window (18wk-2y): 63.1%,
95% CI 33.0-79.7%** → `VE_TARGET=0.631`, `VE_SIGMA≈0.119` (from CI width
÷ 2 ÷ 1.96, same convention as exp61's Nair anchor).

**Two design points carried over from AK's confirmed decisions:**
1. Vaccine mechanism keeps the existing assumed future 3-dose Rotavac-like
   schedule (6/10/14 wk) rather than switching to match PROVIDE's actual
   2-dose Rotarix/RV1 schedule (10/17 wk) — accepted approximation, since
   Bangladesh hasn't introduced a real vaccine and India's schedule is the
   best available anchor for the eventual program.
2. Both models (`age_binned` via exp64's cohort model, `infnum` via
   exp65's) get the joint fit, mirroring exp66 — no model selection.

**Mechanical difference from exp61.** exp61's VE calculation is written
for a **6-11m window** specifically (matching Nair's India age-band). This
experiment needs a **postvaccination-window (18wk to 2y) cumulative VE**
calculation instead — a new `rate_window_take()` generalizing exp60/61's
`rate_6_11m_take()` to arbitrary age bounds, applied here at (4.14, 24.0)
months.

**Plan.** For each model: monkeypatch the model's exp64/65 `run.py`
module (same technique as exp61, since exp57/64/65's `run.py` unconditionally
inserts its own directory into `sys.path`, blocking cross-module
`cohort_model` swapping any other way) to jointly optimize the 12
natural-history+symptom parameters against a composite log-likelihood =
(0-35mo natural-history logL) + (Gaussian logL on predicted VE vs
VE_TARGET/VE_SIGMA at a **fixed** `take=0.74`, same fixed value exp61
used for India — Bangladesh has no independently validated take of its
own, so this reuses India's assumed value rather than introducing a new
untethered free parameter). Single-seed `differential_evolution` per
model to start (like exp61); expand to multi-seed only if the single fit
looks informative.

**Success criteria.** India's exp61 found VE-constraining narrowed
`sus_r2` ~6x (the transition the vaccine's own doses act on) but left
`sus_r3` essentially unnarrowed (governs a transition most vaccinated
infants haven't reached within a 6-11m window) — a partial, mechanistically
-scoped win, not full ridge collapse. One reason to expect a *different*
outcome here: PROVIDE's window is much wider (18wk-2y, not 6-11m only),
so by the end of it a meaningful fraction of the cohort has reached
order 2-3+ — this anchor may have a real chance at constraining `sus_r3`
too, where Nair's narrower window structurally couldn't. Also worth
comparing whether the VE-constrained fit still preserves exp64/65's
model-fit gap (age_binned still clearly wins the natural-history peak)
or whether the added VE term changes that ranking.
