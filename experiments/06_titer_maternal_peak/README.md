# Exp 06 — Titer-Based Maternal Immunity: Sharpening the IR Peak

**Question.** With the IBM-correct **titer-based maternal model** (per-infant
log-normal initial titer → common exponential decay → sigmoidal Hill protection)
as the peakiness mechanism, a **low `young_reservoir` (1–15)** to stay off the
mixing knife-edge, and the **acquired-immunity ladder** producing the 12-23m
decline, can we reproduce MAL-ED Bangladesh's *sharp* 6-11mo IR peak — deep `<6m`
trough (~36% of peak), peak at 6-11mo, decline after — at a smooth (non-bimodal)
endemic?

Motivated by: exp 04/05 got the peak *bin* right but the *magnitude/sharpness*
wrong; exp 05 showed structured mixing with a hot reservoir is bimodal; and the
realization (with literature support) that the `<6m`→`6-11m` ramp is a maternal-
antibody-waning signal best modeled per-infant (titer + Hill), not via an Erlang
shape. The peakiness now has mechanistic levers: titer **gsd** + **Hill slope** =
sharpness; **median titer** + **half-life** = drop age (~6mo per Malawi data).

**Plan.**
- Network: 3-group `ss.MixingPools`, **`young_reservoir` swept 1–15** (the focus),
  `infant_exposure` low, homogeneous background.
- Maternal: **`maternal_immunity_model='titer'`**, sweep `median` (drop age),
  `gsd` (sharpness), `half_life` (drop age), `hill_slope` (sharpness), `efficacy`
  (trough depth).
- Severity: infection-number (`p_symp_1/2/3+`).
- Acquired immunity: `sus_after_1/2/3+` ladder → the **12-23m decline** (not mixing).
- Observation: cohort + surveillance detection + individual dropout (unchanged).
- 40k agents, ~1000 draws, capybara. Score **primarily on IR-by-age** (peak *bin*
  AND *magnitude/sharpness*: target `<6m`/`6-11m`/`12-23m` = 1.91/5.37/2.35).

**Success criteria.**
- Good: a region produces a sharp 6-11mo peak (`<6m` ≈ 30–45% of the peak,
  magnitude ~5) at a smooth endemic (no bimodality), with the maternal drop near
  ~6mo. Then read off reinfection (predict ~0.2 at the matching low FOI).
- Informative failure: even per-infant maternal + low reservoir can't sharpen the
  peak → the flattening is the **observation model** (cohort/surveillance), which
  exp 03 (steady-state) hints at — and that becomes the next target.
- Watch: degeneracy between titer `median`/`half_life` (both set drop age) and
  between `gsd`/`hill_slope` (both set sharpness); whether `young_reservoir` stays
  smooth across 1–15.
