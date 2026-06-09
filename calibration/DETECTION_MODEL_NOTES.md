# MAL-ED detection-model decisions (rotavirus calibration)

Decisions for the detection/observation model, settled 2026-06-09 with A. Kraay.
These govern **adopting D. Klein's cohort/detection emulation** (his exp 04+). They do
**not** change the current `process_model` pipeline (exp 09/10) — see Scope note.

## Target construction — KEPT AS-IS

- A "case" = **EIA-positive** among the **TAC-tested subset** of samples.
- Person-time is **also computed within the TAC-tested subset**, so numerator and
  denominator live in the *same* frame → the IR is internally consistent (a
  sub-population rate, not a deflated one).
- Provenance: the TAC subset comes from a prior **coinfection** analysis, which needed
  the full TAC panel to call true coinfection. This rotavirus calibration inherits the
  subset. Gel corruption leaves some samples with TAC for other pathogens (e.g.
  campylobacter) but **missing the rotavirus-TAC field** → those drop out. Assumed
  **non-differential by age** (technical, pathogen-specific field dropout), so the
  IR-within-TAC is a reasonable estimate for the cohort.
- **Rule:** do not change the numerator (e.g. to TAC positivity, or to all samples)
  without changing the denominator to match. Don't break the frame.

## Detection model (for the cohort emulation)

- **Positivity is EIA-defined → model detection SENSITIVITY ~0.85, applied to BOTH
  symptomatic and asymptomatic branches.** (TAC's ~100% does NOT apply here — positivity
  is not TAC-defined.) This is the haircut that genuinely belongs: it is a true miss of
  EIA-negative-but-infected samples and does **not** cancel in num/denom.
- **Do NOT also apply a TAC-tested-completeness factor (`tested~=0.85`).** Because num
  and denom are both inside the TAC-tested frame, that completeness **cancels**; applying
  it would double-count. (Earlier `sampled x tested` framing collapses to just EIA
  sensitivity.)
- **Diarrheal-stool *collection* completeness (`sampled`)**: only apply to the extent it
  is NOT already balanced by the matched num/denom. If PT tracks the covered/TAC frame it
  cancels; if PT is full child-time it deflates. Verify against the exact denominator
  before adding.
- **No specificity term.** EIA antigen positivity is load-thresholded (~ etiologic), so
  intrinsically fairly specific; and this target is not Cq-cutoff-defined.
- **Keep the age-varying asymptomatic surveillance schedule** (monthly to 12mo ->
  quarterly after; DK's `_p_surv` = shed/interval). It does NOT cancel — it governs which
  infections fall in a sampling window — and it is the real age-dependent detection
  effect. Caveat: EIA sensitivity is lower for asymptomatic (low viral load), so a single
  0.85 likely *overstates* asymptomatic detection; consider a lower asymptomatic
  sensitivity if the repeat-fraction / age-at-first-detection targets misbehave.

## Open / parked

- Which assay defines **asymptomatic surveillance** positivity (EIA or TAC)? If it
  differs from the symptomatic branch, that's a mixed-method target — confirm.
- Per-branch EIA sensitivity (symptomatic vs asymptomatic, load-dependent).

## Scope note

These apply when wiring D. Klein's cohort/detection emulation. The **current** pipeline
(`process_model`, exp 09/10) uses symptomatic ~100% capture + `p_asymp_detect=0.4`. The
symptomatic EIA haircut is **age-flat**, so it only rescales the symptomatic-IR level
(absorbed by `base_beta`) — it would NOT change the exp 09/10 peaked-vs-infection-number
**shape / model-selection** conclusion. So no re-run is needed for this alone; fold it in
when the cohort emulation (with the repeat-fraction and first-detection targets) is
adopted, where the symp:asymp detection ratio actually matters.
