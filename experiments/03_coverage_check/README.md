# Exp 03 — Prior Predictive / Coverage Check

**Question.** Two things at once, across the *whole* parameter space rather than
one point:
1. **Reachability.** Do the observed MAL-ED targets (IR-by-age, first-infection
   quartiles) fall *inside* the envelope of outputs the model can produce under
   the prior? If the data sits outside the achievable range, no optimizer can
   ever fit it — that's a model/prior problem, not a calibration problem.
2. **Endemic sanity.** Across the prior, does the model reach sensible, stable
   endemic prevalence — or does it collapse to extinction / saturate at the
   ~41%-prevalence degenerate state exp 02 found at a bad point? Characterise the
   distribution of endemic prevalence the prior produces.

Motivated by exp 02, which showed one (poorly-fit) parameter point drove the
model to a saturated, no-age-gradient state. This experiment asks whether
*good* regions of the space exist at all, and whether the targets live there.

**Plan.** Draw 50 parameter sets from the calibration prior (the 9-parameter
MAL-ED space: `base_beta` log-uniform 0.05–0.5; `beta0/1/2` age-symptom logistic;
`sus_after_1/2/3+` monotone immunity ladder; `maternal_immunity_efficacy`
0.5–0.99; `maternal_immunity_half_life_days` 30–365). Run the full ABM (20k
agents, 10 yr, 1 replicate per draw — coverage is a binary reachability question,
not a noise-estimation one). For each draw, push outputs through the *same*
detection pipeline the calibration uses (`process_incidence_maled.process_model`)
and the **corrected** person-time denominators (`rs.PersonTimeByAge`, from exp 01).
Record: overall endemic prevalence + a stationarity flag, IR per MAL-ED age bin,
and first-infection quartiles. Plot observed Bangladesh & Pakistan targets on top
of the simulated ensemble (5–95% envelope). Write results incrementally to
`outputs/results.jsonl` so a larger VM run can extend the same file.

Run plan: 50-draw quick pass locally (parallelised, ~3–5 min). Escalate to a
thorough pass (200 draws ± replicates, or 100k agents) on covaguest only if the
quick pass is promising or ambiguous.

**Success criteria.**
- Covered: every MAL-ED age-bin IR and the first-infection quartiles fall within
  the 5–95% ensemble envelope, and a meaningful fraction of draws produce
  plausible endemic prevalence (few %, with an age gradient). → proceed to
  calibration / method selection.
- Not covered: targets systematically outside the envelope. Diagnose which of the
  three causes (prior too narrow / model can't reach / observation model wrong) —
  e.g. if the 24-35m IR is always too high (no age gradient), that points at the
  immunity ladder or network structure (exp 04), not the optimizer.
- Watch for: the over-infection seen in exp 02 dominating the ensemble (most
  draws saturated) — would say the prior admits too many implausible regimes and
  the endemic operating point needs structural attention before calibrating.
