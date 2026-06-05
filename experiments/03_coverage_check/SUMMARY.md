# Exp 03 — Prior Predictive / Coverage Check — SUMMARY

**Question.** Across the whole 9-parameter prior: (1) do the MAL-ED Bangladesh
targets fall inside the model's achievable envelope? (2) does the model reach
sensible, *stable* endemic prevalence, or the degenerate saturated state exp 02
found at one point?

**Setup.** 1000 prior draws, Bangladesh, 20k agents, 1 replicate each, run on
the `capybara` VM (120-core HB120, 753s, 0 failures). Targets and detection
computed in-step by the new `rs.MALEDTargets` analyzer (validated exact vs the
old `InfectedStrainStats`→`process_model` path; O(agents) memory).

## Result

**Immune/transmission stability: confirmed sweep-wide.** In-window prevalence
drift (late-half − early-half mean) has median 0.0003, 95th pct 0.0016, max 0.011
(pp); only 2/1000 draws exceed 1 pp, none exceed 2 pp; drift is uncorrelated with
prevalence level (corr −0.12). Exp 02's single-point stationarity holds across the
entire prior — the 5-year burn-in is adequate everywhere.

**Marginal coverage passes — but trivially (prior far too wide).** All 4 IR bins
and the first-infection median fall inside the 5–95% ensemble envelope. But the
envelope spans ~[0, 90] IR per bin vs targets ~2–5 — the prior admits everything
from extinction to hyperendemic, so marginal coverage is uninformative (the
"prior too wide" failure mode).

**The prior is dominated by over-infection.** Median endemic prevalence 0.23;
57% of draws have prevalence >0.2. Real rotavirus point prevalence is a few %.
Plausible-prevalence draws are a minority — the prior needs structural attention /
tightening before calibration.

**Joint reachability — the meaningful question — is favourable, with a caveat.**
0/1000 draws put all 4 bins within 3× of target, but that is driven by the
near-zero 24-35m bin (the data there is a *single case*, IR=0.14, where ratios are
meaningless). The **best draw (#314) reproduces the shape well** — model IR
`[1.34, 5.35, 2.61, 0.0]` vs target `[1.91, 5.37, 2.35, 0.14]`, peak correctly at
6-11m, **and at a realistic 3.6% prevalence.** So the symptomatic-IR shape *is*
reachable.

## Figures

![Coverage](figures/coverage.png)

## Observations

- **Reconciles with Alicia's exp 05/06.** Her result: a *peaked* (quadratic)
  age-symptom curve reproduces the shape, while infection-number-only cannot. Our
  prior includes the peaked age-symptom curve (`beta0/1/2`), and draw #314
  confirms it reaches the shape. No contradiction with her exp 02 "0/250 fits":
  her structural tension was for the *infection-number* symptom model; the
  age-severity model resolves it.
- **Important caveat on what IR-shape coverage proves.** Symptomatic IR ≈
  infections × symptom-prob(age). A peaked age-symptom curve can manufacture the
  6-11m peak almost regardless of the underlying infection dynamics. So matching
  IR-by-age does **not** strongly constrain the transmission/immunity model. The
  discriminating target is age-at-first-infection (actual infection timing), which
  is still biased by the unfixed censoring (44% BD / 64% PK) — exp 05.
- 26.7% of draws place the symptomatic peak at 6-11m; the rest are flat/saturated.

## Next

- The coverage property is established (shape reachable, system stable), so we
  *can* calibrate — but the marginal check is the wrong instrument for joint
  identifiability (cf. Alicia's exp 01). The real constraint is the joint IR +
  first-infection fit.
- **Fix the first-infection censoring (KM) — exp 05.** Until then the most
  discriminating target is biased, and over-infection in the prior is unconstrained.
- Consider tightening the prior away from the hyperendemic region (e.g. lower
  `base_beta` upper bound) so calibration doesn't spend its budget there.
- `rs.MALEDTargets` is validated and ready to PR (also fixes the calibration
  driver's own memory exposure).
