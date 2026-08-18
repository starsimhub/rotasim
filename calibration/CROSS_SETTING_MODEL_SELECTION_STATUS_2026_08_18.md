# Cross-setting symptom-model selection and the India VE identifiability problem

*Status update, 2026-08-18.*

## Project context

The underlying question across all three sites is whether the LMIC
age-distribution of rotavirus infection produces a different **achieved**
vaccine effectiveness than a high-income setting, at similar underlying
per-dose vaccine efficacy. That requires, per site: (1) selecting which
immune/symptom-model structure the data actually supports — age-based
severity (`age_binned`: P(symptomatic) varies by age, susceptibility
depends only on infection count) vs infection-number-based severity
(`infnum`: P(symptomatic) varies by infection order) — and (2) checking
that the selected structure gives a stable, decision-usable basis for
forward-predicting vaccine impact.

**Headline: the three sites do not share one answer, and getting the
selection right required checking it twice** — an earlier pass and a
later, more rigorous re-test disagreed for Bangladesh, and this week's
work found that even a confirmed selection isn't the end of the story for
India.

## Site-by-site model selection

| Site | Result | Evidence |
|---|---|---|
| **UK** | **Infnum decisively required.** Every age-based structure tested fails by a wide margin. | exp28: clean 4-model head-to-head (infnum, age-quadratic, age_binned, age_and_infection) against real UK surveillance (England & Wales 2008-2012, N=3,498). Deviance: infnum 49-63 vs age_binned 544-614 (**NROY emptied — a fit failure**, not just worse), age-quadratic 834-856, age_and_infection 598-710. Infnum wins by **>10x**. |
| **Bangladesh** | **Genuinely non-identifiable between the two structures** — not a clean win either way. | exp18-27 (HM, corrected maternal): infnum gets higher ESS (107.9) than age_binned (61.5), but both are "usable." exp20 (VE comparison over that same matched pair): *"pre-vaccine data alone cannot decisively distinguish the two models' achieved VE"* — infnum's median VE is ~1.7x higher, but 95% credible intervals overlap. exp29 (icddr,b Dhaka external validation): both models reproduce the observed risk peak — doesn't discriminate. exp28 states it directly: *"The MAL-ED Bangladesh birth cohort could not discriminate the symptom structure."* (An earlier, differently-parameterized test, exp09, had favored age decisively — that result used an older age form and does not survive the later re-test with the canonical `age_binned` structure.) |
| **India (Vellore)** | **age_binned decisively required**, confirmed twice — once in the original ABM/HM arc, and independently in a new, much faster ODE pipeline built this week. | Original arc (exp31-52, ABM/HM): every attempt to fit India under `infnum` hit the same &lt;6m/6-11m incidence-shape failure that `age_binned` eventually resolved. New confirmation (exp57-59, ODE direct-fit — see below): `age_binned`'s best fit (logL -284.2 to -284.7 across 6 seeds) beats `infnum`'s (-293.4 to -293.8 across 6 seeds) by **~9 log-units**, both equally stable. Visibly, infnum's best fit is a nearly flat IR-by-age curve that cannot reproduce the real, sharp 6-11m incidence peak — order-keyed symptom probability can't target a specific age the way age-keyed probability can. |

## Fitted parameters across sites

All models share one parameterization convention: susceptibility and
symptom probability after the 2nd/3rd+ infection are fit as **ratios**
relative to the previous order (`sus_after_2 = sus_after_1 × sus_r2`,
`sus_after_3plus = sus_after_2 × sus_r3`, and identically for `p_symp`
under infnum) — confirmed identical in the underlying code
(`hm_calibrate.py`'s `untransform()` for Bangladesh/UK, `ode_model.py`'s
`sigma_from_params()` for India), so the derived values below are directly
comparable across sites.

### India vs Bangladesh, both under `age_binned` (the comparison you asked about)

| Parameter | **India** (ODE, exp58, median [range across 6 seeds]) | **Bangladesh** (ABM/HM, exp25, posterior median) |
|---|---|---|
| `base_beta` | 0.233 [0.140, 0.360] | 0.153 |
| `sus_after_1` (residual susceptibility after 1st infection) | **0.918** [0.865, 0.962] | **0.729** |
| `sus_after_2` (derived) | 0.290 [0.008, 0.662]† | 0.382 |
| `sus_after_3plus` (derived) | **0.023** [0.001, 0.101]† | **0.185** |
| `maternal_efficacy` | **0.508** [0.502, 0.518] | **0.805** |
| `maternal_titer_median` | 4.07 [4.00, 4.26]‡ | 20.0 (fixed, not fitted) |
| `maternal_titer_half_life_days` | 25.4 [25.0, 27.8]‡ | 50.0 (fixed, not fitted) |
| `titer_gsd` / `hill_slope` | 2.78 [1.42, 3.24] / 4.33 [2.80, 5.38] | 2.3 / 4.7 (both fixed, not fitted) |
| `p_symp_age_<6m` | **0.177** [0.150, 0.193] | **0.484** |
| `p_symp_age_6-11m` (the peak bin) | **0.785** [0.734, 0.814] | **0.566** |
| `p_symp_age_12m+` | 0.261 [0.251, 0.286] | 0.313 |

† These two rows carry exp58's ridge — the 6 seeds agree tightly on logL
(within 0.45 units) and on `p_symp`, but scatter widely on
`sus_after_2`/`sus_after_3plus`. Read India's range here as "not well
constrained by this cohort's data," not as "0.29 with meaningful
precision."

‡ Bangladesh's titer *shape* (median/half-life/gsd/hill) was held **fixed**
at literature-informed values (`--fix-titer-shape`; only `maternal_efficacy`
was fitted) — only India's ODE fit freed all four. India's fitted values
also sit at or near the floor of their allowed search range
(`titer_median` bounded ≥4, `half_life_days` bounded ≥25), so treat them as
"the data wants this as short/weak as the search space allows," not a
precisely pinned-down curve.

**The two most interpretable, best-supported differences** (tightly
estimated on both sides, not ridge-affected):
1. **Maternal protection is much weaker in the India fit** (efficacy 0.51
   vs Bangladesh's 0.81 fixed-shape fit at 0.805) — consistent with the
   documented biology that Vellore's persistent community strain
   (G10P[11]) is not blocked by maternal titer, a India-specific mechanism
   this arc identified separately (see project memory on neonatal priming).
2. **India's symptom-by-age curve is far more sharply peaked**: <6m/6-11m/12m+
   = 0.18/0.79/0.26 (a >4x jump into the peak bin) vs Bangladesh's
   0.48/0.57/0.31 (a much flatter profile, <6m nearly as high as the peak).
   This is the quantitative signature of the qualitative finding driving
   this whole arc — India has a real, sharp 6-11m incidence peak that
   Bangladesh's data doesn't show as strongly, and it's the age-symptom
   channel, not transmission or susceptibility, carrying that difference.

### For context: the other selected/available models

| Parameter | Bangladesh infnum (exp27, ESS 107.9) | UK infnum (exp28, the selected model) |
|---|---|---|
| `base_beta` | 0.138 | 0.058 |
| `sus_after_1` | 0.751 | 0.975 |
| `sus_after_2` (derived) | 0.441 | 0.900 |
| `sus_after_3plus` (derived) | 0.224 | 0.647 |
| `maternal_efficacy` | 0.896 | 0.733 |
| `p_symp` order 1 / 2 / 3+ (derived) | 0.566 / 0.370 / 0.145 | 0.762 / 0.535 / 0.175 |

UK's much lower `base_beta` (0.058 vs Bangladesh/India's 0.14-0.23) and
much higher `sus_after_1/2/3+` (weaker acquired immunity, i.e. people stay
more susceptible after each infection) both point the same direction as
this project's core FOI-gradient premise: UK is the low-transmission,
older-age-of-infection setting by construction of the fit, not just by
assumption.

## This week's methods advance: a fast, deterministic ODE pipeline for India

The India ABM/HM calibration takes ~3 days per run on a 160-core VM. This
week we built a deterministic ODE reduction of the same model (age-structured
transmission equilibrium + a birth-cohort detection layer) and fit it
directly against the exact same likelihood via global optimization —
**full 12-parameter fits in ~5 minutes**, a roughly 500-1000x speedup. It
reproduces the ABM/HM posterior's mode almost exactly (within 1 log-unit)
and let us run stability and structural checks in an afternoon that would
otherwise cost weeks of compute.

## New finding: India's fit is not identifiable, and it changes predicted VE

Confirming `age_binned` is the right *structure* for India turned out not
to be enough. Refitting the confirmed model from 6 different random starts
found a **compensating ridge**: many different combinations of the
transmission rate and the susceptibility-after-reinfection parameters fit
the 0-35-month cohort data equally well (logL spread of only 0.45 units
across 6 independent fits) — the natural-history data alone cannot tell
these parameter combinations apart.

That ridge turns out to matter a great deal for vaccine predictions. Using
a mechanism that mirrors the model's actual vaccine-dose logic (a
successful dose is mechanically equivalent to a prior infection), we
computed direct/individual vaccine efficacy at a fixed, plausible per-dose
take (0.74) across 10 of these equally-good natural-history fits:

- **Direct VE at 6-11 months ranged from 39.8% to 68.5%** — a 29-point
  spread — depending only on *which* equally-good natural-history fit was
  used.
- **The single best-fitting natural-history point in the entire ridge gave
  VE = 39.8%**, below the real-world test-negative estimate for India
  (Nair et al., 52-59% at 6-11m). A different point, only 0.14 log-units
  worse, gave 68.5%, above it.

In other words: for India, model selection alone (picking age_binned over
infnum) does not pin down a vaccine-impact prediction. Two calibrations
that are statistically indistinguishable on pre-vaccine data can disagree
by nearly 30 percentage points on achieved VE. **This scope is
individual/direct VE specifically — population-level (herd-inclusive)
VE sensitivity across the same ridge has not yet been tested.**

## Update: real VE data partially resolves the ridge — done, and the result is genuinely mixed

A similar problem surfaced for Bangladesh in 2026 (exp42): adding a VE
target to the calibration's scoring, on top of the existing ABM/HM
posterior, only relocated the tension (pulling toward VE-plausibility cost
the cohort fit on its two weakest dimensions) rather than resolving it. We
re-tested this idea for India in the new, much cheaper ODE pipeline —
not reweighting a fixed, already-degenerate sample (exp42's approach) but
jointly optimizing natural history and VE together from scratch, across
the same 6 seeds as the unconstrained fit (exp61, now complete).

**No sacrifice to the natural-history fit**: the joint fit's natural-history
component lands at -284.23 to -284.80 across the 6 seeds — statistically
indistinguishable from the unconstrained fit's -284.22 to -284.67. **VE
lands tightly on target**: 58.4-61.6% across all 6 seeds, essentially right
on Nair et al.'s 59% point estimate, with much less scatter than the
unconstrained ridge's 40-68.5% spread. So the joint approach succeeds where
exp42's reweighting failed — a real result, not just a repeat of the old
attempt with a faster method.

But it only narrows **part** of the ridge:

| Parameter | Unconstrained (exp58) range | VE-constrained (exp61) range |
|---|---|---|
| `sus_r2` | 0.009 – 0.745 (span 0.74) | **0.401 – 0.518 (span 0.12, ~6x tighter)** |
| `sus_r3` | 0.049 – 0.245 (span 0.20) | 0.072 – 0.441 (span 0.37, **wider, not narrower**) |

`sus_r2` (governing the order-1→2 susceptibility drop, the transition most
of a just-vaccinated infant's doses actually act on) narrows sharply.
`sus_r3` (order-2→3+, mostly reached only later, beyond the 3-dose
schedule's direct reach) does not — if anything it's less constrained under
the joint fit. **VE data is a real, useful identifiability constraint, but
only for the part of the parameter space the vaccine's own mechanism
actually touches** — it isn't a general fix for the ridge.

## Bottom line for the cross-setting question

The three sites appear to need genuinely different symptom/immune
structures — not the same model with different parameters. UK requires
infection-number-based severity; India requires age-based severity;
Bangladesh's pre-vaccine data cannot distinguish the two. That heterogeneity
is itself informative for the achieved-VE-gradient question this project is
built around — but for India specifically, the added finding that natural-
history model selection doesn't identify vaccine impact means any single-
point VE forward-prediction for India should be treated as provisional.
Jointly fitting real VE data alongside natural history (exp61) helps —
it pins down VE tightly and narrows the part of the ridge (`sus_r2`) the
vaccine mechanism actually acts on, at no cost to the natural-history fit
— but it does not fully resolve the underlying non-identifiability
(`sus_r3` remains as loose as before, or looser). Any India VE
forward-prediction should still be reported as a range reflecting this
residual ridge, not a single point estimate.
