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

**Updated 2026-08-21** to replace the original ODE-vs-ABM/HM comparison
below with a clean ODE-vs-ODE comparison, now that Bangladesh has its own
fast ODE fit (exp64 age_binned, exp65 infnum) using the identical method,
code, and freed-titer convention as India's (exp58/59) — the two sites are
now directly comparable for the first time, on both symptom structures.

All models share one parameterization convention: susceptibility and
symptom probability after the 2nd/3rd+ infection are fit as **ratios**
relative to the previous order (`sus_after_2 = sus_after_1 × sus_r2`,
`sus_after_3plus = sus_after_2 × sus_r3`, and identically for `p_symp`
under infnum), confirmed identical in `ode_model.py`'s
`sigma_from_params()` for both sites, so the derived values below are
directly comparable.

**A caveat before the tables: `base_beta` is not independently
identified, and comparing it site-to-site at face value is misleading.**
In every one of the four site/model fits below, `log_base_beta` is
strongly anti-correlated (r = −0.70 to −1.00 against whichever of
`sus_r2`/`sus_r3` dominates) with the same known compensating ridge
documented in exp58 onward: a fit that lands on a point with less
susceptibility drop after reinfection (high `sus_r2`/`sus_r3`) needs a
*lower* `base_beta` to reproduce the same observed incidence, and vice
versa. Because that ridge is a near-flat plateau in the likelihood, a
single optimization run can land on either side of it — which is exactly
why `base_beta`'s apparent direction of difference **flips between the two
symptom structures** for the same site pair (age_binned: India > Bangladesh;
infnum: Bangladesh > India) even though nothing is wrong with either fit.
The quantity that *is* well identified and directly comparable is the
**equilibrium force of infection** (`foi_eq`, computed downstream of
`base_beta` and the susceptibility ratios together) — it stays tight
within each site/model despite `base_beta` swinging 2-7x across the same
seeds, and it does **not** flip direction: Bangladesh's is consistently
~2x higher than India's in both models, matching the sharper, earlier
Bangladesh incidence peak. `base_beta` is kept in the tables for
completeness, but `foi_eq` is the row to trust for a cross-site
transmission-intensity comparison.

### India vs Bangladesh, `age_binned` (both ODE, exp58 vs exp64, median [range across 6 seeds])

| Parameter | India (exp58) | Bangladesh (exp64) |
|---|---|---|
| `base_beta` (derived, see caveat above) | 0.233 [0.139, 0.360] | 0.100 [0.062, 0.206] |
| **`foi_eq`** (derived — the trustworthy comparison) | **0.0019 [0.0018, 0.0020]** | **0.0041 [0.0040, 0.0045]** |
| `sus_after_1` (residual susceptibility after 1st infection) | **0.918** [0.865, 0.962] | **1.000** [0.971, 1.000] |
| `sus_after_2` (derived)† | 0.290 [0.008, 0.661] | 0.845 [0.491, 1.000] |
| `sus_after_3plus` (derived)† | **0.023** [0.001, 0.101] | **0.273** [0.096, 0.464] |
| `maternal_efficacy` | 0.508 [0.502, 0.518] | 0.502 [0.500, 0.530] |
| `maternal_titer_median` (derived)‡ | 4.07 [4.00, 4.26] | 4.00 [4.00, 4.85] |
| `titer_half_life_days`‡ | 25.4 [25.0, 27.8] | 26.7 [25.0, 30.5] |
| `titer_gsd` / `hill_slope` | 2.78 / 4.33 | 2.87 / 5.37 |
| `p_symp_age_<6m` | **0.177** [0.150, 0.193] | **0.328** [0.283, 0.331] |
| `p_symp_age_6-11m` (the peak bin) | **0.785** [0.734, 0.814] | **0.999** [0.961, 1.000] |
| `p_symp_age_12m+` | 0.261 [0.251, 0.286] | 0.457 [0.436, 0.493] |

### India vs Bangladesh, `infnum` (both ODE, exp59 vs exp65, median [range across 6 seeds])

| Parameter | India (exp59) | Bangladesh (exp65) |
|---|---|---|
| `base_beta` (derived, see caveat above) | 0.244 [0.166, 0.344] | 0.455 [0.084, 0.589] |
| **`foi_eq`** (derived — the trustworthy comparison) | **0.0020 [0.0019, 0.0021]** | **0.0047 [0.0045, 0.0049]** |
| `sus_after_1` | 0.942 [0.823, 0.968] | **1.000** [0.980, 1.000] |
| `sus_after_2` (derived)† | 0.118 [0.015, 0.276] | 0.325 [0.254, 0.516] |
| `sus_after_3plus` (derived)† | 0.028 [0.005, 0.076] | 0.025 [0.011, 0.326] |
| `maternal_efficacy` | 0.505 [0.501, 0.529] | 0.531 [0.517, 0.582] |
| `maternal_titer_median` (derived)‡ | 4.06 [4.01, 4.31] | 4.76 [4.14, 4.96] |
| `titer_half_life_days`‡ | 25.4 [25.0, 26.6] | 26.8 [25.4, 29.0] |
| `titer_gsd` / `hill_slope` | 2.21 / 3.41 | 2.23 / 5.52 |
| `p_symp_order1` | 0.331 [0.302, 0.345] | 0.376 [0.345, 0.395] |
| `p_symp_order2` | 0.379 [0.335, 0.406] | **0.790** [0.734, 0.904] |
| `p_symp_order3plus` | 0.381 [0.111, 0.848] | 0.643 [0.391, 0.925] |

† `sus_after_2`/`sus_after_3plus` (via `sus_r2`/`sus_r3`) carry the ridge
described in the caveat above in all four fits — read every range here as
"not well constrained by this cohort's data alone," not as a precisely
estimated value. `age_binned`'s `sus_after_3plus` for Bangladesh was
partially narrowed by a real VE anchor (exp67, using PROVIDE's trial
data) — see that experiment's SUMMARY for the constrained version;
`infnum` did not get the same benefit.

‡ Both sites' titer fits sit at or near the floor of their allowed search
range (`titer_median` bounded ≥4, `half_life_days` bounded ≥25) in both
models — "the data wants this as short/weak as the search space allows,"
not a precisely pinned-down curve, on either side.

**The most interpretable, best-supported differences** (consistent
across both symptom structures, so less likely to be a ridge artifact of
one particular fit):
1. **Bangladesh's force of infection is consistently ~2x higher than
   India's**, in both models (`foi_eq` 0.0041-0.0047 vs 0.0019-0.0020) —
   a genuine, well-identified cross-site difference (see caveat above for
   why `foi_eq`, not `base_beta`, is the right quantity to read this from).
2. **`sus_after_1` saturates at its 1.0 ceiling for Bangladesh in both
   models, while India sits meaningfully below it** (0.92 age_binned,
   0.94 infnum) — Bangladesh's data wants zero susceptibility reduction
   from a single prior infection; India wants a modest one. This is the
   cleanest, most structure-independent signal in the whole comparison.
3. **Bangladesh's symptom probability at its peak bin/order runs much
   higher than India's in both models** (`p_symp_age_6-11m` 0.999 vs
   0.785; `p_symp_order2` 0.790 vs 0.379) — the same mechanism exp64/65
   documented: Bangladesh's sharper 6-11m incidence peak (~2.8x the &lt;6m
   rate, vs India's smaller relative jump) demands more symptomatic-
   detection pressure to reproduce, regardless of which structure carries it.
4. **Maternal protection strength (`maternal_efficacy`) and titer shape
   are close across both sites and both models** (efficacy ~0.50-0.53;
   titer median/half-life near the same floor-of-bounds values) — a
   genuine similarity, not just an artifact of shared fixed values, since
   both sites now have this fully freed.

### For context: the other selected/available models (older ABM/HM method, not the ODE tables above)

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
