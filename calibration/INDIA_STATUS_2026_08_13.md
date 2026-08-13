# India Vellore calibration — status for 2026-08-13 discussion

## Project context

Cross-setting achieved-VE gradient project: does the LMIC age-distribution of
infection produce a different *achieved* vaccine effectiveness than a
high-income setting, at similar underlying vaccine efficacy? Three sites:
Bangladesh (MAL-ED cohort), UK (pre-vaccine surveillance), India (MAL-ED
Vellore cohort + Tamil Nadu surveillance). Same model, same code, fit
independently per site.

## Bangladesh — done, high confidence

`infnum` (infection-number symptom model) selected via a decisive model
comparison (exp09, `age_only` vs `infnum` under a shape-aware Poisson
likelihood). Fits the MAL-ED Bangladesh cohort well, and independently
validated against icddr,b Dhaka surveillance (exp29, no refit) — predicted
risk-by-age matched observed. Closed, not currently active.

## UK — done, moderate confidence

Same `infnum` structure selected (exp28, decisive vs 3 alternatives). Vaccine
validation (exp30): direct VE at take=0.9 predicts 74%, matching the UK
test-negative estimate of 77%. Known gap: the model (no waning) doesn't
reproduce the observed post-vaccine age-shift — a diagnosed, understood
limitation, not an open mystery. Closed, not currently active.

## India — the open problem

**The core issue, since exp31 (first India attempt):** the model persistently
overshoots symptomatic incidence at &lt;6 months and undershoots the 6-11 month
peak, no matter what else is changed. `age_binned` (exp39) is the best
structure found — ESS 9.15/3000, but that pair remains off:

| Target | exp39 (best fit) | Target |
|---|---|---|
| IR &lt;6m | 0.59 | **0.40** |
| IR 6-11m | 1.39 | **1.71** |
| IR 12-23m | 0.60 | 0.61 (good) |
| repeat_frac | 0.12 | 0.14 (close) |
| Q25 first-infection | 17.5mo | 15.1mo |

### What's been tried (all failed to close the &lt;6m/6-11m gap)

![Every India lever tried vs. target](experiments/meeting_2026_08_13_figures/india_overview_all_experiments.png)

| Exp | Lever | Result |
|---|---|---|
| 31-38 | infnum + various neonatal-priming symptom-flag designs | Same &lt;6m/6-11m pattern persists across every variant |
| 40 | `age_and_infection` symptom model | **Invalidated by a bug** found 2026-08-12 (`MALEDCohort` never implemented it — silently scored 100% of infections as symptomatic). Fixed; re-tested properly in exp44 |
| 42 | Add VE as a trajectory-selection scoring target | Pulls VE toward target but only by moving &lt;6m/6-11m further off; ESS 9.15→4.60 |
| 43 | Make neonatal priming a real, detected infection | &lt;6m/6-11m untouched; overshoots Q25 the *other* direction; ESS→4.73 |
| 44 | Fractional neonatal order-crediting, under `infnum` and (bug-fixed) `age_and_infection` | Both negative; `age_and_infection`'s ESS collapsed to 1.9/3000 (14 free params — too many for 44 cases) |
| 45 | Per-agent infection-count distribution check | One homogeneous exposure pool, not two risk groups — rules out simple agent heterogeneity as the explanation. Separately: ~85-87% of sampled parameter space goes fully extinct, consistent across every model tried |
| 46 | Swap p_symp source: MAL-ED-derived (ascertainment-corrected) instead of slum-cohort-derived | First lever with correctly-signed effect on both &lt;6m and 6-11m — overshot past target on both sides, but in the *right direction*. ESS→1.26 |
| **47** | **Free p_symp between the slum and MAL-ED anchors (not fixed at either)** | **Best joint fit in the whole India arc.** IR 6-11m = 1.75 vs target 1.71 (essentially exact); IR &lt;6m = 0.48 vs target 0.40 (closest yet). ESS still ~1/3000 — see below |

**exp47 update (result landed this morning, before the meeting):**

![exp39 vs 46 vs 47](experiments/47_india_age_psymp_interp/figures/exp39_46_47_comparison.png)

| Metric | exp39 (fixed slum) | exp46 (fixed MAL-ED) | **exp47 (free)** | Target |
|---|---|---|---|---|
| IR &lt;6m | 0.59 | 0.23 | **0.48** | 0.40 |
| IR 6-11m | 1.39 | 1.96 | **1.75** | **1.71** |
| IR 12-23m | 0.60 | 0.66 | 0.57 | 0.61 |
| repeat_frac | 0.12 | 0.12 | 0.10 | 0.14 |
| Q25 (mo) | 17.5 | 13.9 | 18.0 | 15.1 |
| ESS (/3000) | 9.15 | 1.26 | 1.02 |

Freeing p_symp between the two anchors (rather than fixing at either) finds
a point that nails the 6-11m peak and gets far closer on &lt;6m than anything
tried before — by far the best result on the actual persistent problem. But
ESS is still ~1/3000 (the fitted posterior has literally zero spread across
resamples) — same degenerate pattern as everything since exp39. One
actionable detail: `p_symp_age_6_11` landed at 0.548, pinned against its
upper bound (0.55) — the search wants to go higher than we allowed it to.

### A separate, structural issue found today: single-seed extinction scoring

Every India HM run (39 through 47) scores whether a parameter point is
"extinct" from **one simulation seed per point**. Given this fit sits at an
~85-87% baseline extinction rate — a knife's edge — a parameter set with a
genuinely low (say 15%) extinction *probability* has a real chance of losing
its one coin flip and being wrongly treated as fully implausible. Since this
signal is cycled first in history matching and shapes wave-1 NROY before
anything else, this could be adding real noise to *every* result above, not
just a specific run.

**Fix staged (exp48, no new simulations needed):** replace the raw
single-draw signal with a logistic classifier's predicted P(extinct), fit on
all (parameters, extinct/not) pairs accumulated across waves — borrows
information across nearby already-sampled points the way logistic regression
normally does. Smoke-tested clean; staged to run automatically after exp47.

### Where things stand right now (updated 10:58am EST, before the meeting)

- **exp47 finished** — result above (best joint fit yet, ESS still ~1/3000).
- **exp48 (exp47's design + the classifier-based extinction fix) is running
  now** on zebra (auto-launched the moment exp47 finished, wave 1 in
  progress) — result not available yet, likely after the meeting.
- Both use the same free-parameter set otherwise (`base_beta`, `sus_after_*`,
  maternal titer params) as exp39.

### Open questions for discussion

1. **Is the &lt;6m/6-11m tension a p_symp-calibration problem (exp46/47 line) or
   a deeper structural one (two-strain population, not yet tried)?** exp47
   is the strongest evidence yet for "calibration problem" — the joint fit
   on the persistent targets is now good. Worth seeing exp48 out before
   fully concluding, since ESS is still degenerate.
2. **Is ESS collapsing because the joint constraint is genuinely tight, or
   because single-seed extinction noise has been distorting the search this
   whole time?** exp48 (running now) is a direct test of this.
3. If exp47/48 don't resolve it: pursue a two-strain structural model
   (the persistent Vellore community strain vs. wild-type, weakly
   cross-protective), or lean on the larger-N Tamil Nadu surveillance data
   instead of continuing to refine against the sparse 44-case Vellore cohort.
