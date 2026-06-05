# Feedback for Alicia — MAL-ED Calibration Review (Dan, 2026-06-05)

Reviewed the MAL-ED calibration on `dec_calibration_akraay`. Overall the
structure is solid. Below are findings from a set of quick diagnostic
experiments (in `experiments/`), ordered by how much they affect the fit.

---

## 1. HIGH PRIORITY — Person-time denominators are biased 10–23% too high

**Where:** `process_incidence_maled.py :: compute_person_months_steady_state`

**The bug.** Person-months per age bin is computed as:

```python
count_in_bin_at_sim_end × calibration_window_months
```

i.e. it snapshots the population age distribution **once at the end of the
sim** and assumes that headcount held constant across the whole 5-year
window. But the population is *growing* (births > deaths: Bangladesh
19/6, Pakistan 27/7 per 1000/yr). The end-of-sim headcount is therefore
**larger** than the time-averaged headcount over the window.

**Impact.** Measured directly (snapshot-PT vs. true accumulated-PT,
`experiments/01_demographics_check/validate_fix.py`):

| Site       | <6m  | 6-11m | 12-23m | 24-35m |
|------------|------|-------|--------|--------|
| Bangladesh | 0.99 | 1.00  | 1.05   | 1.16   |
| Pakistan   | 0.99 | 1.04  | 1.08   | 1.16   |

(ratio of snapshot-PT to true accumulated-PT — i.e. how much the
denominators are inflated)

The bias is **age-dependent**: negligible for the youngest bins, growing to
~16% in the 24-35m bin (those cohorts were born earlier, when the population
was smaller, so the end-of-sim snapshot most overstates them). Because
IR = cases / PT, **inflated denominators make the modelled IR look
artificially low** — and because the distortion is concentrated in the older
infant bins, it doesn't just shift `base_beta`, it **flattens the modelled
age-incidence gradient**, biasing the fitted immunity ladder (`sus_after_3plus`
in particular). It's a *systematic* error, not noise.

**The fix.** Accumulate person-time *during* the science window instead of
assuming. At each timestep, add `count_in_bin × dt` to a running total.
That's the literal definition of person-time and it's exact regardless of
population growth. The `AgeStats` analyzer already records per-bin counts
every step, so the time series exists — we just need to integrate it over
the calibration window rather than reading the final frame.

A drop-in `PersonTimeByAge` analyzer + `compute_person_months_accumulated()`
is in this branch (`dec_calibration_akraay_dk`); see
`experiments/01_demographics_check/validate_fix.py` for the validation
showing it recovers the unbiased denominators. Happy to PR it.

---

## 2. The science window (60 months) doesn't match MAL-ED follow-up (~24 months)

MAL-ED followed each child from birth for about 24 months (Bangladesh avg
23.1mo, Pakistan 25.2mo, max ~30mo). The calibration computes model IR over
a **5-year (60-month) window** of an endemic population. These aren't the
same exposure: a birth cohort followed 0–24mo vs. a steady-state slice
0–60mo. For the infant bins it's probably close, but worth confirming.

**Pin for later:** consider recreating the study directly — a within-sim
"mini MAL-ED": enroll a birth cohort, follow each enrolled child for 24
months, and compute IR-by-age and first-infection on that cohort exactly
as the study did. Removes the cohort-vs-cross-section assumption entirely.

---

## 3. HIGH PRIORITY — First-infection target ignores censoring (44% / 64%!)

**Where:** `process_incidence_maled.py :: load_first_infection_quartiles`

The target quartiles are computed from `event_observed == 1` only —
children who actually had an observed first infection. But **44% of
Bangladesh and 64% of Pakistan children were censored** (never had an
observed first infection during follow-up). Dropping them conditions on
"was infected," which biases the target median toward *earlier* ages.

The model side (`compute_model_first_inf_quartiles`) includes every
simulated agent infected by 36 months — no equivalent selection. So the
model and data quartiles aren't measuring the same thing.

This is likely a big part of why **Pakistan** (64% censored, flat incidence)
is hard to fit. The fix is a Kaplan-Meier estimate of the age-at-first-
infection distribution that properly accounts for censoring, then compare
model to the KM quartiles. (exp 05, in progress.)

---

## 4. Initial age distribution is UK, not South Asian — but it self-corrects fast

The sim initializes agents from `uk_age_data.csv` (~94% aged 5+), then
applies site birth/death rates. Exp 01 shows the population re-equilibrates
to the correct infant fraction within ~1 year (well inside the 5-year
burn-in), and the equilibrium <6m fraction matches theory (~0.95% BD,
~1.35% PK). So this is **not** causing a problem for the infant bins —
but it's worth swapping in a site-appropriate age file for cleanliness,
and it *would* matter for any older-child targets (the <36m fraction is
still slowly drifting up at year 10).

---

## 5. Network is RandomNet — no age structure

`ss.RandomNet(n_contacts=7)` mixes all ages uniformly. For rotavirus,
infant exposure is dominated by household/caregiver contact, not random
mixing with the whole population. An age-structured network (MixingPools
with infant-protective mixing) may change the infant IR meaningfully —
testing in exp 04. (There's already `AGE_ASSORTATIVE_NETWORK_FINDINGS.md`
from earlier work — worth revisiting.)

---

## Summary of priorities

1. **Fix the person-time denominators** (systematic beta bias) — fix ready.
2. **Fix the censoring in the first-infection target** (KM) — esp. Pakistan.
3. Match / recreate the MAL-ED follow-up window (24mo cohort).
4. Try an age-structured network.
5. Site-appropriate initial age distribution (low priority — self-corrects).
