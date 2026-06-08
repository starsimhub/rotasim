# Exp 01 — Demographics Equilibrium Check

**Question.** Does the modelled age distribution reach demographic equilibrium during the burn-in period, and is that equilibrium plausible for Bangladesh/Pakistan? The sim initializes from `uk_age_data.csv` — a UK age structure with ~94% of agents aged 5+ — but then applies South Asian birth/death rates (Bangladesh: 19/6 per 1000/year; Pakistan: 27/7). If the equilibrium infant fraction is inconsistent with those birth rates, or if equilibrium isn't reached by year 5, every IR-by-age comparison is built on a wrong denominator.

**Plan.** Run a 10-year sim at 20k agents with Bangladesh demographics. Snapshot the full age distribution at years 0, 1, 2, 3, 4, 5, 7, 10. Compare the <6m fraction at each snapshot to the expected equilibrium value (birth rate / (birth rate + death rate) × fraction of life spent in that bin, roughly). Also check whether the person-months denominator the calibration code computes (steady-state snapshot at sim end) matches the theoretical expectation. Run a second pass for Pakistan. Outputs: age-distribution trajectory plot and a summary table of infant fraction by year.

**Success criteria.** 
- Good: <6m fraction stabilises within years 2–4 and is consistent with `birth_rate / 1000 × bin_width_years` to within ~20%. The calibration window (years 5–10) is clean.
- Failure: the <6m fraction is still drifting in years 5–10 (burn-in inadequate), or the equilibrium value is off by more than 2× from expectation (wrong demographics). Either implies the IR-by-age denominators are wrong, and the calibration is fitting noise.
