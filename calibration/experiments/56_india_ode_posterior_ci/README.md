# Exp 56 — India Vellore: ODE equilibrium across 10 high-likelihood draws (parameter CI)

**Question.** exp55 computed the age-structured equilibrium at a single
point — India's MLE (exp47's collapsed posterior). That answers "where does
the best fit want to sit" but says nothing about how sensitive the
equilibrium picture is to parameter uncertainty. This experiment runs the
same age-structured ODE (exp55's `ode_model_age.py`, reused directly) across
10 high-likelihood parameter draws instead of one, to put a range on the
equilibrium immune-status composition and IR-by-age — entirely locally via
the ODE (fast, no ABM sims), while exp52's real HM/TS sims continue on
zebra.

**Point selection.** Top 10 by `logL` from exp47's already-complete
trajectory-selection scoring (`experiments/47_india_age_psymp_interp/
outputs/ts/sir_results.jsonl`, 3000 draws already scored against the real
composite likelihood) — not exp47's collapsed `posterior.csv` (ESS=1.02,
only 6 unique rows post-resampling, too little diversity for a spread).
exp52's own survival-vote-weighted TS run is still in progress on zebra;
this uses the best complete dataset available now.

**Method.** For each of the 10 draws: run the age-structured ODE to
equilibrium (60 years, same convergence check as exp55), compute the
immune-status composition and detection-adjusted all-infection IR-by-age
(same age-dependent detection-probability correction as exp55, using each
draw's own fitted `p_symp`). Report median + [min, max] across the 10 draws
per age bin/metric — a range, not a formal percentile CI, given n=10.

See SUMMARY.md for results.
