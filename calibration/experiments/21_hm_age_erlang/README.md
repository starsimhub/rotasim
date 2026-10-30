# Exp 21 — History-matching posterior for the AGE-symptom + ERLANG maternal model

**Question.** age+titer (exp 16) cannot be turned into a usable posterior: importance resampling
collapses (ESS ~1, even with φ=2/ρ=0.05 → 2.6), and emulator-MCMC fails real-sim validation
(64% of draws go extinct, posterior-predictive off-target). Root cause: age+titer's NROY is far
too loose (the age-symptom curve and the rich titer maternal are *redundant* <6m-suppressors →
a degenerate, weakly-identified parameter space). So we fall back — as pre-registered in exp 14's
success criteria — to **age+Erlang** as the age member of the VE comparison. age+Erlang fit *well*
under Optuna (exp 9/10, cosine 0.997), so its HM NROY should be tight and its posterior
well-behaved. Cost: the matched pair becomes maternal-confounded (age+Erlang vs infnum+titer)
rather than same-maternal. See [`../16_hm_age_titer/`](../16_hm_age_titer/),
[`../18_age_posterior/`](../18_age_posterior/), [`../10_denominator_rerun/`](../10_denominator_rerun/).

**Plan.** Same HM driver/observation/targets as exp 16, only the maternal model differs:
`hm_calibrate.py --model age --maternal erlang` (transmission + sus-ladder + age betas +
Erlang maternal: `maternal_efficacy`, `maternal_mean_duration_days`, n_stages fixed at 6 — vs the
5-param titer block). Bayes-linear, cohort observation, ~6 waves → NROY → posterior
(importance resampling should work here given a tight NROY; MCMC as backup).

**Success criteria.** A tight NROY (contrast age+titer's loose 0.52→0.26) with low extinction,
and a posterior that **validates** (real-sim posterior-predictive on the 5 targets). That gives
the age VE member for exp 20. If age+Erlang's NROY is *also* loose/high-extinction, the problem
is the age model's identifiability itself, not the maternal pairing — a deeper finding.
