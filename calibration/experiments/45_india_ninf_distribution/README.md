# Exp 45 — India Vellore: per-agent infection-count distribution (exp39)

**Question.** After exp39/42/43/44 all showed low, and progressively worse,
ESS for the India Vellore cohort fit, AK asked: can we get the actual
distribution of number-of-infections-per-agent from exp39's model, to check
whether the population is really one exposure-risk pool or splits into two
groups (low vs. high infection counts) — which would point toward missing
population heterogeneity (e.g. a two-strain / two-exposure-route structure)
as the explanation for the persistent &lt;6m/6-11m tension, rather than a pure
parameter/data-adequacy problem.

**Plan.** `MALEDCohort` already tracks true per-child cumulative infection
count (`n_inf`) internally but didn't expose it — added as a harmless,
additive field to `calibrate_maled._run_one_replicate`'s cohort-observation
output (ignored by the GOF/HM path; not a calibration target). Re-run 8
draws from exp39's resampled posterior (40k agents each, same model/flags as
exp39) and pool the per-child counts across replicates, checking for
bimodality/overdispersion vs a single homogeneous-population null
(Poisson-like, tempered by declining post-infection susceptibility).
