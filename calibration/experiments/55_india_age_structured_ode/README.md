# Exp 55 — India Vellore: age-structured extension of exp54's ODE

**Question.** exp54's ODE tracks infection-order/immune-status with no age
dimension, matching the ABM's own age-blind, well-mixed transmission
(confirmed: `ss.RandomNet` has no age-assortativity, and susceptibility-by-
order is age-independent — see exp54). But since people are born, age, and
die, the *cross-sectional* age distribution of immune status is age-
structured anyway, purely mechanically — an 18-month-old has simply had more
time to accumulate infections than a 3-month-old. This experiment adds that
age dimension (MAL-ED bins: `<6m`/`6-11m`/`12-23m`/`24-35m`/`36m+`) to
compute the equilibrium age-structured immune-status distribution.

**Motivation (AK):** use this to initialize an ABM population directly at
its endemic-equilibrium immune structure — instead of starting the whole
population fully naive plus a synchronized 4% seed wave, which exp50 showed
drives the extinction problem in the first place (a huge, near-total initial
sweep followed by a fragile post-wave trough). If the ABM could start
already-equilibrated, that transient (and its associated stochastic
extinction risk) might be avoidable, or at least much smaller. A useful side
effect: this age-structured equilibrium also implies an all-infection
IR-by-age, which is directly comparable to the real `ir_all_by_age`
calibration targets, as a fast fidelity check.

**Method.** `ode_model_age.py` extends exp54's 22-compartment
(maternal-chain × infection-order) layout with 5 age bins, each a
well-mixed sub-population with a fixed exit ("aging") rate = 1/bin-width —
the standard "uniform-flux-out-the-top" approximation for age-structured
compartmental models. Transmission (force of infection) is a single
population-wide number applied identically to every age bin's susceptible
classes (matching the ABM's age-blind mixing); order-based susceptibility
and the maternal chain are unchanged from exp54. Run at India's current MLE
(exp47's collapsed posterior) to 60 years, confirmed converged (compartment
*proportions* — not raw counts, since births > deaths means the population
grows ~0.9%/year forever — change by <1e-13 between year 50 and 60).

See SUMMARY.md for results.
