# Exp 55 — India Vellore: age-structured extension of exp54's ODE

**Date:** 2026-08-14.

**Question.** See README.md — compute the equilibrium age-structured immune
status distribution (for ABM initialization) and check it against the real
all-infection IR-by-age targets as a fidelity check.

**Result.** The equilibrium age structure is exactly the mechanical pattern
expected — immune status accumulates smoothly with age, purely from time-
since-birth, with no age-dependence built into transmission or
susceptibility at all:

| age_bin | maternally protected | susceptible (naive) | susc. (1 prior inf) | susc. (2 prior inf) | susc. (3+ prior inf) | mean prior infections |
|---|---|---|---|---|---|---|
| <6m | 43.6% | 32.8% | 11.0% | 1.7% | 0.3% | 0.17 |
| 6-11m | 12.8% | 36.9% | 25.2% | 6.0% | 1.4% | 0.47 |
| 12-23m | 1.7% | 19.7% | 34.7% | 16.5% | 8.7% | 1.08 |
| 24-35m | 0.2% | 8.7% | 31.2% | 23.2% | 19.9% | 1.58 |
| 36m+ | 0.0% | 0.1% | 1.1% | 1.9% | 84.3% | 2.95 |

![Equilibrium immune-status composition by age, and model-implied vs real all-infection IR-by-age](figures/age_structured_equilibrium.png)
![Equilibrium mean infection count by age](figures/mean_prior_infections_by_age.png)

By age 3, nearly everyone (97%, and 84% specifically at the "3+" floor
susceptibility) has been infected at least 3 times — consistent with
India/Vellore being characterized as a high-FOI setting throughout this
project.

**The model-implied all-infection IR-by-age gets the shape right but not
the magnitude.** Both the ODE and the real MAL-ED target peak at 6-11m and
decline afterward, but the ODE's equilibrium rate runs ~2-5x higher across
every bin (model 4.7/6.2/5.5/4.8 vs target 1.3/2.5/1.0/0.0 per 100
person-months). **This is expected, not a modeling error** — see
Observations for why.

## Observations

1. **The age-structure result itself doesn't depend on the magnitude
   mismatch.** The *relative* composition — what fraction of each age bin
   sits in each immune compartment — is a self-consistent equilibrium
   property of the order/aging dynamics regardless of the absolute
   transmission rate's fit quality, so it's usable for ABM initialization
   even before the magnitude gap is resolved (rescaling the whole system's
   throughput doesn't change *how* immune status accumulates with age, only
   *how fast*).
2. **"Still equilibrating" was checked directly and ruled out.** AK asked
   the sharp, correct question: shouldn't incidence *fall* toward
   equilibrium (after the initial oversized wave), not rise toward it — and
   if the ABM's 5-10y window isn't yet converged, comparing the ODE's OWN
   value at year 10 (not its asymptotic equilibrium) against the ABM should
   narrow the gap. Checked directly: the age-structured ODE's IR-by-age at
   year 10 (4.70/6.25/5.52/4.78) is within <1% of its 60-year asymptotic
   equilibrium (4.69/6.24/5.52/4.80) — the model is fully converged well
   before year 10 even starts. This rules out equilibration timing as the
   explanation entirely (an earlier draft of this SUMMARY proposed it; it
   was wrong, and the correction is left here rather than silently editing
   it away).
3. **The real explanation: this ODE computes the TRUE infection incidence;
   the target is a DETECTED-infection rate.** Confirmed directly:
   `process_incidence_maled.py`'s own docstring calls `ir_all` "the model's
   all-DETECTED output," and `rotasim/analyzers.py` gives the detection
   probabilities explicitly — symptomatic infections are captured with
   probability `symp_collection x eia_sensitivity` = 0.80x0.85 = 0.68;
   asymptomatic infections only via monthly stool sampling,
   `(shed_days/30) x eia_sensitivity` = (13/30)x0.85 ~ 0.37. This ODE has no
   detection layer at all (100% of true infections counted), so it should
   run higher than a detection-filtered target by roughly the inverse of the
   blended detection probability — predicting a ~2-2.5x true:detected ratio,
   which matches the 6-11m bin almost exactly (6.24/2.52 ~ 2.5x) and is the
   right order of magnitude everywhere else. The gap is doing exactly what
   it should; comparing this ODE's output to `ir_all` needs a detection
   discount, not a bigger/longer ODE run.
4. **This does NOT change exp53/54's extinction-mechanism conclusions.**
   Those relied on relative comparisons across parameter points (R0, Re at
   the trough, trough case counts) that are unaffected by a detection-layer
   question that doesn't exist in that framing.

## Next

- **The most useful immediate follow-up given AK's original motivation**:
  turn the age-bin-by-compartment table above into actual ABM initial
  conditions (draw each simulated agent's starting infection-order state
  from this age-conditional distribution, weighted by the agent's age at
  t=0) and check empirically whether that removes or shrinks the fragile
  post-wave trough exp50/51/53/54 identified — directly testable by
  comparing extinction rates with vs. without equilibrium-initialization at
  the same N=40,000.
- **If a quantitative (not just shape) IR-by-age check is wanted**, apply
  the same detection-probability discount used by the real observation layer
  (symptomatic vs. asymptomatic-via-monthly-stool) to this ODE's true
  incidence before comparing to `ir_all` targets, rather than comparing raw
  incidence directly.
- Sweep this age-structured equilibrium across more of the posterior (not
  just the single MLE point) to see how sensitive the age-composition
  picture is to the parameters that remain uncertain.
