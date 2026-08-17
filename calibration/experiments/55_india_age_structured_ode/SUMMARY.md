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

![Equilibrium immune-status composition by age, and true vs detection-adjusted vs real all-infection IR-by-age](figures/age_structured_equilibrium.png)
![Equilibrium mean infection count by age](figures/mean_prior_infections_by_age.png)

By age 3, nearly everyone (97%, and 84% specifically at the "3+" floor
susceptibility) has been infected at least 3 times — consistent with
India/Vellore being characterized as a high-FOI setting throughout this
project.

**The model-implied all-infection IR-by-age gets the shape right; the
detection adjustment narrows the magnitude gap substantially but does not
close it.** Both the ODE and the real MAL-ED target peak at 6-11m and
decline afterward. The ODE's raw (true-infection) rate runs 2-5x higher
than the target (4.7/6.2/5.5/4.8 vs 1.3/2.5/1.0/0.0 per 100 person-months) —
expected, not a modeling error by itself, since this ODE has no detection
layer at all (every true infection counted) while the target is explicitly
a *detected*-infection rate. Applying the real observation model's own
age-dependent detection probabilities (symptomatic 0.68 age-independent;
asymptomatic 0.363 for `<12mo` monthly surveillance, dropping to 0.121 for
`>=12mo` quarterly surveillance — see Observations) to the ODE's true
incidence, using the MLE's own fitted `p_symp` by age, shrinks the ratio
from 2.5-5.6x down to **1.3-1.7x** (detection-adjusted 2.18/3.35/1.64/1.43
vs target 1.35/2.52/0.99/0.00). **That's a real, substantial narrowing, not
a close match** — every bin is still 30-70% too high, and the 24-35m bin is
qualitatively wrong in a different way (target is 0 cases; the ODE predicts
a still-substantial 1.4 detection-adjusted rate there). The detection
adjustment identifies a genuine, correctly-signed piece of the gap; it is
not the whole explanation.

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
3. **Part of the explanation: this ODE computes the TRUE infection
   incidence; the target is a DETECTED-infection rate.** Confirmed directly:
   `process_incidence_maled.py`'s own docstring calls `ir_all` "the model's
   all-DETECTED output," and `rotasim/analyzers.py` (`MALEDCohort._p_surv`)
   gives the detection probabilities explicitly and age-dependent —
   symptomatic infections: `symp_collection x eia_sensitivity` = 0.80x0.85 =
   0.68 (age-independent); asymptomatic infections: `(shed_days/interval) x
   eia_sensitivity`, where the surveillance interval is MONTHLY (30.4d)
   below age 12mo and QUARTERLY (91.3d) at/above 12mo — a 3x drop, giving
   asymptomatic detection ~0.363 (`<6m`/`6-11m`) vs ~0.121 (`12-23m`/
   `24-35m`). Blending with the MLE's own fitted `p_symp` by age gives
   overall detection probabilities of 0.465/0.537/0.297/0.297 for the four
   bins, and applying those to the ODE's true incidence is what produces the
   1.3-1.7x (not 2-5x) residual ratio above.
4. **The remaining 1.3-1.7x gap is NOT resolved by this experiment** — it
   is smaller than the raw mismatch, correctly signed, and worth reporting,
   but calling it "close" (an earlier draft of this SUMMARY did) overstated
   it, per AK's direct pushback on the figure. Candidates for the remaining
   gap, not disentangled here: (a) genuine approximation error in this ODE
   reduction (the per-age-bin single-exponential exit rate, the Erlang-chain
   maternal approximation, no between-individual heterogeneity); (b) the
   24-35m bin's real target is based on a very small sample (0 cases in 746
   person-months per `load_ir_all_targets`), so a real-vs-model comparison
   there may just be dominated by sampling noise rather than a genuine
   model gap; (c) some other simplification in how this ODE's order/age
   dynamics interact with the true ABM's discrete, individual-level process
   that a mean-field reduction can't capture exactly.
5. **This does NOT change exp53/54's extinction-mechanism conclusions.**
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
- **Pin down the remaining 1.3-1.7x gap** (observation 4) before treating
  the detection-adjusted IR-by-age numbers as quantitatively validated —
  e.g. run the real ABM at the MLE parameters and directly tabulate its own
  `ir_all_by_age` output for comparison, rather than relying solely on this
  reduction's approximation.
- Sweep this age-structured equilibrium across more of the posterior (not
  just the single MLE point) to see how sensitive the age-composition
  picture is to the parameters that remain uncertain.
