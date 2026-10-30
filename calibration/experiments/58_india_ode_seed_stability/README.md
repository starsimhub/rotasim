# Exp 58 — India Vellore: ODE MLE stability across seeds + re-check equilibrium uncertainty

**Question.** exp57's ODE direct-fit used a single `differential_evolution`
seed and found an MLE (logL=-284.37) matching exp52's HM posterior mode —
but a single seed on a 12-dim, possibly-multimodal surface doesn't rule out
other, materially different optima (exp57's own "Next" flagged this). Now
that a full search costs ~5 minutes (post the BDF solver fix) instead of
days, this experiment (1) repeats the search across several more seeds to
check whether the optimum is stable, and (2) uses the resulting pool of
high-likelihood draws to redo exp56's equilibrium-uncertainty analysis
(age-structured compartment fractions), to see whether directly-optimized
ODE draws give a tighter uncertainty band than exp56's ABM-HM-posterior
draws did.

**Plan.**
1. Run exp57's exact search (`differential_evolution`, 12 params, `BDF`
   solver, `workers=-1`) 5 more times with different seeds on zebra,
   sequentially (each run already saturates all 160 cores, so no benefit to
   running concurrently — and concurrent runs would compete for the same
   cores anyway). Also re-run the original seed (20260817) once more inside
   this experiment so every run captures the same extra diagnostic: not
   just the single best point, but the top-K members of the *final
   population* (`result.population`/`result.population_energies`), giving
   more than 6 candidate points to pool from.
2. Compare best logL and best parameters across all runs — report the
   range, not just agreement/disagreement in prose.
3. Pool the top ~10 highest-logL points across all runs' final populations,
   run `simulate_age` to equilibrium for each (same recipe as exp56), and
   plot the compartment-fraction-by-age median+range, directly next to
   exp56's original figure for comparison.

**Success criteria.** Stability: do the ~6 runs agree on logL within roughly
1 log-unit and on the key freed parameters (p_symp bins, susceptibility
structure), the way exp57's single run and exp52's HM posterior agreed? If
they scatter across meaningfully different logL values or parameter
regions, the surface is more multimodal than exp57 alone suggested, and
exp57's specific numbers should be read as *a* good fit, not *the* fit.
Equilibrium check: does the compartment-fraction CI narrow relative to
exp56's (which found the 12-23m/24-35m bins wide, specifically in the
higher-order susceptible classes) when using directly-optimized draws
instead of ABM-HM-posterior draws?
