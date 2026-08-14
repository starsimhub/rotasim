# India Vellore calibration — status update, 2026-08-14

*Follow-up to the 2026-08-13 discussion (`INDIA_STATUS_2026_08_13.md`). That
update ended with exp47 as the best joint cohort fit yet, but flagged that
ESS was still collapsing (~1/3000) and that single-seed extinction scoring
was a suspected cause. This update covers what we found chasing that down —
a real mechanistic answer, not yet a fix to the original &lt;6m/6-11m tension,
but it changes how we should read every India result to date.*

## Recap: where exp47 left off

`age_binned`, p_symp freed between the slum- and MAL-ED-derived anchors, is
still the best joint fit on the persistent problem (symptomatic IR <6m vs
6-11m):

| Metric | exp39 (fixed slum) | exp47 (free) | Target |
|---|---|---|---|
| IR <6m | 0.59 | 0.48 | 0.40 |
| IR 6-11m | 1.39 | 1.75 | 1.71 |
| ESS (/3000) | 9.15 | 1.02 | — |

ESS near 1 means the fitted posterior has essentially zero spread — one
particle dominates the importance weights. That's the thread we pulled on.

## What we found: extinction near the viable boundary is genuinely random, not just noisy

**exp48** (logistic-classifier smoothing of the single-seed extinction
signal) helped (ESS 1.02→2.39) but didn't resolve it, and traded some
cohort-fit quality for the ESS gain — a partial fix, not the underlying
answer.

**exp50** asked the question directly: take 10 India parameter points already
classified "extinct" from their one original simulation seed, spanning a 20x
range of transmission rate (`base_beta`), and re-run each at 10 fresh seeds.
Result: **100/100 sims went extinct**, deterministically — every point died
out within ~2 years regardless of how hot or mild its initial epidemic wave
was. That ruled out "bad luck on one seed" as the *whole* story — this looked
like a real population-level ceiling.

**exp51** tested that directly: hold parameters fixed, scale the simulated
population from 40,000 (used throughout this entire project) up to
100k/200k/400k. Population size **does** rescue extinction — but only above
a sharp threshold, not gradually:

| Parameter set | R0 (corrected*) | 40k | 100k | 200k | 400k |
|---|---|---|---|---|---|
| lowest transmission rate | 2.0 | 100% extinct | 100% | 100% | 100% |
| mid (India's actual best-fit region) | 3.0 | 100% | 100% | 80% | 50% |
| next step up | 4.1 | — | 0% | 0% | 0% |

*Naive R0 = transmission-rate × infectious-duration undercounts real R0 by
ignoring each person's ~7 daily contacts; naively multiplying by contacts
overcounts it by ignoring that transmission is ~90% suppressed during the
~8-day asymptomatic phase. Correcting for both gives R0≈2-4 across this
project's fitted parameter range — nowhere near measles-scale (12-18), a
useful sanity check we'd gotten wrong initially.

![Extinction rate and time-to-extinction vs population size, by R0](experiments/51_india_population_size/figures/ccs_rescue_by_beta.png)

**exp53** dug one level deeper: does accounting for post-infection immunity
(people who survived the first epidemic wave are harder to reinfect) resolve
the threshold more cleanly? We computed the *effective* reproduction number
right after the initial wave for the same three parameter sets. All three
are still comfortably above 1 (1.6, 1.8, 2.0) — yet empirical extinction
risk at N=400k still swings from 100% to 50% to 0% across that same range.

![Mean-field Re (all above 1) vs actual extinction risk](experiments/53_india_effective_r/figures/re_after_wave.png)

**The upshot:** this is not a deterministic viability question — a
textbook-supercritical epidemic (Re comfortably >1) can still go extinct by
chance with real probability when it's only *modestly* above the threshold,
because the number of infected individuals surviving the trough between the
initial wave and any recovery is small and finite. That probability is
extremely sensitive to exactly how far above 1 the effective R sits — which
is exactly why a 2-3x change in transmission rate flips the outcome from
"always dies" to "always survives." Population size matters because it sets
how many individuals are actually available to carry the chain through that
bottleneck, not because it changes the reproduction number itself.

**Practical consequence:** every India HM/trajectory-selection result to
date (exp39 through exp49) classified extinction from a *single* simulation
seed per parameter point. Given the mechanism above, a real fraction of
those "extinct" classifications were parameter points that are actually
viable most of the time and just lost one coin flip — meaning the search has
likely been discarding real posterior mass, not just correctly ruling out
implausible regions.

## The fix, now running: a multi-seed survival vote

Rather than trying to compute a sharper analytical threshold (exp53 shows
that's the wrong lever — the relevant quantity is a stochastic survival
probability, not a formula), we replaced single-seed extinction scoring
outright:

- **`hm_calibrate.py`**: each parameter point now runs **5 independent
  seeds**. The fraction that survive becomes a new calibration target
  (target = 1.0, i.e. a fully viable point should survive every seed). The
  other targets are averaged across the surviving replicates only.
- **`trajectory_select.py`**: same 5-seed vote at the posterior-resampling
  stage; the likelihood is weighted by that survival fraction (a standard
  "hurdle model" combination — real cohort data implicitly assumes the
  population *did* survive, so a parameter point's overall support should be
  discounted by how *likely* it was to survive at all).

**exp52** re-runs exp39's exact model configuration with this mechanism
active (on zebra, 155 cores, `--early-stop` enabled so extinct replicates
don't waste time simulating a dead population for the full 10 years). Wave 1
of 6 is running now; given the ~5x extra simulation cost, the full run will
take a while — current estimate is wave 1 within the next ~1-1.5 hours, full
6-wave run to follow.

## Where this leaves us for next week

1. **Once exp52 finishes**: does ESS improve materially over exp39/47/48? Is
   the fitted posterior less degenerate? This is the direct test of whether
   single-seed extinction noise was corrupting the search all along, versus
   the &lt;6m/6-11m tension being a separate, still-open problem.
2. **If exp52 looks healthy**: immediately follow up with the widened
   p_symp bracket (exp49's staged design, using the corrected slum-cohort
   anchor found this week) layered on top of the survival-vote fix, to
   re-attempt closing the &lt;6m/6-11m gap on a search that isn't fighting
   single-seed extinction noise.
3. Proposing to reconvene once exp52 (and ideally the exp49-style follow-up)
   are in — expect this to be sometime next week.

## Open questions carried forward

- Is the &lt;6m/6-11m tension itself a calibration problem (exp46/47 line,
  now to be re-tested under exp52's cleaner search) or a deeper structural
  one (two-strain population)? Still open — exp52/49 is the next real test.
- Should the India HM pipeline move to a larger `N_AGENTS` (100-200k)
  instead of / in addition to the multi-seed vote? exp51 suggests either
  lever works for points near the threshold; the survival vote is cheaper
  per point but the two are not mutually exclusive if the vote alone doesn't
  fully resolve the ESS problem.
