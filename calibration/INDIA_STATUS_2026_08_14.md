# India Vellore calibration — full update since 2026-07-07/08

*Prepared for Nick Grassly and Ben Lopman. The last shared status (2026-07-07/08)
covered the Bangladesh and UK fits, both closed by that point. This is
everything since then on the third site, India/Vellore, which has remained
an open problem throughout — including a mechanistic discovery this week
that changes how we should read every India result to date.*

## Project context

Cross-setting achieved-VE gradient project: does the LMIC age-distribution
of infection produce a different *achieved* vaccine effectiveness than a
high-income setting, at similar underlying vaccine efficacy? Three sites,
same model/code, fit independently: Bangladesh (MAL-ED cohort, done), UK
(pre-vaccine surveillance, done), India (MAL-ED Vellore cohort + Tamil Nadu
surveillance, **open** — the subject of this update).

## The core India problem, unchanged since the first attempt

Since exp31 (the first India HM run, late June), the model has persistently
**overshot symptomatic incidence under 6 months and undershot the 6-11 month
peak**, regardless of symptom-model structure, neonatal-priming mechanism,
or p_symp source. That single tension is the thread running through
everything below.

## Timeline

| Exp | Date | What | Result |
|---|---|---|---|
| 31 | Jun 25-29 | `infnum` symptom model, 3 HM variants (incl. neonatal priming) | Fails on the same <6m/6-11m pair every time; ESS≈1 (point estimates) |
| 32/33 | Jun 29 | Fix p_symp from Vellore biweekly data; try `age_and_infection` | Same pattern persists |
| 35 | ~Jul 16 | Free p_symp under `age_binned` | Same pattern persists |
| **39** | **Jul 14** | `age_binned`, p_symp fixed from Vellore biweekly cohort, titer maternal, neonatal priming, FOI anchored via all-infection IR | **Best-mixing India run to that point**: ESS 9.15/3000 (up from ≈1). Still misses: IR 6-11m 1.39 vs target 1.71 |
| 40 | Jul 14 | `age_and_infection` (age + infection-order combined) | ESS 1.25 — badly degenerate. **Later found to be invalid** (see below) |
| 41 | Jul 17 | Forward-predict vaccine impact from exp39's posterior (Rotavac, vs. Nair et al. India test-negative VE ~52-59%) | Direct VE at 6-11m came out **negative** (-41% to -79%) — wrong sign, not just off-target |
| 42 | Jul 17 | Re-score exp39's posterior adding a VE-plausibility likelihood term | Confirms the tension is structural: pulling VE toward target (0.47, close to 0.50) made the cohort fit *worse* on exactly its two weakest dimensions. ESS 9.15→4.60 |
| 43 | Aug 11 | Make neonatal priming a real, detectable infection (previously a silent no-op under `age_binned` — a bug found while scoping this experiment) | Overshot first-infection timing the *other* way (Q25 8.7mo vs target 15.1); <6m/6-11m untouched. ESS→4.73 |
| 44 | Aug 11-12 | Fractional neonatal order-crediting under `infnum` and (bug-fixed) `age_and_infection` | Both negative; `age_and_infection` ESS collapsed to ~2/3000 (14 free params, too many for 44 cases) |
| 45 | Aug 12 | Per-agent infection-count distribution check | One homogeneous exposure pool, not two risk groups — rules out simple agent heterogeneity. ~85-87% of sampled parameter space extinct, consistently |
| **46** | **Aug 12-13** | Swap p_symp source: MAL-ED-derived (ascertainment-corrected) instead of slum-cohort | **First lever with correctly-signed effect on both bins** — overshot past target on both sides. ESS→1.26 |
| **47** | **Aug 13** | Free p_symp *between* the slum and MAL-ED anchors | **Best joint fit in the whole India arc**: IR 6-11m 1.75 vs target 1.71 (essentially exact), IR <6m 0.48 vs target 0.40 (closest yet). ESS still ~1.02/3000 |
| 48 | Aug 13 | Replace single-seed extinction scoring with a smoothed logistic-classifier estimate | ESS 1.02→2.39, extinction rate 79%→64%, much healthier HM convergence — but trades away exp47's near-exact 6-11m fit |
| 49 | staged, not run | Widen the 6-11m p_symp bracket for a corrected slum-cohort anchor found this week | — |
| 50 | Aug 13 | **Direct test: is extinction seed-dependent?** 10 diverse "extinct" parameter points × 10 fresh seeds each | **100/100 went extinct, deterministically** — ruled out "bad luck on one seed" as the whole story |
| 51 | Aug 13-14 | Scale population size (40k→400k) holding parameters fixed | Population size *does* rescue extinction, but only above a sharp threshold in transmission rate — not gradually |
| 52 | Aug 14 (running) | Replace single-seed/classifier extinction scoring with a 5-seed survival vote, in both the HM and posterior-resampling stages | In progress — see below |
| 53 | Aug 14 | Compute effective R after the initial epidemic wave | Confirms this is a stochastic near-critical persistence problem, not a deterministic one |

## Phase 1 (Jun 25 – Jul 14): establishing the best structure

`infnum` (the model that won for both Bangladesh and the UK) could not
simultaneously fit India's IR-by-age and age-at-first-infection targets
under any tested variant (exp31-35). Fixing symptom probabilities directly
from an independent, near-complete-detection Vellore biweekly cohort and
switching to an **age-binned** symptom structure (`age_binned`) gave the
first real improvement: **exp39**, ESS 9.15/3000 (vs. ≈1 before), though the
6-11m undershoot remained (1.39 vs. target 1.71).

![Exp 39 vs 40 — age_binned vs age_and_infection, posterior-weighted fit](experiments/39_india_age_binned_fixed/figures/exp39_vs_exp40_comparison.png)

*(Note: exp40's `age_and_infection` comparison in this figure was later found
to be invalid — a bug in the cohort observer meant every infection was
scored 100% symptomatic, unrelated to the intended model. Fixed in code;
`age_and_infection` has not yet been validly re-tested against Vellore.)*

## Phase 2 (Jul 17): vaccine-effectiveness validation exposes a deeper tension

Following the same forward-prediction approach that validated well for the
UK (exp30: model 74% vs. observed 77% direct VE), we forward-predicted
Rotavac impact from exp39's posterior. **Direct VE at 6-11m came out
negative** (-41% to -79%, vs. a ~52-59% target) — the vaccinated arm's
modeled incidence was *higher* than the unvaccinated arm's, not just
off-target. Re-scoring the same posterior with an explicit VE-plausibility
term (exp42) confirmed this is structural, not a search/weighting artifact:
pulling weighted-mean VE to 0.47 (close to the 0.50 target) only worked by
moving the cohort fit *further* off on its two already-weakest dimensions
(IR 6-11m 1.39→1.18, first-infection timing 17.5→18.8mo). This is the same
Pareto tension named in the project's own working notes, now extended to
show VE plausibility sits in the same trade-off, not a separately fixable
issue.

## Phase 3 (Aug 11-12): three neonatal-priming mechanisms, all negative

India's model includes a persistent community neonatal rotavirus strain
(literature-documented, ~50% of neonates, distinct from the strains driving
later community transmission). We tried three different ways of crediting
this mechanism against the <6m/6-11m tension:

1. **Real, detectable infection** (exp43) — overshot first-infection timing
   the opposite direction (Q25 collapsed to 8.7mo vs. target 15.1) without
   touching the original <6m/6-11m pair.
2. **Fractional order-crediting under `infnum`** (exp44) — made the 6-11m
   undershoot *worse* (1.14 vs. exp39's 1.39), ESS collapsed to 2.98.
3. **Fractional order-crediting under `age_and_infection`** — even more
   degenerate (14 free parameters for 44 cases).

A companion check (exp45) on exp39's per-agent infection-count distribution
found one smooth, homogeneous exposure pool (not a bimodal high/low-exposure
split) — ruling out simple agent-level heterogeneity as the explanation, at
least within a single non-extinct simulation.

**By this point, three distinct symptom-model structures and three distinct
neonatal-priming designs had all failed to close the same gap** — evidence
this wasn't a missing single-mechanism problem.

## Phase 4 (Aug 12-13): the first correctly-signed lever

Every prior attempt used a **slum-cohort-derived** estimate of
P(symptomatic | infected) by age. Recalculating this **directly from MAL-ED
India** with an ascertainment correction (~50% asymptomatic under-detection)
gave a strikingly different profile: <6m 0.172 (vs. slum's 0.381), 6-11m
0.511 (vs. 0.407), 12-23m 0.444 (vs. 0.189) — closer to a flip than a
refinement.

**exp46** (fixed at the MAL-ED values): the first lever in 15 experiments to
move *both* problem bins in the correct direction — but it overshot past
target on both sides (<6m 0.23 vs. target 0.40; 6-11m 1.96 vs. target 1.71).

**exp47** (p_symp freed *between* the slum and MAL-ED anchors, letting HM
find the interior point): **the best joint fit in the whole India arc.**

![Exp 39 → 46 → 47 — freeing p_symp between two anchors finds the best fit yet](experiments/47_india_age_psymp_interp/figures/exp39_46_47_comparison.png)

| Metric | exp39 | exp46 | **exp47** | Target |
|---|---|---|---|---|
| IR <6m | 0.59 | 0.23 | **0.48** | 0.40 |
| IR 6-11m | 1.39 | 1.96 | **1.75** | **1.71** |
| ESS (/3000) | 9.15 | 1.26 | 1.02 | — |

The joint fit is now good. But ESS is still ~1/3000 (a single dominant
particle, not a real posterior spread) — the same degenerate pattern seen
throughout this arc. That's what led to this week's investigation.

## Phase 5 (Aug 13-14): single-seed extinction scoring was hiding a real, mechanistic problem

Every India HM/posterior-resampling run to date (exp39-47) classified a
parameter point as "extinct" (and therefore fully implausible) from **one**
simulation seed, at a fit sitting on a ~79-87% baseline extinction rate — a
knife's edge where a single unlucky seed could rule out a point that's
actually viable most of the time.

- **exp48** (smoothed classifier instead of a raw single-seed signal): ESS
  1.02→2.39, extinction rate 79%→64%, much healthier HM convergence — a real
  improvement, but it traded away exp47's near-exact 6-11m fit. Helpful, not
  sufficient.
- **exp50** asked the question directly: take 10 diverse India parameter
  points already classified "extinct" from one seed, and re-run each at 10
  fresh seeds. **100/100 went extinct, deterministically** — every point
  died within ~2 years regardless of how hot or mild its initial epidemic
  wave was. This ruled out "just bad luck" as the whole explanation and
  pointed at something more fundamental about the population itself.
- **exp51** tested that directly: hold parameters fixed, scale the simulated
  population from 40,000 (used throughout this entire project, all three
  sites) up to 100k/200k/400k. Population size **does** rescue extinction —
  but only above a sharp threshold in transmission rate, not gradually:

  | Transmission rate (R0*) | 40k | 100k | 200k | 400k |
  |---|---|---|---|---|
  | Low (R0≈2.0) | 100% extinct | 100% | 100% | 100% |
  | Mid — India's actual best-fit region (R0≈3.0) | 100% | 100% | 80% | 50% |
  | Higher (R0≈4.1) | — | 0% | 0% | 0% |

  \* corrected for each person's ~7 daily contacts and the ~90% transmission
  suppression during the ~8-day asymptomatic phase of infection — a naive
  calculation ignoring both gives a misleadingly low R0≈1.5.

  ![Extinction rate and time-to-extinction vs population size, by R0](experiments/51_india_population_size/figures/ccs_rescue_by_beta.png)

- **exp53** went one level deeper: does accounting for post-infection
  immunity (people who survive the first epidemic wave are harder to
  reinfect) explain the threshold more cleanly? Computing the *effective*
  reproduction number right after the initial wave, all three tested points
  are still comfortably above 1 (1.6, 1.8, 2.0) — yet empirical extinction
  risk at N=400k still swings from 100% to 50% to 0% across that same range.

  ![Mean-field Re (all above 1) vs actual extinction risk](experiments/53_india_effective_r/figures/re_after_wave.png)

**The mechanism:** this is not a deterministic viability question. A
textbook-supercritical epidemic (Re clearly above 1) can still die out by
chance with real probability when it's only *modestly* supercritical,
because the number of infections surviving the trough between the initial
wave and any recovery is small and finite — and that survival probability is
exponentially sensitive to exactly how far above 1 the effective R sits.
That's why a 2-3x change in transmission rate flips the outcome from "always
dies" to "always survives," and why population size matters (it sets how
many individuals are available to carry the chain through the bottleneck,
not the reproduction number itself).

For context: Bangladesh's fit (checked directly from its own HM/posterior
data) has an extinction rate of 37.5% (`infnum`, its selected symptom model)
to 65.9% (`age`) — real, but well below India's 79-87%, because Bangladesh's
much higher fitted transmission rate sits well clear of this knife-edge.

## Phase 6 (Aug 14, in progress): the fix

Rather than chase a sharper analytical threshold (exp53 shows that's the
wrong lever — the relevant quantity is a stochastic survival probability,
not a formula), we replaced single-seed extinction scoring outright, in both
pipeline stages:

- Each parameter point now runs **5 independent seeds**; the surviving
  fraction becomes a new calibration target (target = 1.0), and the other
  targets are averaged across survivors only.
- At the posterior-resampling stage, the likelihood is weighted by that same
  survival fraction — a standard "hurdle model" combination (real cohort
  data implicitly assumes the population *did* survive, so a parameter
  point's support should be discounted by how likely it was to survive at
  all).

**exp52** re-runs exp39's exact model configuration with this mechanism
active, on a dedicated 160-core VM with early-termination for extinct
replicates (so the ~5x extra simulation cost isn't wasted re-simulating
already-dead populations for their full 10-year span). Wave 1 of 6 is
running now.

## Where this leaves us

1. **Once exp52 finishes**: does ESS improve materially over exp39/47/48?
   That's the direct test of whether single-seed extinction noise has been
   corrupting the search this whole time, versus the <6m/6-11m tension being
   a separate, still-open problem.
2. **If exp52 looks healthy**: immediately follow up with the widened
   p_symp bracket (exp49's staged design, using a corrected slum-cohort
   anchor found this week) on top of the survival-vote fix, to re-attempt
   closing the <6m/6-11m gap on a search that isn't fighting single-seed
   noise.
3. Proposing to reconvene once exp52 (and ideally the exp49-style
   follow-up) are in — expecting that to be next week.

## Open questions

- Is the <6m/6-11m tension itself a calibration problem (the exp46/47 line,
  now to be re-tested under exp52's cleaner search) or a deeper structural
  one (e.g. a genuine two-strain population — the persistent community
  neonatal strain and wild-type strains as separate, weakly cross-protective
  populations)? Still open.
- Should the India pipeline also move to a larger simulated population
  (100-200k agents) in addition to the multi-seed vote? exp51 suggests
  either lever works for points near the threshold; the survival vote is
  cheaper per point, but the two aren't mutually exclusive if the vote alone
  doesn't fully resolve the ESS problem.
- Is this a data-adequacy problem (44 Vellore cases jointly constraining
  9-14 parameters) as much as a structural one? Worth weighing leaning on
  the larger-N Tamil Nadu surveillance data against continuing to refine
  against the sparse Vellore cohort.
