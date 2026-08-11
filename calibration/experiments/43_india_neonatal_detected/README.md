# Exp 43 — India Vellore: neonatal priming as a real detectable event

**Question.** Exp 39-42 confirmed a structural Pareto tension (cohort fit vs. VE
plausibility) under `age_binned` + `NeonatalPriming` (`NEO_PRIME=1`). Re-reading the
analyzer code while planning this experiment turned up why priming couldn't have
helped: `MALEDCohort._symp_prob` for `age_binned` depends only on age, never on
infection order — so the *only* effect `NeonatalPriming` had (bumping a primed
child's next-infection order by +1) was a complete no-op under `age_binned`. With
`sus_effect=0` (the default), priming was silently inert for all of exp39-42; the
neonatal-priming hypothesis was never actually tested against the Vellore data in
those runs.

Separately, AK confirmed (2026-08-11) the biological mechanism, narrow reading:
maternal titer does **not** block infection with the specific antigenically-distinct
neonatal-adapted community strain (G10P[11]/116E), and does not protect much against
disease in general — but an infection occurring while maternal titer is still high
is less likely to seroconvert (build durable immunity against the different wild-type
strains that drive later disease). This is a claim about *this one bespoke event*,
not a general statement that titer fails to protect against ordinary community-strain
exposure — so the general titer→susceptibility Hill function in `immunity.py` is
untouched.

**Code changes made (this experiment, committed with it):**
1. `rotasim/analyzers.py`, `MALEDCohort`: priming is now a **real** early infection —
   it increments `n_inf` (true infection order) and is checked for detection via the
   *same* asymptomatic-surveillance pathway (`_p_surv(age) * eia`) as any other
   subclinical infection. It is never symptomatic (`cases_symp` is never incremented
   for it), consistent with the biology. The now-redundant "+1 order if primed" bonus
   on the child's next real infection is removed — `n_inf` already reflects it.
2. `calibration/calibrate_maled.py`, `NeonatalPriming` docstring updated to describe
   the mechanism accurately (was stale: said "NOT a detected case").
3. **No change to `sus_effect`** — stays fixed at 0.0 (narrow interpretation: this
   event doesn't confer strong future protection, so we are *not* pushing it toward
   1.0 as an earlier plan considered). No new free parameters: `p_neo=0.5`,
   `age_weeks=2.0`, `sus_effect=0.0` all remain literature-fixed, not fitted — MAL-ED's
   own data structurally cannot identify them (same non-identifiability argument as
   before).

**Smoke test (single seed, 40k agents, exp39 posterior row 0 params, directional
only — see caveat below):** repeat_frac 0.089→0.161 (target 0.138, moved toward it);
Q25 first-detected-infection 22.6→8.3mo (target 15.1 — swung from a large overshoot
to a large undershoot). Confirms the mechanism has real bite now, not inert. **Caveat:**
this is a single paired seed; the new code draws extra random numbers from
`MALEDCohort.rng` earlier in the stream (for the priming-detection roll), which
reshuffles every downstream symptom/detection draw for the rest of that seed's cohort
— so the single-seed delta conflates the intended mechanism with RNG-stream-shift
noise. Do not read these numbers as the expected refit outcome; only the full
multi-seed HM run below marginalizes that out properly.

**Plan.**
- Run HM: `--model age_binned --maternal titer --fix-age-psymp --all-targets`,
  `MALED_SITE=india NEO_PRIME=1` (same flags as exp39 — the only change is the code
  behind `NEO_PRIME`, not the run configuration).
- No bounds change needed — India's `base_beta` ceiling is already 1.5 (vs UK's 0.5),
  set for exactly this "may need higher FOI" contingency; exp39 sat at ~0.12-0.16, well
  under the ceiling.
- 6 waves × 1500 samples, `--early-stop`, on zebra (check current load before
  launching — another user was running a full-core job earlier this session).
- Follow with trajectory selection (`trajectory_select.py`, n=3000), same as exp39.

**Parameters (free) — unchanged from exp39, no new free parameters:**
`log_base_beta, sus_after_1, sus_r2, sus_r3, log_titer_median, titer_gsd,
titer_half_life_days, hill_slope, maternal_efficacy` (9 parameters).

**Success criteria (vs exp39's ESS=9.15, weighted IR&lt;6m 0.59 / 6-11m 1.39 / 12-23m
0.60 / repeat 0.116 / Q25 17.5, targets 0.40 / 1.71 / 0.61 / 0.138 / 15.1):**
- ESS not much worse than exp39's 9.15 (a collapse would suggest the added mechanism
  makes the posterior harder to satisfy, not easier).
- Q25 closer to 15.1 (exp39 overshot high; the smoke test suggests this run may now
  need to guard against overshooting low instead — watch for that in the refit, not
  assumed to land in between for free).
- repeat_frac closer to 0.138.
- IR by age bins not worse than exp39, ideally closer on the 6-11m peak (the
  persistent miss across all India variants so far).

**Next:** if this resolves (or meaningfully narrows) the Pareto tension, re-run the
VE forward-prediction (exp41-style) on the new posterior to check whether direct VE
comes out plausible without needing the VE-scoring workaround from exp42. If it does
not resolve, that's still useful — it would mean the tension is not (solely) about
the neonatal-detection gap, and the broader "titer generally fails against community
strains" hypothesis (deferred, bigger structural change) would need reconsidering.
