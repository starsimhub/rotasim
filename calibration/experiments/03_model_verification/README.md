# Exp 03 — Model verification: is the unfittable peak a bug, or a genuine homogeneous-mixing limitation?

**Question.** `02_joint_identifiability` showed no optimized fit across 5 model
families reproduces the MAL-ED Bangladesh shape (sharp 6–11 mo peak on low `<6m`
and near-zero oldest shoulders). Before adding age-structured contacts to fix it,
rule out the alternative the researcher rightly flagged: many groups fit rotavirus
incidence *without* age-structured mixing, so the failure may be an undiagnosed
**model bug or artifact** rather than a real structural need. This experiment
interrogates the model's underlying behavior to decide bug-vs-limitation. See
`../02_joint_identifiability/SUMMARY.md`.

**Plan.** Diagnostic runs on the VM (a handful of sims, no calibration):
1. **Person-time / denominator — LEAD hypothesis.** The modeled IR-by-age divides
   cases by a **cross-sectional steady-state** person-time
   (`compute_person_months_steady_state` = end-of-sim pyramid headcount × window;
   its own docstring flags this is "inaccurate for a true cohort sim"), using a
   **UK** pyramid (`uk_age_data.csv`, under-5 ≈ 6.2%). But MAL-ED is a **birth
   cohort** — every child passes through every bin, so cohort PT ≈ proportional to
   bin width, independent of any population pyramid. A wrong denominator distorts
   the incidence *shape* directly. For one fixed sim, recompute IR-by-age under
   three denominators: (a) current cross-sectional UK PT; (b) cross-sectional
   Bangladesh PT (`bangladesh_age_data.csv`, under-5 ≈ 12.5%); (c) birth-cohort PT
   (∝ bin width). Does the shape move materially — and toward the data? If so, the
   "unfittable peak" was a denominator artifact, not missing structure.
2. **Can homogeneous mixing make a peak at all?** Plot the model's **all-infection**
   incidence by age (raw infection events, *no* symptom filter) over the
   calibration window for a few parameter sets (exp-02 best-per-family + a
   `base_beta` sweep). Surge-then-decline, or only flat/monotone?
3. **Rule out the t=0 UK seeding + mechanism sanity.** Re-run with
   `init_prevalence=0` and confirm the window's incidence-by-age is unchanged (the
   UK seeding via `utils.DEFAULT_INITIAL_INFECTION_AGE_DIST` washes out 5 yr
   upstream); snapshot `rel_sus` vs age (maternal suppressing `<6m`? infection-
   number steps visible?); confirm age-at-infection is recorded sensibly.
Outputs/figures to this experiment's `outputs/` and `figures/`.

**Success criteria.** *Model sound* — UK artifacts wash out / don't matter,
mechanisms behave as intended, and the all-infection age profile is sensible but
homogeneous mixing **cannot** produce the sharp 6–11 mo peak under any reasonable
parameters → the limitation is genuinely structural (not a bug), and
age-structured contacts (exp 04) is justified. *Bug/artifact found* — a UK
artifact materially distorts the window, or a mechanism misbehaves (e.g. maternal
not suppressing `<6m`, age-at-infection mis-recorded) → fix it and re-check
whether the data fits *without* age structure, vindicating the concern. Either
outcome decides exp 04.
