# Exp 33 — India Vellore: age_and_infection symptom model

**Question.** Exp 31 showed infnum-only cannot resolve the <6m / 6-11m IR tension in
Vellore. Exp 32 tests whether fixing p_symp from Vellore biweekly data resolves it within
infnum. This experiment asks the complementary question: does adding a continuous age term
to the symptom model — P(symptomatic | infected, age, order) via the age_and_infection
model — provide the extra flexibility needed to fit all three IR bins simultaneously?

The age_and_infection model was tested for UK surveillance (exp 28) and found under-identified
there (G²=710, ESS=1), because the UK age gradient in symptomatic fraction is weak. India is
the opposite case: the Vellore biweekly cohort shows a steep decline in P(symp|infected) from
0.407 at 6-11m to 0.189 at 12-23m, with a muted <6m value (0.381). This sharper gradient
may make the age coefficient identifiable where it was not for UK.

**Plan.** Run HM (age_and_infection model, --fix-titer-shape, --all-targets, MALED_SITE=india,
NEO_PRIME=1) with the same 5 symptomatic + repeat + first-infection targets as exp 31 hm_neoprime.
The age_and_infection model adds an age coefficient alongside the infnum order parameters,
increasing parameter count relative to exp 32. 3 waves × 1500 samples on covaguest in parallel
with exp 32. Follow with trajectory selection (n=3000).

Compare to exp 32 (fixed p_symp, infnum) and exp 31 neoprime baseline (logL −287.95):
- If age_and_infection beats exp 32 logL substantially → age term is doing real work
- If exp 32 ≈ exp 33 → fixing p_symp is sufficient and the simpler model is preferred
- If both beat exp 31 → symptom under-identification was the main problem

**Success criteria.** Best-fit IR by age closer to observed on all three bins simultaneously,
and repeat fraction ≥ 0.12. Secondary: ESS > 1 (the age coefficient being identifiable from
the India data would be visible as a less-degenerate posterior than the ESS=1 seen in exp 31).
