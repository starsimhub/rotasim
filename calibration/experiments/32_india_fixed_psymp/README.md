# Exp 32 — India Vellore: fixed p_symp from Vellore biweekly data, refit FOI + maternal

**Question.** Exp 31 showed that infnum with free symptom parameters cannot fit the Vellore
MAL-ED cohort — the model fails simultaneously on <6m IR (overshoots) and 6-11m IR
(undershoots), suggesting the 8-parameter space is under-constrained on the symptom side.
The Vellore biweekly cohort (near-complete detection) provides direct estimates of
P(symptomatic | infected) by age bin: <6m=0.381, 6-11m=0.407, 12-23m=0.189, 24-35m=0.122.
These imply infnum parameters approximately p_symp_1≈0.40, p_symp_2≈0.19, p_symp_3+≈0.12.
This experiment fixes p_symp at those values and asks: with the symptom structure pinned,
can HM find FOI + maternal parameters that reproduce the MAL-ED targets?

**Plan.** Run HM (infnum, --fix-titer-shape, --all-targets, MALED_SITE=india, NEO_PRIME=1)
with p_symp_1, p_r2, p_r3 fixed at Vellore biweekly values and removed from the fitted
parameter set. Free parameters reduce from 8 to 5: log_base_beta, sus_after_1, sus_after_2
(sus_r2), sus_after_3 (sus_r3), maternal_efficacy. Neonatal priming kept on (p_neo=0.5 fixed
from literature) as it is biologically supported. 3 waves × 1500 samples on covaguest.
Follow with trajectory selection (n=3000) and compare best-fit logL and finite-trajectory
fraction to exp 31 neoprime baseline (logL −287.95, 566/3000 finite).

**Success criteria.** A good result is: (1) the best-fit IR by age bin moves closer to
observed on all three bins simultaneously (currently <6m overshoots and 6-11m undershoots),
and (2) repeat-detected fraction ≥ 0.12. A failure — same structural miss persists despite
fixed symptoms — would implicate the susceptibility/FOI structure itself and motivate a
combined age×infnum symptom model.
