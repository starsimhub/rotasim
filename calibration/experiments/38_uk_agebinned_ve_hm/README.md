# Exp 38 — UK HM (age-binned): VE + case-shape joint calibration; Bangladesh-constrained immunity

**Question.** Exp 37 runs the infnum model for the UK with VE as an identifiability anchor.
The age-binned symptom model (best Bangladesh model by exp25/26, ESS=61.5 vs infnum's 107.9)
may behave differently under the same identifiability fix. In Bangladesh the two models fit
similarly; in the UK the symptom parameterisation could interact with beta and sus_after
differently. This experiment runs the same VE + shape joint calibration as exp37 but with
the age-binned symptom model.

**Plan.**
- Same design as exp37 except symptom_model='age_binned' (parameters: p_symp_age_0_6,
  p_symp_age_6_11, p_symp_age_12plus instead of p_symp_1/p_r2/p_r3).
- Bangladesh bounds from exp25 (age-binned corrected, ESS=61.5) posterior medians ±50%:
  sus_after_1=[0.365, 0.99], sus_r2=[0.316, 0.947], sus_r3=[0.320, 0.96].
  Age-bin symptom bounds from exp25 medians ±50%:
  p_symp_age_0_6=[0.242, 0.726], p_symp_age_6_11=[0.283, 0.848], p_symp_age_12plus=[0.157, 0.470].
- Same VE target: ve_overall=(0.74, 0.05), same vaccine config (2+4mo, 90% cov, rp=0.85).
- Same init override, same 40k agents, 3 HM waves × 2000 samples on zebra.

**Success criteria.**
- Non-empty NROY jointly satisfying case shape and VE.
- Compare NROY volume and sus_after posterior vs exp37 (infnum) — consistent across symptom models?
- If age-binned produces a tighter sus_after posterior, it is the preferred model for UK.
