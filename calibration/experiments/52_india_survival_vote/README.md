# Exp 52 — India Vellore: re-run the age_binned HM wave with the multi-seed survival vote

**Question.** exp39 (`age_binned`, p_symp fixed per age bin, titer maternal,
NeonatalPriming active, all-targets HM) is India's best-mixed HM run so far
(ESS=9.15/3000), using the single-seed `log_symp_ir_sum` extinction sentinel
(`EXT_PENALTY=1`). exp50/51 showed that near the viable R0 threshold,
extinction is genuinely seed-dependent (not just noise on a deterministic
outcome) — a single unlucky seed can rule out a parameter point that would
survive most of the time. `hm_calibrate.py`/`trajectory_select.py` were just
rewritten to replace the single-seed sentinel/classifier with a 5-seed
survival vote (`SURVIVAL_VOTE=1`): `frac_survived` (Laplace-smoothed, target
1.0) is a new HM target, and at trajectory selection the composite
likelihood among survivors is multiplied by `frac_survived` (a hurdle-model
decomposition — see commit message / `hm_calibrate.py` header comment for
the full justification).

This experiment re-runs exp39's exact model configuration with
`SURVIVAL_VOTE=1` replacing `EXT_PENALTY=1`, to see whether the new
mechanism produces healthier mixing (higher ESS, less NROY collapse) than
the old sentinel/classifier approaches (exp39 ESS=9.15, exp47 ESS=1.02,
exp48 ESS=2.39 — all single- or classifier-scored). **This is a fresh HM run
(new checkpoint), not a literal `--resume`** — the checkpoint's emulator
bank and observation schema were built around `log_symp_ir_sum`, which no
longer exists; adding `frac_survived` as a wave-1 target requires starting
over, not continuing the old checkpoint.

**Design.** Same as exp39 exactly, except the extinction-handling flag:
```
MALED_SITE=india NEO_PRIME=1 SURVIVAL_VOTE=1 \
  python hm_calibrate.py --model age_binned --maternal titer \
    --fix-age-psymp --all-targets --n-samples 1500 --max-iter 6 \
    --out-dir experiments/52_india_survival_vote/outputs/hm
```
**Cost note:** 5 seeds/draw means ~5x the simulation count of exp39 per
wave (1500 samples x 6 waves x 5 seeds = 45,000 sims vs. exp39's 9,000) — a
real compute commitment, run on zebra (160 cores, non-spot).

**Success criteria.** Does ESS improve materially over exp39/47/48? Does the
fitted posterior move meaningfully on the dimensions the old mechanism was
struggling with (IR6-11m, Q25 first-infection)? If this looks healthy,
follow up by adding the widened p_symp bracket (`AGE_PSYMP_INTERP=1
AGE_PSYMP_INTERP_V=2`, exp49's staged design, corrected slum anchor) on top
of this same survival-vote mechanism.
