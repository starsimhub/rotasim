# Exp 27 — infnum + titer (fixed shape) on the CORRECTED maternal

**Question.** The infnum member of the clean same-maternal pair, re-run on the **corrected
maternal** (PR #38: per-infant titer now persists instead of being redrawn every step). Exp 24
(infnum, ESS 47) ran on the *buggy* maternal, so its posterior isn't valid for the exp-20 VE
comparison — and the bug fix shifts first-infection timing later (+~2.4 mo in a single-seed
check), which is exactly the channel infnum relies on. We need a corrected-maternal infnum
posterior to pair with the corrected-maternal age member (exp 25 binned and/or exp 26 quadratic).

**Plan.** Same pipeline as exp 24, only the maternal is now correct. Run on **raccoon** (uv venv,
120 cores):
```
hm_calibrate.py --model infnum --maternal titer --fix-titer-shape --all-targets \
    --out-dir experiments/27_infnum_titer_fixedshape_corrected/outputs/hm \
    --early-stop --n-samples 5000 --max-iter 8 --resume
```
then `trajectory_select.py --model infnum --maternal titer --fix-titer-shape` →
`reweight_overdispersed.py --model infnum --exp-dir 27_… --phi 3 --rho 0.10`
(auto-chained by `orchestrate_exp27.sh`).

**Success criteria.** A usable posterior (non-degenerate reweighted ESS) hitting all 5 targets.
Compare to exp 24 (buggy maternal) to see how much the persistence bug moved the infnum fit —
especially first-infection timing (exp 24 sat at 10.65 vs target 12.12; the fix should push it
later/closer). With the corrected-maternal age member, this completes the clean same-maternal
pair for the exp-20 VE comparison.

Cross-refs: [`../24_infnum_titer_fixedshape/`](../24_infnum_titer_fixedshape/) (buggy maternal,
ESS 47), [`../25_age_binned_titer_fixedshape/`](../25_age_binned_titer_fixedshape/),
[`../26_age_quadratic_titer_fixedshape/`](../26_age_quadratic_titer_fixedshape/).
</content>
