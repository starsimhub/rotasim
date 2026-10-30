# Exp 49 — India Vellore: p_symp free, widened 6-11m bracket (corrected slum anchor)

**Question.** AK found a calculation error in the slum-cohort 6-11m p_symp
estimate used throughout exp31-48: the corrected value is **0.593**, not
0.407 — above `FIXED_AGE_PSYMP_MALED`'s 0.511, flipping which anchor is
higher for this bin. exp47's fitted `p_symp_age_6_11` landed at 0.548,
pinned against the (0.35, 0.55) bracket's upper bound — consistent with the
search wanting to reach the corrected 0.593 anchor, which that bracket
didn't cover. Does widening the bound let the fit reach a value nearer the
corrected anchor, and does IR 6-11m/&lt;6m improve further, or was 0.548
already close to the true optimum (in which case widening changes little)?

**Design.** `AGE_PSYMP_INTERP_V=2` (new in `hm_calibrate.py`) widens
`p_symp_age_6_11`'s bound to `(0.40, 0.60)` — covers the corrected slum
anchor (0.593) with a small margin, up to AK's suggested ceiling (~0.62 was
mentioned; used 0.60 to leave a touch of margin without reopening it to an
uninformed range). `p_symp_age_0_6` and `p_symp_age_12plus` bounds unchanged
from exp47/48 (`(0.15,0.40)`, `(0.15,0.50)`) -- AK's correction was specific
to the 6-11m bin.

exp39/46/47/48 deliberately NOT rerun with this correction — kept as-is for
clean historical comparability. This experiment is the first to use it.

**Staged, not yet run:** waiting on exp48's result (same design + the
extinction-classifier fix) before deciding whether to layer the widened
bound on top of exp48's approach or exp47's, or run both.

**Run (once launched):** `--model age_binned --maternal titer --all-targets`,
`MALED_SITE=india NEO_PRIME=1 EXT_PENALTY=1 AGE_PSYMP_INTERP=1
AGE_PSYMP_INTERP_V=2` (+ `EXT_CLASSIFIER=1` if building on exp48 instead of
exp47), 6 waves x 1500 samples, `--early-stop`.

**Success criteria:** does `p_symp_age_6_11` move meaningfully above 0.548
(toward the corrected 0.593 anchor) and no longer sit pinned at the new
upper bound; does that further improve IR 6-11m/&lt;6m, or was exp47 already
near the true optimum within measurement noise.
