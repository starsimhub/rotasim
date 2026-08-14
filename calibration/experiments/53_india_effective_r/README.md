# Exp 53 — India Vellore: effective R after the initial wave

**Question.** exp51's "Next" section flagged computing the *effective*
reproduction number after the initial wave (accounting for post-infection
susceptibility depletion) rather than relying on the naive fully-susceptible
R0 used there. Does this sharpen the R0-threshold story, or reveal something
else going on?

No new simulations — reuses exp50/51's already-collected `peak_frac` (the
attack rate of the initial synchronized wave) and the corrected R0 formula
from exp51, for the same 3 parameter points (`orig_idx` 2320/1720/1072).

**Method.** Right after a single fast synchronized wave (the whole
population is exposed within weeks — see exp50), almost everyone who was
infected has exactly one prior infection, and everyone else is still fully
naive. So the population-average relative susceptibility right after the
wave is
```
<rel_sus> = (1 - peak_frac) * 1.0 + peak_frac * sus_after_1
Re_after_wave = R0 * <rel_sus>
```
using each point's own fitted `sus_after_1` (susceptibility multiplier after
a first infection) and `peak_frac` at N=400k.

See SUMMARY.md for the result.
