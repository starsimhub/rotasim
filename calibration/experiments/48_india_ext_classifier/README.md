# Exp 48 — India Vellore: smoothed extinction-probability classifier

**Question.** Every India HM run so far (exp39-47) scores extinction from a
**single seed per sampled parameter point** — a parameter set with, say, a
15% true extinction probability has an 85% chance of surviving its one
assigned seed, but on the unlucky draws gets the full sentinel penalty
(`log(1e-9)≈-20.7`) and is treated by the emulator as if that whole
neighborhood is extinct. Given this fit sits at a ~85-87% baseline extinction
rate (a knife's edge), this single-seed noise could be spuriously ruling out
good, low-risk regions early — `log_symp_ir_sum` is cycled first and shapes
wave-1 NROY before anything else, so this isn't just scatter, it actively
narrows the search. Does replacing the raw single-draw sentinel-mixed
regression target with a smoothed extinction-probability estimate change the
ESS collapse pattern seen in every exp39+ variant?

**Design (AK + a separate Claude conversation, 2026-08-13).** No new
simulations needed — same wave budget. Instead of feeding the raw
`log_symp_ir_sum` value (continuous for viable sims, a `-20.7` sentinel for
extinct ones) into HM's emulator, fit a **logistic classifier**
(`StandardScaler` + `LogisticRegression`, `sklearn`) on ALL `(params,
extinct 0/1)` pairs accumulated across every wave of the run so far, and feed
its **predicted P(extinct)** for the current wave's points instead. This
borrows information across nearby already-sampled points — the way logistic
regression normally works, no repeated trials needed at each point — giving
a smooth probability surface instead of one noisy Bernoulli draw per point.
Implemented in `hm_calibrate.py`: `EXT_CLASSIFIER=1` env var (requires
`EXT_PENALTY=1`), accumulator lives in `make_simulator`'s closure (persists
across waves within one run), falls back to the empirical extinction rate
until ≥20 points AND both classes are seen. Target rescaled to `(0.0, 0.15)`
since the feature is now a probability, not `log(ir_sum)`.

**This is exp47's exact design (`age_binned`, `AGE_PSYMP_INTERP=1`, free
p_symp bounded to the slum↔MAL-ED bracket) plus this one change** — isolates
the extinction-scoring fix as the only variable between the two.

**Run:** staged to launch automatically once exp47's HM + trajectory
selection complete (same machine, sequential — not run concurrently).
`--model age_binned --maternal titer --all-targets`, `MALED_SITE=india
NEO_PRIME=1 EXT_PENALTY=1 EXT_CLASSIFIER=1 AGE_PSYMP_INTERP=1`, 6 waves x
1500 samples, `--early-stop`.

**Success criteria:** does ESS improve over exp47's (whatever it turns out
to be) at a comparable or better fit — the specific test of whether
single-seed extinction noise has been artificially suppressing ESS across
this whole arc, or whether the low ESS is a genuine feature of the joint
constraint regardless of how extinction is scored.
