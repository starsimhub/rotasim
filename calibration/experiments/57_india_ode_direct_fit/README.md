# Exp 57 — India Vellore: fit the ODE reduction directly against the real composite likelihood

**Question.** exp52 (survival-vote HM re-run) confirmed the &lt;6m/6-11m tension
is structural, not a search or mixing problem — but each India HM run still
costs a multi-day wave-by-wave job (45,000+ sims for exp52). exp53/54 showed
the ABM's extinction/persistence behavior is a knife-edge phenomenon the ODE
reduction can't capture — but extinction isn't a real-world concern here (the
disease doesn't actually die out), so that ODE limitation doesn't block using
it for the fit itself. exp55/56 validated the age-structured ODE's equilibrium
fidelity against the real all-infection IR-by-age targets. This experiment
asks: does adding a detection layer (`cohort_model.py`) on top of the
validated ODE let us fit the *exact* composite likelihood used in
`trajectory_select.py` directly, via cheap deterministic optimization, instead
of waiting on expensive noisy ABM HM waves?

**Plan.** For a given 12-parameter draw (age_binned + titer maternal,
`hm_calibrate.py`'s `bounds_for('age_binned','titer')`): (1) run the
age-structured equilibrium model (`ode_model_age.py`) to get the population's
steady-state FOI, (2) run the birth-cohort model (`cohort_model.py`) forward
from age 0 using that FOI to get the age-at-first-detection survival curve
and detected-count distribution at the real cohort's empirical exit ages,
(3) compute the exact composite log-likelihood used in
`trajectory_select.py` (Poisson symptomatic-IR-by-age + binomial
repeat-detected fraction + survival log-likelihood on age-at-first-DETECTED-
infection), (4) optimize over all 12 parameters via `scipy.differential_evolution`
(global, gradient-free — matches HM's own experience of a possibly
multimodal/degenerate surface).

**Status note (retroactive, written 2026-08-17 after the fact):** this was
first launched locally on a laptop with `differential_evolution(...,
workers=-1)`. It printed the exp47-MLE sanity check (logL=-326.74) and began
optimizing, then the machine froze and the run died silently — no traceback,
no `fit_result.json`. Diagnosis: `workers=-1` + a fairly heavy per-eval ODE
simulation (`n_agents=40_000`, two solves per likelihood call) saturating all
cores on a laptop is a bad combination, not a code bug. Re-launching on the
zebra VM (160 cores, non-spot, per `reference_zebra_vm`) under `tmux` instead,
so it survives an SSH drop and isn't competing with the laptop's own load.

**Success criteria.** Does the ODE-direct MLE land near exp52's posterior mode
(IR&lt;6m ~0.6, IR6-11m ~1.4, both still off-target) — corroborating that this
is the same likelihood surface HM has been exploring, just found faster? Or
does the deterministic global optimizer find a materially better point HM's
noisy multi-seed sampling missed? Either result is informative: the first
validates the ODE as a fast proxy for future India HM design work; the second
would suggest the ABM HM runs so far have been under-exploring the space.
