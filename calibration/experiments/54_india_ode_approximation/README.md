# Exp 54 — India Vellore: deterministic ODE reduction of the ABM

**Question.** Run alongside exp52 (multi-seed survival vote, in progress) as
an exploratory side-track: build a deterministic ODE that reproduces the
ABM's transmission mechanics (order-based reinfection risk, maternal
protection, two-phase symptomatic/asymptomatic infectiousness) faithfully
enough to study steady-state/threshold behaviour numerically, much faster
than running the stochastic ABM. Can this (a) confirm exp53's finding that
all three exp50/51 test points have Re>1 deterministically, (b) locate the
exact deterministic Re=1 threshold cheaply, and (c) say anything sharper
about *why* extinction risk varies so much between points that are all
comfortably supercritical?

**Scope, deliberately limited.** This ODE reproduces transmission dynamics
only — it does not model the age_binned/infnum symptom-detection layer,
which (confirmed by reading `rotasim/rotavirus.py` directly) is a pure
observation-layer construct that never feeds back into transmission. So this
tool can speak to persistence/extinction, not to the cohort IR-by-age fit.
See `ode_model.py`'s docstring for the full compartment structure and every
simplification relative to the real ABM (the maternal titer/Hill curve is
approximated as an Erlang(6) chain matched on mean protected duration; the
post-recovery temporary strong-immunity period — a real, fixed ABM
mechanism, mean 91 days, that prior back-of-envelope R0/Re calculations in
this project omitted — is included).

**Method.** `ode_model.py` defines the compartments and right-hand side;
`run.py` (1) runs the same 3 parameter points as exp50/51 (`orig_idx` 2320,
1720, 1072) forward 15 years and checks whether they reach a stable positive
endemic equilibrium, (2) sweeps `base_beta` (holding 1720's other parameters
fixed) to find the deterministic critical point, and (3) computes the
effective R *at the post-peak trough* (using the ODE's actual susceptible
composition at that moment, not the naive fully-susceptible R0) as an input
to a simple branching-process extinction-probability approximation.

See SUMMARY.md for results.
