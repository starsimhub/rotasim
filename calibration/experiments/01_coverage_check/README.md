# Exp 01 — Prior-predictive coverage check: are symptomatic incidence and age-at-first-infection jointly achievable?

**Question.** Across every symptom-model and maternal-waning variant tried so
far (age-only logistic; infection-number per-infection probabilities;
age+infection linear and categorical-offset; single-phase, two-phase, and
Erlang maternal), the model cannot fit MAL-ED Bangladesh **symptomatic
incidence-by-age** and **age-at-first-infection** at the same time — a good fit
to one is a poor fit to the other (a Pareto tension). All of that was
optimizer-driven (Optuna/TPE point fits, i.e. workflow steps 9–10). This
experiment asks the prior instead of the optimizer: drawing parameter sets
*independently* from the prior, does **any single set** land *both* targets at
once? This is the coverage check (step 3) that was skipped before the
iteration began, and it distinguishes a **structural** mismatch (the model
literally cannot produce both) from a **search/likelihood** failure (it can,
but TPE and/or our squared-log-IR GOF aren't finding it).

**Plan.** 50 independent prior draws (NOT Optuna — uniform over the ranges,
log-uniform for `base_beta`, respecting sus monotonicity), **1 replicate each**
(the binary "can it reach the data" question; replicate noise is a separate
`model-setup` concern). Symptom model = `age_and_infection_offsets` — the most
flexible (age logistic + categorical per-infection offsets; it nests age-only),
so a failure implicates the **infection dynamics, not the symptom model**.
Maternal = Erlang n=6; ranges per `calibration/CLAUDE.md`. Run on covaguest
(~40 workers). Targets: symptomatic IR in 4 age bins (`<6m`=1.91, `6-11m`=5.37,
`12-23m`=2.35, `24-35m`=0.14 /100 PM) and age-at-first-detected-infection
quartiles (Q25/med/Q75 = 5.1/8.0/11.2 mo). Raw per-draw results → `outputs/`
(incremental JSONL, so a spot-VM eviction doesn't lose the run); coverage
figure → `figures/`. Reports **marginal** coverage (each target inside the draw
envelope) and **joint** coverage (a single draw near both).

**Success criteria.** *Joint coverage exists* (≥1 draw simultaneously near both
targets) → the tension is a **search/likelihood** problem; the next experiment
keeps the structure and switches to a count-based likelihood for the sparse
bins and/or better search — NOT a structural change. *No joint coverage*
(marginal per-target coverage may still pass) → the tension is **structural**;
no symptom-model or optimizer change will fix it, and the next experiment tests
a **structural** change (age-modulated protection against *infection* — the
original hypothesis). Either outcome is a clean, decision-grade result; "no
joint coverage" is the stronger, more publishable finding and is the expected
one given the iteration history.
