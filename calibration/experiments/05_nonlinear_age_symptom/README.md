# Exp 05 — Non-linear (peaked) age-symptom curve + strong maternal: does it reproduce the shape?

**Question.** exp 03–04 established that the 6–11 mo peak is a **symptom-severity-
by-age** effect (mild `<6m`, worst 6–12 mo, milder after — a non-linear/quadratic
curve, per the literature/Lewnard), not an infection-rate effect, and that maternal
immunity works. exp 02's age-quadratic *failed* only because the optimizer drifted
to a monotone-declining curve paired with weaker maternal. This experiment
forward-runs **peaked** age-symptom curves with the now-verified **strong maternal**
under **homogeneous mixing**, to test whether the data shape (low `<6m` → 6–11 mo
peak → decline to ~0) is reproduced — before any calibration. See
`../04_age_structured_contacts/SUMMARY.md`.

**Plan.** Reuse `calibrate_maled._run_one_replicate` (RandomNet `n_contacts=7`,
symptom_model `age_only` = quadratic logistic in age centered at 12 mo, `beta3=0`),
with Erlang maternal (n=6, efficacy ~0.95, mean duration ~200 d). Forward-run a few
**peaked** beta sets — anchored near Lewnard's values plus a sharper peak-at-9-mo
variant — across a small `base_beta` sweep. Plot symptomatic IR by age vs. the
MAL-ED data; assess whether any combination gives the low-`<6m` / 6–11 mo-peak /
declining-tail shape. No calibration yet — this is the feasibility check (workflow
step 3) for the symptom-severity-by-age hypothesis.

**Success criteria.** A peaked age curve + strong maternal reproduces the data
shape under homogeneous mixing → confirms the hypothesis and that age-structured
contacts are unnecessary → proceed to calibration. **Failure** → the symmetric
quadratic can't carve the (slightly asymmetric) shape, or the oldest-bin decline is
too weak → add infection-number on top (the Lewnard age+infection form, exp 06).
