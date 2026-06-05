# Exp 02 — Epidemiological Burn-in Check

**Question.** Has the *transmission system* reached endemic steady state by the
time the calibration window opens (year 5)? Exp 01 settled the *demographic*
burn-in — age structure equilibrates within ~1 year. But the disease dynamics
are slower: prevalence, the circulating strain mix, and especially the
population immunity distribution (which builds up over multiple infection
cycles) can take much longer to stabilise. If the science window (years 5–10)
opens during a transient — e.g. immunity still accumulating, or prevalence
still settling from the initial seeding — then the IR-by-age and first-infection
targets are being matched against non-stationary model output, and the fit
chases a moving baseline.

**Plan.** Run the full disease model (Bangladesh demographics, a representative
parameter set from a recent trial) at 20k agents for the full 10-year span.
Track over time: overall prevalence / weekly incidence, number of co-circulating
strains, and a summary of the population immunity distribution (e.g. mean
susceptibility, fraction with ≥1 prior infection). Plot each against time and
mark the year-5 window boundary. Judge visually and numerically whether each
series is flat (within noise) before year 5.

**Success criteria.**
- Good: prevalence, strain count, and immunity summaries are all stationary
  (flat to within run-to-run noise) by year 5 — the calibration window sits on
  a clean endemic plateau.
- Failure: any series still trending at year 5. If so, the burn-in is too short
  and either (a) the window should start later, or (b) the model should be
  seeded closer to its endemic state. Quantify how long burn-in actually needs
  to be.
