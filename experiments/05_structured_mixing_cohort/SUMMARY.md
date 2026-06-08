# Exp 05 — Structured Mixing + Infection-Number Severity — SUMMARY

**Question.** Can age-structured low-infant-exposure mixing + infection-number
severity + Erlang maternal jointly fit MAL-ED Bangladesh (IR peak, KM timing,
~10% repeats), with a mechanism chosen to generalize across FOI?

**Setup.** 3-group `ss.MixingPools` (inf 0-1 / young 1-5 / rest 5+), infection-
number symptom severity, Erlang maternal, cohort observation + dropout. 1000 draws,
40k agents, capybara (0 fail). Prior re-centered via a 36-cell range-check pilot.

## Result — the structured mixing was the wrong move; it induced a knife-edge

**The MixingPools dynamics are bimodal (extinct or saturated), with nothing in
between.** `ever-infected` histogram: 119 draws at 0–0.1, **0 draws in 0.1–0.7**,
881 at 0.7–1.0. This is a real dynamical knife-edge, and it is caused by the
**mixing structure, not immunity**: the matrix is one giant `young→young = 30`
cell on a flat 0.5 background — a strongly-connected near-isolated reservoir whose
internal R₀ flips between saturate and die.

**Homogeneous mixing (exp 04, RandomNet) is smooth and sits near the data at low
FOI** — `base_beta∈[0.05,0.07]` gives ever-infected 0.71 (data ~0.63) and repeat
0.23 (data ~0.2). So the model is *not* fundamentally far off; the over-infection
is a FOI-level issue, and the apparent "structural tension" was an artifact of the
reservoir.

**Incidence (the primary target): shape right, magnitude undershot.** Best exp 05
IR_symp `[3.35, 4.1, 2.8]` (logSSE 0.42) vs target `[1.91, 5.37, 2.35]` — peak in
the right bin (6-11m) but not sharp enough (peak ~1.5× neighbours vs the data's
~2.8×). Note exp 03 (steady-state, age-symptom) draw #314 *did* match the magnitude
(`[1.34, 5.35, 2.61]`), so the sharp peak is reachable — the **cohort + surveillance-
detection framing flattens it**, implicating the observation model, not just
transmission/immunity.

## Figure

![Mixing knife-edge](figures/mixing_knife_edge.png)

## Corrections / retractions (recorded honestly)

- **Retract the "missing immune sink / LTI" diagnosis** from the mid-run notes. It
  was induced by the MixingPools bimodality. The leaky floored immunity
  (`sus_after_1/2/3+`, permanent in `use_fixed_susceptibility` mode — no waning) is
  reasonable; homogeneous mixing with it behaves well.
- The earlier "hyperendemic-or-extinct = structural tension" framing over-claimed:
  it's specific to the reservoir matrix; homogeneous mixing has a smooth stable
  endemic.

## Observations

- The matrix is both **too spiky** (one dominant cell → knife-edge) and **too
  coarse** (infants are one 0–1y block → no lever on the `<6m` vs `6-11m` shape,
  which is therefore all maternal-immunity's job).
- Incidence-magnitude and reinfection are **aligned, not in tension**, under smooth
  mixing: lowering FOI to match incidence magnitude also pulls repeats toward the
  data (exp 04 evidence). The tension was a reservoir artifact.

## Next (exp 06)

Isolate the `young→young` cell: **sweep `young_reservoir` from 1 to 15** (30 is too
hot → knife-edge), keep infant exposure low and background homogeneous, and **rely
on acquired immunity (the `sus_after_*` ladder), not mixing, to produce the 12-23m
decline.** Question: what reservoir level gives a smooth (non-bimodal) endemic with
a 6-11m peak? Score primarily on IR-by-age (bin + magnitude). Also flag for a later
experiment: check whether the cohort/surveillance observation model is flattening a
peak that the transmission/immunity actually produces.
