# Exp 06 — Titer Maternal + Adult Reservoir: Sharp IR Peak Achieved — SUMMARY

**Question.** With the IBM titer-based maternal model (peakiness from titer spread +
Hill slope), `young_reservoir` swept low (off the knife-edge), and an adult-adult
reservoir for persistence, can we reproduce MAL-ED Bangladesh's *sharp* 6-11mo IR
peak at a smooth endemic?

**Setup.** Cohort emulation (MixingPools, infection-number severity, surveillance
detection + dropout) + `maternal_immunity_model='titer'`. 1000 draws, 40k agents,
capybara (1229s, 0 fail, 77% persistent). New `adult_contacts` lever (rest→rest,
swept 0.5–2.5) sustains circulation decoupled from the child targets.

## Result — YES. The incidence-first goal is met.

**Best draw: `IR_symp = [1.76, 6.0, 2.83]` vs target `[1.91, 5.37, 2.35]`, log-SSE
0.05** — a ~10× improvement over exp 04/05 (best ~0.4–0.67). It is a genuinely
*sharp* peak: `<6m`/peak = 0.29 (target 0.36), peak 2.1× its neighbours, correct
magnitude — **and it persists at a smooth endemic** (ever-infected 0.77, no
bimodality). Winning params are exactly the design hypothesis:

| param | value | role |
|---|---|---|
| `young_reservoir` | **3** (low) | off the knife-edge |
| `adult_contacts` | **1.8** | persistence (your lever) |
| titer `median` / `half_life` | 41 / 68d | maternal drop ~6mo |
| titer `gsd` / `hill_slope` | 2.6 / 6.2 | sharp protection edge |
| `base_beta` | 0.22 | FOI level |

The sharp peak lives in a **narrow but real region** (~0.5% of draws hit
peak>1.8×-neighbours at the right magnitude) — low `young_reservoir` + adult
persistence + late-ish sharp maternal drop. Exp 05 had *no* such region; the
titer maternal + adult reservoir created it.

## Figure

![Peak achieved](figures/peak_achieved.png)

## The remaining gap — reinfection (the next target, by design)

Per the agreed sequence ("get incidence right first, then reinfection"), incidence
is now solved but the **repeat-infection fraction in the best-IR draws is ~0.33–0.61
vs data ~0.2** — still ~2–2.5× high. Incidence and reinfection are now *separated*:
we have a region that nails the IR shape; the next experiment tunes the acquired-
immunity ladder (`sus_after_*`) — and/or reconsiders detection of repeat infections —
to pull repeats down to ~0.2 *while holding the IR shape*.

## Mechanistic takeaways

- **The "structural age issue" is resolved at the incidence level** by three
  separable, mechanistically-grounded pieces: titer maternal owns the `<6m` trough +
  peak onset; `young_reservoir`/FOI sets the level; the acquired ladder shapes the
  decline; and adult-adult mixing sustains persistence without touching the targets.
- It is **not** a deep immunity-durability failure (the exp-05 "missing sink" was a
  reservoir artifact) — leaky floored acquired immunity + a literature-grounded titer
  maternal model suffices for the incidence shape.
- The titer maternal model generalizes by construction: drop-age tracks FOI via
  infection timing, so the same mechanism should slide the peak later at low-FOI
  sites (UK) — the cross-FOI validation banked for later.

## Next (exp 07)

Fix the **reinfection** gap: take the winning IR region and sweep/strengthen the
acquired-immunity ladder (and/or revisit repeat-infection detection) to bring the
repeat fraction to ~0.2 while preserving the IR-by-age fit. Then the UK cross-FOI
validation (does the same mechanism reproduce UK's 2-5yr peak?).
