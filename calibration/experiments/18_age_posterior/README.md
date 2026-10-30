# Exp 18 — Trajectory-selection posterior for the AGE-symptom + titer model

**Question.** Turn the exp-16 age+titer **NROY** (a feasibility region) into a **posterior**
via trajectory selection (importance resampling), so the downstream VE comparison is over a
*distribution*, not a point. age's NROY is loose/weakly-identified (exp 16: plateaued at 0.257,
emulator-limited) — the goal here is to carry that uncertainty forward honestly, producing a
wide-but-correct posterior, not to force it narrow. Matched partner:
[`../19_infnum_posterior/`](../19_infnum_posterior/). Consumed by the VE comparison (exp 20).

**Plan.** Shared driver `../../trajectory_select.py --model age` (mirrors `hm_calibrate.py`):
1. **Draw** N≈5000 NROY samples from the exp-16 HM checkpoint (`hm.HistoryMatching.load_checkpoint`
   → `get_nroy_samples`), **cached** to `outputs/nroy_draw.csv` so re-runs use the identical set.
2. **Simulate** each at 40k agents, cohort observation, **one fixed seed per draw** (`seed = BASE + idx`,
   deterministic → reproducible; per-index so the matched pair and any resume are identical).
3. **Score** with the composite log-likelihood (Dan's `trajectory_selection.py` form, agreed
   2026-06-11):
   - IR by age bin (<6, 6–11, 12–23): **Poisson** log-L, λ = IR/100·PT
   - repeat-detected fraction: **Binomial** log-L
   - age-at-first-infection: **full censored-survival** log-L of every infant's (age, event)
     record under the model's monthly KM first-detection survival
   - **No ever-detected channel** (it's a marginal of the survival data — would double-count).
   - extinct / failed sims → log-L = −∞ (dropped).
4. **Importance-resample** by weight ∝ exp(logL − max) → `outputs/posterior.csv`; report ESS.
   Stream per-sim results to JSONL (resumable). Pinned env (`../../hm_env_pins.txt`); on a 120-core VM.

**Success criteria.** A posterior over the age box params + a posterior-predictive that covers
the five MAL-ED targets, with a usable ESS. A *wide* posterior is the expected, correct outcome
(age is weakly identified) — the test is coverage of the targets and a non-degenerate ESS, not
tightness. A collapsed ESS (≈1) would mean the likelihood is far tighter than the NROY (revisit
N or the likelihood scale). Feeds the exp-20 VE-distribution comparison.
