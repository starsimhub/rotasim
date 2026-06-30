# Exp 34 — VE vs FOI gradient: UK-fitted infnum parameters

**Question.** Exp 20 showed that the Bangladesh-fitted infnum model produces a steep
VE-FOI gradient under infection-blocking MOA (VE 0.096 at the highest-FOI end rising
to ~1.0 at low FOI) and a nearly flat gradient under symptom-blocking MOA (0.826–0.932).
That result used Bangladesh posterior draws (exp 27). Does the same gradient appear when
the UK-fitted infnum posterior (exp 28) is used instead? The UK calibration settled on
lower FOI and older age-of-infection than Bangladesh — does this shift the baseline VE
upward, and does the gradient slope change?

The practical question: if VE is measured in a high-income setting (UK, older first
infection), will the achieved VE at that FOI look different from what we'd predict using
Bangladesh parameters? This is the cross-setting translation problem at the core of the
VE-gap hypothesis.

**Plan.** Run the same beta-factor sweep as exp 20 (factors 1.5, 1.25, 1.0, 0.85, 0.7)
using 60 draws from the UK infnum posterior (exp 28, `posterior_hmreweight.csv`).
Infection-blocking and symptom-blocking MOA in separate runs. n=40k agents, paired seeds.
Output: `outputs/uk_infnum_foi_sweep_ib.csv` and `outputs/uk_infnum_foi_sweep_sb.csv`.
Plot the same two-panel VE-vs-age-of-infection figure as exp 20, overlaying UK draws on
top of the Bangladesh draws from exp 20 for direct visual comparison.

**Success criteria.** A good result clearly shifts the UK baseline VE relative to
Bangladesh (IB panel). If the gradient slope is similar but the baseline is higher, that
directly supports the VE-gap hypothesis via the FOI channel. A flat/null result (UK ≈
Bangladesh at the same FOI) would implicate the symptom model rather than FOI as the gap
driver.
