# Exp 24 — infnum+titer with the titer SHAPE fixed (infnum member of the clean same-maternal pair)

**Question.** The infnum partner to exp 23. To get the **clean same-maternal matched pair** for
the VE comparison, fit *both* models with maternal held at the *same* identified curve
(`FIXED_TITER_SHAPE`), differing only in the symptom model. infnum already fits with titer free
(exp 11/19, ESS 59) but with poorly-identified params (titer shape spans its prior); fixing the
shape should give a **tighter, more identifiable infnum posterior** and — paired with exp 23
(age+titer-fixedshape) — a same-maternal pair where the *only* difference is age vs
infection-number symptoms. See [`../23_age_titer_fixedshape/`](../23_age_titer_fixedshape/),
[`../19_infnum_posterior/`](../19_infnum_posterior/), the maternal-model discussion.

**Plan.** `hm_calibrate.py --model infnum --maternal titer --fix-titer-shape --all-targets`
(cycle all 5 targets/wave, 8 waves, 40k agents, cohort obs). Fits `maternal_efficacy` +
transmission + `p_symp` ladder; titer shape fixed at the identified curve. Queued to auto-launch
on covaguest when exp 23 completes.

**Success criteria.** A tighter infnum posterior than exp 19 (titer-free), validating on the 5
targets, forming — with exp 23 — the clean same-maternal pair for exp 20 VE (plus a maternal
sensitivity sweep over plausible fixed curves). If exp 23 (age) fails to identify but exp 24
(infnum) tightens, the fix-maternal strategy still improves infnum; the age member then falls
back to exp 21 (age+Erlang).
