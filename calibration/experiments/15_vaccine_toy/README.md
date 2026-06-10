# Exp 15 — Toy vaccine simulations: do the two models predict different VE?

**Question.** The payoff of the two-model program. Two structurally-different models both
fit the pre-vaccine MAL-ED data: age-symptom + Erlang (exp 10) and infection-number +
titer (exp 11). If we add the SAME hypothetical vaccine to both, do they predict
different vaccine impact? Divergence => VE depends on unidentifiable structure;
agreement => robust.

**Plan.** Vaccine per A. Kraay's VIMC model SI: 2 doses at 2 & 4 months; each dose a
person seroconverts to (prob = response) advances them one infection-equivalent
(`num_recovered_infections += 1`, up to +2). That counter feeds BOTH susceptibility
(`sus_after_k`) AND the infection-number symptom probability (`p_symp`) -- so the vaccine
cuts acquisition + symptoms in the infnum model, but only acquisition in the age model
(symptoms age-driven). `VaccinePrime` + `SympIRObserver` in `vaccine_toy.py`. Forward-run
the two fitted models +/- vaccine, sweep response 0.6/0.75/0.9 and base_beta {fitted=LMIC,
x0.5=HIC proxy}, 50k agents, 3 reps. VE = 1 - symptomatic-IR(vaccinated)/(unvaccinated),
age<=36mo (total effect, whole-population).

**Success criteria.** A clear answer either way. Toy/scoping: point-fit params (no
uncertainty) -- the rigorous version is HM posteriors -> VE distributions. Large
divergence => motivates the HM phase; small => the structural question is weak.
