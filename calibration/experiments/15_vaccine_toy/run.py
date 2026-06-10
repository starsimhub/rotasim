"""
Exp 15 — analysis/plot of the toy vaccine runs (reads outputs/toy_runs.csv produced by
vaccine_toy.py). VE = 1 - mean symptomatic-IR(vaccinated) / mean(unvaccinated), per
(model, beta, response). Plots VE vs response for the two models at the fitted (LMIC) beta.

  python run.py
"""
import sys, json, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
df = pd.read_csv(HERE / 'outputs' / 'toy_runs.csv')

rows = []
for model in df.model.unique():
    for bm in sorted(df.beta_mult.unique()):
        sub = df[(df.model == model) & (df.beta_mult == bm)]
        nov = sub[sub.kind == 'novax'].overall_ir.mean()
        for resp in sorted(sub[sub.kind == 'vax'].response.dropna().unique()):
            vax = sub[(sub.kind == 'vax') & (sub.response == resp)].overall_ir.mean()
            ve = 1 - vax / nov if nov > 0 else float('nan')
            rows.append(dict(model=model, beta_mult=bm, base_beta=sub.base_beta.iloc[0],
                             response=resp, novax_ir=nov, vax_ir=vax, VE=ve))
ve = pd.DataFrame(rows)
ve.to_csv(HERE / 'outputs' / 've_table.csv', index=False)
print(ve.to_string(index=False))

# Plot: VE vs response, both models, at the fitted (LMIC) beta (beta_mult=1.0).
fit = ve[ve.beta_mult == 1.0]
fig, ax = plt.subplots(figsize=(8, 5.5))
colors = {'age': '#1b7837', 'infnum': '#2166ac'}
labels = {'age': 'age-symptom (exp 10)', 'infnum': 'infection-number (exp 11)'}
for model in fit.model.unique():
    s = fit[fit.model == model].sort_values('response')
    ax.plot(s.response, s.VE, 'o-', lw=2.5, ms=9, color=colors.get(model, 'k'), label=labels.get(model, model))
ax.set_xlabel('Vaccine response (seroconversion) probability')
ax.set_ylabel('Achieved VE (1 - vax/novax symptomatic IR)')
ax.set_ylim(0, 1)
ax.set_title('Exp 15 (toy) — same vaccine, same data fit, different predicted VE\n'
             'infection-number vs age-symptom model (fitted/LMIC beta; half-beta faded out)')
ax.legend(frameon=False); ax.spines[['top', 'right']].set_visible(False)
fig.tight_layout()
(HERE / 'figures').mkdir(parents=True, exist_ok=True)
fig.savefig(HERE / 'figures' / 've_divergence.png', dpi=150)
print(f"\nwrote {HERE/'figures'/'ve_divergence.png'} + outputs/ve_table.csv")
