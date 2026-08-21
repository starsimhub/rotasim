"""Exp 67 analysis: compares the VE-constrained ridge (this experiment's
6-seed pools, outputs/seed_runs_{age_binned,infnum}.jsonl) against exp64/65's
unconstrained ridge (their own outputs/seed_runs.jsonl), for both models.
No new simulation -- pure post-hoc comparison. See SUMMARY.md.
"""
import json, pathlib
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
EXP64 = HERE.parents[0] / '64_bangladesh_age_binned_ode'
EXP65 = HERE.parents[0] / '65_bangladesh_infnum_ode'
FIG_DIR = HERE / 'figures'
FIG_DIR.mkdir(exist_ok=True)


def load(path):
    return [json.loads(l) for l in open(path)]


unconstrained = {
    'age_binned': load(EXP64 / 'outputs' / 'seed_runs.jsonl'),
    'infnum': load(EXP65 / 'outputs' / 'seed_runs.jsonl'),
}
constrained = {
    'age_binned': load(HERE / 'outputs' / 'seed_runs_age_binned.jsonl'),
    'infnum': load(HERE / 'outputs' / 'seed_runs_infnum.jsonl'),
}


def span(rows, key):
    vals = [r['best_params'][key] for r in rows] if 'best_params' in rows[0] else [r['best_params'][key] for r in rows]
    return min(vals), max(vals)


print(f"{'model':<12}{'param':<10}{'unconstrained':<22}{'constrained':<22}{'span change'}")
for model in ['age_binned', 'infnum']:
    for key in ['sus_r2', 'sus_r3']:
        u_vals = [r['best_params'][key] for r in unconstrained[model]]
        c_vals = [r['best_params'][key] for r in constrained[model]]
        u_lo, u_hi = min(u_vals), max(u_vals)
        c_lo, c_hi = min(c_vals), max(c_vals)
        u_span, c_span = u_hi - u_lo, c_hi - c_lo
        pct = 100 * (c_span - u_span) / u_span
        print(f"{model:<12}{key:<10}{u_lo:.3f}-{u_hi:.3f} (span {u_span:.3f})   "
              f"{c_lo:.3f}-{c_hi:.3f} (span {c_span:.3f})   {pct:+.0f}%")

fig, axes = plt.subplots(1, 2, figsize=(11, 5))
for ax, model in zip(axes, ['age_binned', 'infnum']):
    u = unconstrained[model]; c = constrained[model]
    ax.scatter([r['best_params']['sus_r2'] for r in u], [r['best_params']['sus_r3'] for r in u],
               c='gray', marker='o', s=60, label='unconstrained (exp64/65)', zorder=2)
    ax.scatter([r['best_params']['sus_r2'] for r in c], [r['best_params']['sus_r3'] for r in c],
               c='crimson', marker='x', s=70, label='VE-constrained (exp67)', zorder=3)
    ax.set_xlabel('sus_r2'); ax.set_ylabel('sus_r3'); ax.set_title(model)
    ax.set_xlim(0, 1.05); ax.set_ylim(0, 0.85)
axes[0].legend(fontsize=8)
plt.suptitle("Exp 67: Bangladesh sus_r2/sus_r3 ridge, unconstrained vs PROVIDE-VE-constrained")
plt.tight_layout()
plt.savefig(FIG_DIR / 'ridge_narrowing_both_models.png', dpi=140)
print("\nSaved figures/ridge_narrowing_both_models.png")

# VE landing + nh_logL cost table
print(f"\n{'model':<12}{'ve_model range':<20}{'nh_logL best (constrained)':<28}{'nh_logL best (unconstrained)'}")
for model in ['age_binned', 'infnum']:
    c = constrained[model]; u = unconstrained[model]
    ve_vals = [r['ve_model'] for r in c]
    nh_best_c = max(r['nh_logL'] for r in c)
    nh_best_u = max(r['best_logL'] for r in u)
    print(f"{model:<12}{min(ve_vals):.3f}-{max(ve_vals):.3f}{'':<10}{nh_best_c:.2f}{'':<20}{nh_best_u:.2f}")
