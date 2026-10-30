"""Demographic standardization of VE across the FOI gradient.

Takes the by_age output from foi_sweep_byage.py and computes VE
re-weighted by UK vs Bangladesh case-age distributions, isolating the
demographic composition effect on population-level VE.

The key comparison:
  - Bangladesh-weighted VE: age-specific VEs weighted by LMIC case distribution
    (cases cluster in 6-12mo due to high FOI)
  - UK-weighted VE: same per-age VEs but weighted by HIC case distribution
    (cases cluster in 12-24mo due to low FOI)
  - Difference: pure demographic effect, holding per-age efficacy fixed

Reference weights are derived from the simulation's own novax case distribution
at the LMIC (factor=1.0) and HIC (factor=0.65) operating points.

Run:
  python demographic_standardization.py
  python demographic_standardization.py --model infnum --tag _50k
"""
import argparse, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
OUT  = HERE / 'outputs'
FIG  = HERE / 'figures'; FIG.mkdir(exist_ok=True)

BIN_COLS  = ['0_6', '6_12', '12_24', '24_36']
BIN_WIDTH = np.array([6., 6., 12., 12.])
BIN_LABELS = ['0–6 mo', '6–12 mo', '12–24 mo', '24–36 mo']

# FOI reference points for deriving case-weight populations
LMIC_FACTOR = 1.00   # Bangladesh operating point
HIC_FACTOR  = 0.65   # UK-equivalent operating point (~16.5mo)
REF_AGES    = [(8.7, 'LMIC ~8.7mo'), (15.0, 'HIC ~15mo')]


def case_weights(sub):
    """Median case-fraction per bin from novax arm (cases ∝ IR × bin_width)."""
    irn = np.array([sub[f'novax_ir_{c}'].median() for c in BIN_COLS])
    cases = irn * BIN_WIDTH
    total = cases.sum()
    return cases / total if total > 0 else np.ones(4) / 4


def standardised_ve(row, weights):
    """VE reweighted by external case-fraction weights."""
    irn = np.array([row[f'novax_ir_{c}'] for c in BIN_COLS])
    irv = np.array([row[f'vax_ir_{c}']   for c in BIN_COLS])
    wn = (irn * weights).sum()
    wv = (irv * weights).sum()
    return (1 - wv / wn) if wn > 0 else float('nan')


def summarise(df, weights, label):
    """Median standardised VE by factor (alive draws only)."""
    rows = []
    for fac, sub in df.groupby('factor'):
        alive = sub[sub.novax_ir > 0.1]
        if len(alive) == 0:
            continue
        ves = alive.apply(standardised_ve, axis=1, weights=weights)
        rows.append((alive.age_of_inf.median(), ves.median(), len(alive), fac))
    return sorted(rows)


def ap_parse():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='infnum', choices=['infnum', 'age_binned'])
    ap.add_argument('--tag', default='', help='file suffix, e.g. _50k')
    return ap.parse_args()


def main():
    a = ap_parse()
    fpath = OUT / f'{a.model}_foi_sweep_byage{a.tag}.csv'
    if not fpath.exists():
        print(f"Missing: {fpath}\nRun foi_sweep_byage.py first.")
        return

    df = pd.read_csv(fpath)

    # Derive reference case weights from simulation novax arm
    lmic_draws = df[(df.factor == LMIC_FACTOR) & (df.novax_ir > 0.1)]
    hic_draws  = df[(df.factor.round(2) == HIC_FACTOR) & (df.novax_ir > 0.1)]

    if len(lmic_draws) == 0 or len(hic_draws) == 0:
        print(f"Not enough alive draws at reference factors "
              f"(LMIC factor={LMIC_FACTOR}: n={len(lmic_draws)}, "
              f"HIC factor={HIC_FACTOR}: n={len(hic_draws)})")
        return

    w_bd = case_weights(lmic_draws)
    w_uk = case_weights(hic_draws)

    print(f"\n=== Case-age distribution at reference points (median across draws) ===")
    print(f"{'Bin':<12}  {'Width':>6}  {'BD weight':>10}  {'UK weight':>10}  {'UK/BD ratio':>12}")
    for lab, c, w, wb, wu in zip(BIN_LABELS, BIN_COLS, BIN_WIDTH, w_bd, w_uk):
        print(f"{lab:<12}  {w:>6.0f}mo  {wb:>10.1%}  {wu:>10.1%}  {wu/wb:>12.2f}x")

    # Compute three curves
    curves = [
        ('BD-weighted VE',  w_bd, '#e07b00', '-',  'o'),
        ('UK-weighted VE',  w_uk, '#1a6b3a', '--', 's'),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))

    # Left panel: the two standardised VE curves vs age-of-infection
    ax = axes[0]
    print(f"\n{'Weighting':<20}  {'age~8.7mo':>10}  {'VE@8.7':>8}  {'age~15mo':>10}  {'VE@15':>8}")
    for label, weights, col, ls, mk in curves:
        rows = summarise(df, weights, label)
        if not rows:
            continue
        ages = [r[0] for r in rows]; ves = [r[1] for r in rows]
        ax.plot(ages, ves, ls=ls, color=col, lw=2.2, marker=mk, ms=7,
                alpha=0.88, label=label)
        c87 = min(rows, key=lambda r: abs(r[0] - 8.7))
        c15 = min(rows, key=lambda r: abs(r[0] - 15.0))
        ax.annotate(f'{c15[1]:.2f}', xy=(c15[0], c15[1]),
                    xytext=(c15[0] + 0.3, c15[1] + 0.02),
                    fontsize=8, color=col)
        print(f"{label:<20}  {c87[0]:>10.1f}  {c87[1]:>8.3f}  {c15[0]:>10.1f}  {c15[1]:>8.3f}")

    for x, lab in REF_AGES:
        ax.axvline(x, color='gray', ls=':', lw=1.2)
        ax.annotate(lab, (x, 0.03), rotation=90, va='bottom', ha='right',
                    fontsize=8, color='gray')
    ax.set_xlabel('Simulated median age-of-infection (months)')
    ax.set_ylabel('Standardised VE (overall symptomatic, resp=0.75)')
    ax.set_xlim(left=0); ax.set_ylim(-0.05, 1.05)
    ax.legend(frameon=False, fontsize=9, loc='upper left')
    ax.set_title('A. Demographic-standardised VE\n(BD vs UK case-age weights)')

    # Right panel: stacked bar — case-age distribution at LMIC vs HIC operating points
    ax2 = axes[1]
    x = np.arange(2)
    bottom = np.zeros(2)
    colors = ['#d62728', '#ff7f0e', '#2ca02c', '#1f77b4']
    for i, (lab, col) in enumerate(zip(BIN_LABELS, colors)):
        vals = [w_bd[i], w_uk[i]]
        ax2.bar(x, vals, bottom=bottom, color=col, label=lab, width=0.5)
        for xi, v, b in zip(x, vals, bottom):
            if v > 0.04:
                ax2.text(xi, b + v/2, f'{v:.0%}', ha='center', va='center',
                         fontsize=9, color='white', fontweight='bold')
        bottom += np.array(vals)
    ax2.set_xticks(x); ax2.set_xticklabels(['Bangladesh\n(LMIC, ~8.7mo)', 'UK-equiv\n(HIC, ~16.5mo)'])
    ax2.set_ylabel('Fraction of symptomatic cases')
    ax2.set_ylim(0, 1.05); ax2.legend(frameon=False, fontsize=8, loc='upper right')
    ax2.set_title('B. Case-age distribution\nat reference operating points')

    fig.suptitle(
        f'exp20 — {a.model} model, IB resp=0.75, infnum g11 grid\n'
        f'Demographic standardization: same per-age VE, different case-age weights',
        fontsize=10)
    fig.tight_layout()
    out_path = FIG / f've_demographic_standardization_{a.model}{a.tag}.png'
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
