"""Burden gap closure across the FOI gradient.

For EDGE stakeholders: shows how much of the LMIC-HIC absolute burden gap
can be closed by vaccination at different take levels, and where each setting
sits on the burden map.

Data:
  g11 resp=0.75 : 11-point factor grid (factors 0.55–1.60), n=373 draws
  5pt resp=0.63/0.75/0.90 : factors [0.70, 0.85, 1.00, 1.25, 1.50], n=373 draws

Panels:
  A. Absolute remaining burden (symptomatic IR per 100 child-months, ages 0-36mo)
     vs median age of first infection — the "burden map" showing how much
     disease remains at each setting for no-vaccine and three take levels.
  B. Burden gap closure — at LMIC anchor (factor=1.0, ~8mo), how much burden
     remains vs take, with reference lines for a lower-FOI setting (~factor=0.90)
     and approximate HIC (factor=0.65, sparse n=5 noted).

Run:
  python burden_gap_closure.py
  python burden_gap_closure.py --model infnum
"""
import argparse, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

HERE = pathlib.Path(__file__).resolve().parent
OUT  = HERE / 'outputs'
FIG  = HERE / 'figures'; FIG.mkdir(exist_ok=True)

# Reference operating points (median age of first infection, months)
ANCHORS = [
    (7.8,  'Bangladesh\n~8mo',  '#c0392b'),
    (11.5, 'India\n~12mo',      '#7d3c98'),
    (15.0, 'UK\n~15mo',         '#1a6b3a'),
]

RESP_CONFIGS = [
    (0.63, '#f0a500', '-',  'o', '5pt'),
    (0.75, '#c0392b', '--', 's', '5pt'),
    (0.90, '#6c0017', ':',  '^', '5pt'),
]


def load_data(model='infnum'):
    g11 = pd.read_csv(OUT / f'{model}_foi_sweep_alluniq_g11.csv')
    d75 = pd.read_csv(OUT / f'{model}_foi_sweep_alluniq.csv')
    d63 = pd.read_csv(OUT / f'{model}_foi_sweep_alluniq_r063.csv')
    d90 = pd.read_csv(OUT / f'{model}_foi_sweep_alluniq_r090.csv')
    return g11, {0.63: d63, 0.75: d75, 0.90: d90}


def summarise_burden(df, filter_novax=0.1):
    """Per-factor: median novax_ir, vax_ir, age_of_inf, n alive."""
    rows = []
    for fac, sub in df.groupby('factor'):
        alive = sub[sub.novax_ir > filter_novax]
        if len(alive) < 3:
            continue
        rows.append(dict(
            factor=fac,
            n=len(alive),
            age=alive.age_of_inf.median(),
            novax_ir=alive.novax_ir.median(),
            vax_ir=alive.vax_ir.median(),
            novax_ir_lo=alive.novax_ir.quantile(0.25),
            novax_ir_hi=alive.novax_ir.quantile(0.75),
            vax_ir_lo=alive.vax_ir.quantile(0.25),
            vax_ir_hi=alive.vax_ir.quantile(0.75),
        ))
    return pd.DataFrame(rows).sort_values('age')


def gap_closure_at_lmic(resp_dfs, g11, lmic_factor=1.0, hic_factor=0.65):
    """
    For each response rate: compute how much of the REMAINING burden gap
    (after both settings are vaccinated at current take=0.75) is closed
    by increasing LMIC take.

    Reference gap = LMIC_vax(0.75) − HIC_vax(0.75)   [current vaccinated world]
    % closed      = (ref_gap − new_vax_gap) / ref_gap × 100

    HIC reference: factor=0.65 from g11 (n=5 alive — sparse, flagged in figure).
    If fewer than 3 draws alive, return None for HIC gap calc.
    """
    # LMIC novax IR (same seeds across resp CSVs)
    lmic_novax_draws = resp_dfs[0.75][(resp_dfs[0.75].factor == lmic_factor) &
                                       (resp_dfs[0.75].novax_ir > 0.1)]
    lmic_novax_ir = lmic_novax_draws.novax_ir.median()

    # HIC vax IR at resp=0.75 from g11 (sparse)
    hic_draws     = g11[(g11.factor.round(2) == hic_factor) & (g11.novax_ir > 0.1)]
    hic_novax_ir  = hic_draws.novax_ir.median() if len(hic_draws) >= 3 else None
    hic_vax_ir    = hic_draws.vax_ir.median()   if len(hic_draws) >= 3 else None
    hic_n         = len(hic_draws)

    # Mid-income reference: factor=0.90 from g11 (n=232, robust)
    mid_draws    = g11[(g11.factor == 0.90) & (g11.novax_ir > 0.1)]
    mid_novax_ir = mid_draws.novax_ir.median()
    mid_vax_ir   = mid_draws.vax_ir.median()

    # Current vaccinated gap (both at take=0.75) — the denominator
    lmic_vax_ref = resp_dfs[0.75][(resp_dfs[0.75].factor == lmic_factor) &
                                   (resp_dfs[0.75].novax_ir > 0.1)].vax_ir.median()
    current_vax_gap = (lmic_vax_ref - hic_vax_ir) if hic_vax_ir is not None else None

    rows = []
    for resp, df in sorted(resp_dfs.items()):
        lmic_alive  = df[(df.factor == lmic_factor) & (df.novax_ir > 0.1)]
        lmic_vax_ir = lmic_alive.vax_ir.median()

        row = dict(
            resp=resp,
            lmic_novax_ir=lmic_novax_ir,
            lmic_vax_ir=lmic_vax_ir,
            lmic_ve=lmic_alive.ve_overall.median(),
            n_lmic=len(lmic_alive),
            mid_novax_ir=mid_novax_ir,
            mid_vax_ir=mid_vax_ir,
            hic_novax_ir=hic_novax_ir,
            hic_vax_ir=hic_vax_ir,
            hic_n=hic_n,
            current_vax_gap=current_vax_gap,
        )
        if current_vax_gap is not None and current_vax_gap > 0:
            new_vax_gap = lmic_vax_ir - hic_vax_ir
            row['new_vax_gap'] = new_vax_gap
            row['pct_remaining_gap_closed'] = (
                (current_vax_gap - new_vax_gap) / current_vax_gap * 100
            )
        rows.append(row)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='infnum', choices=['infnum', 'age_binned'])
    a = ap.parse_args()

    g11, resp_dfs = load_data(a.model)

    # ── Panel A: full burden map across FOI ──────────────────────────────────
    novax_curve = summarise_burden(g11)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]

    # Novax baseline
    ax.fill_between(novax_curve.age, 0, novax_curve.novax_ir,
                    color='#bbbbbb', alpha=0.25, label='No vaccine (novax)')
    ax.plot(novax_curve.age, novax_curve.novax_ir,
            color='#555555', lw=2.2, zorder=4)

    # Vax curves from the three resp CSVs (5-point factors)
    resp_labels = {0.63: 'Take 63%', 0.75: 'Take 75%', 0.90: 'Take 90%'}
    resp_colors = {0.63: '#f0a500', 0.75: '#c0392b', 0.90: '#6c0017'}
    for resp, df in sorted(resp_dfs.items()):
        curve = summarise_burden(df)
        col = resp_colors[resp]
        ax.plot(curve.age, curve.vax_ir, '-', color=col, lw=2.0,
                marker='o', ms=5, alpha=0.85, label=f'Vaccinated — take={resp:.0%}')
        ax.fill_between(curve.age, curve.vax_ir_lo, curve.vax_ir_hi,
                        color=col, alpha=0.10)

    # Annotate setting anchors
    for age_ref, label, col in ANCHORS:
        ax.axvline(age_ref, color=col, ls=':', lw=1.2, alpha=0.7)
        ax.text(age_ref + 0.2, ax.get_ylim()[1] * 0.97 if ax.get_ylim()[1] > 0 else 4.5,
                label, fontsize=7.5, color=col, va='top')

    ax.set_xlabel('Simulated median age of first infection (months)', fontsize=10)
    ax.set_ylabel('Symptomatic incidence (per 100 child-months, ages 0–36mo)', fontsize=10)
    ax.set_xlim(2, 20)
    ax.set_ylim(bottom=0)
    ax.legend(frameon=False, fontsize=8.5, loc='upper right')
    ax.set_title('A. Burden map across the FOI gradient\n'
                 '(infnum model, IB vaccine, resp=0.63/0.75/0.90)',
                 fontsize=10)

    # ── Panel B: burden gap closure by take ─────────────────────────────────
    ax2 = axes[1]
    gap_df = gap_closure_at_lmic(resp_dfs, g11)

    takes   = gap_df.resp.values
    lmic_vx = gap_df.lmic_vax_ir.values
    lmic_nv = gap_df.lmic_novax_ir.values[0]

    # Bars: LMIC remaining burden at each take
    bar_cols = [resp_colors[r] for r in takes]
    bars = ax2.bar(np.arange(len(takes)), lmic_vx, color=bar_cols, width=0.5,
                   alpha=0.8, zorder=3, label='LMIC remaining (vax)')

    # Horizontal reference lines
    ax2.axhline(lmic_nv, color='#555555', ls='-', lw=1.5, label=f'LMIC no vaccine ({lmic_nv:.2f})')

    mid_nv = gap_df.mid_novax_ir.values[0]
    mid_vx = gap_df.mid_vax_ir.values[0]
    ax2.axhline(mid_nv, color='#7d3c98', ls='--', lw=1.2,
                label=f'Moderate-FOI no vax (~10mo, {mid_nv:.2f})')
    ax2.axhline(mid_vx, color='#7d3c98', ls=':', lw=1.2,
                label=f'Moderate-FOI vaccinated (take=75%, {mid_vx:.2f})')

    # HIC reference (sparse — flag with asterisk)
    hic_nv = gap_df.hic_novax_ir.values[0]
    hic_vx = gap_df.hic_vax_ir.values[0]
    hic_n  = int(gap_df.hic_n.values[0])
    if pd.notna(hic_nv):
        ax2.axhline(hic_nv, color='#1a6b3a', ls='--', lw=1.2,
                    label=f'HIC no vaccine (~16mo, {hic_nv:.2f}, n={hic_n}*)')
        ax2.axhline(hic_vx, color='#1a6b3a', ls=':', lw=1.2,
                    label=f'HIC vaccinated (take=75%, {hic_vx:.2f}, n={hic_n}*)')

    # Annotate bars with: remaining IR and VE
    for i, (resp, lv, ve) in enumerate(zip(takes, lmic_vx, gap_df.lmic_ve.values)):
        ax2.text(i, lv + 0.05, f'IR={lv:.2f}\nVE={ve:.0%}',
                 ha='center', va='bottom', fontsize=8, color='#222222')

    # Annotate bars with % of current vaccinated gap closed
    if 'pct_remaining_gap_closed' in gap_df.columns:
        for i, (resp, pct) in enumerate(zip(takes, gap_df.pct_remaining_gap_closed.values)):
            if pd.notna(pct) and pct > 0:
                ax2.text(i, lmic_vx[i] / 2, f'+{pct:.0f}%\nof vax\ngap closed',
                         ha='center', va='center', fontsize=7.5,
                         color='white', fontweight='bold')

    # 60% gap closure target line (60% of current vaccinated gap)
    cur_gap = gap_df.current_vax_gap.values[0]
    if pd.notna(hic_vx) and pd.notna(cur_gap):
        target_ir = hic_vx + 0.40 * cur_gap   # remaining 40% of gap
        ax2.axhline(target_ir, color='#e74c3c', ls='-.', lw=1.8,
                    label=f'60% of current vax gap closed (IR={target_ir:.2f})')

    ax2.set_xticks(np.arange(len(takes)))
    ax2.set_xticklabels([f'Take\n{r:.0%}' for r in takes], fontsize=9)
    ax2.set_ylabel('Symptomatic incidence (per 100 child-months, ages 0–36mo)', fontsize=10)
    ax2.set_ylim(bottom=0)
    ax2.legend(frameon=False, fontsize=7.5, loc='upper right',
               title='Reference lines', title_fontsize=8)
    ax2.set_title('B. Burden gap closure at LMIC operating point\n'
                  'Reference = current vaccinated gap (both at take=75%); HIC factor=0.65, n=5*',
                  fontsize=10)

    fig.suptitle(
        'exp20 — infnum model (infection-blocking) | Burden gap closure across FOI gradient\n'
        'No waning; indirect effects (herd immunity) captured via dynamic transmission',
        fontsize=10)
    fig.tight_layout()
    out_path = FIG / f'burden_gap_closure_{a.model}.png'
    fig.savefig(out_path, dpi=140)
    plt.close(fig)

    # ── Print summary table ──────────────────────────────────────────────────
    cur_gap = gap_df.current_vax_gap.values[0]
    print(f'\n=== Burden gap closure (reference = current vaccinated gap, both take=75%) ===')
    print(f'  LMIC novax IR   : {lmic_nv:.3f} per 100 child-months')
    if pd.notna(hic_nv):
        print(f'  HIC  novax IR   : {hic_nv:.3f}  (n={hic_n} alive draws*)')
        lmic_vax_ref = gap_df[gap_df.resp == 0.75].lmic_vax_ir.values[0]
        print(f'  LMIC vax IR @75%: {lmic_vax_ref:.3f}')
        print(f'  HIC  vax IR @75%: {hic_vx:.3f}  (n={hic_n}*)')
        print(f'  Current vax gap : {cur_gap:.3f}  ← denominator')
        target_ir = hic_vx + 0.40 * cur_gap
        req_ve    = 1 - target_ir / lmic_nv
        print(f'  60%-closure target LMIC IR: {target_ir:.3f}  (requires LMIC VE = {req_ve:.1%})')
        print()
        print(f'  {"Take":>6}  {"LMIC vax IR":>12}  {"VE":>6}  {"New vax gap":>12}  {"% of vax gap closed":>20}')
        for _, r in gap_df.iterrows():
            pct = r.get('pct_remaining_gap_closed', float('nan'))
            ng  = r.get('new_vax_gap', float('nan'))
            print(f'  {r.resp:>6.0%}  {r.lmic_vax_ir:>12.3f}  {r.lmic_ve:>6.1%}  {ng:>12.3f}  {pct:>19.1f}%')
        n_mid = len(g11[(g11.factor == 0.90) & (g11.novax_ir > 0.1)])
        print(f'\n  * HIC uses only {hic_n} surviving draws at factor=0.65 — high-beta outliers only.')
        print(f'    Moderate-FOI ref (factor=0.90, n={n_mid}): novax={mid_nv:.3f}, vax(75%)={mid_vx:.3f}')
    else:
        print(f'  HIC endpoint: too few alive draws (n={hic_n}) for reliable gap estimate')

    print(f'\nwrote {out_path}')


if __name__ == '__main__':
    main()
