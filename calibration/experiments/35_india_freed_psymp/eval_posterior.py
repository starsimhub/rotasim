"""Overdispersion reweighting + posterior predictive plot for exp 35 (freed p_symp).

Compares exp 35 (infnum, freed p_symp_1, corrected cumul target) against
exp 32 (infnum, fixed p_symp). Key change vs exp 32 eval_posterior.py:
  - K_INF = 93 (biweekly-equivalent corrected count: 37 symp + 2×28 asymp)
    rather than raw detected K_INF = 65
  - EXPS updated to show exp 35 vs exp 32

Run:
  python eval_posterior.py
  python eval_posterior.py --phi 3 --rho 10
"""
import sys, json, argparse, pathlib
import numpy as np, pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE  = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB))
import process_incidence_maled as P

SITE = 'india'
BINS = ['<6 m', '6-11 m', '12-23 m']   # 24-35m has 0 cases — excluded from scoring
EPS  = 1e-9

EXPS = {
    'infnum\n(freed psymp)':  HERE,
    'infnum\n(fixed psymp)':  HERE.parent / '32_india_fixed_psymp',
}

# ── Load India targets ────────────────────────────────────────────────────────
_t     = P.load_targets(SITE)
_ir    = _t['ir_by_age']
_rf    = _t['repeat_frac']
IR_DATA = {b: (int(_ir.loc[b, 'cases']), float(_ir.loc[b, 'PT'])) for b in BINS}
RN = int(_rf['n']); RO = int(round(_rf['frac'] * RN))

# Corrected cumulative first-infection fraction at 36 months.
# Raw MAL-ED Vellore: 65/243 detected (stool + symptomatic, monthly collection).
# Correction: MAL-ED uses monthly stool; Lewnard biweekly cohort collects twice as often.
# Frequency-normalised (biweekly-equivalent): 37 symptomatic + 2×28 asymptomatic = 93/243.
# This is the target the model (which uses biweekly-equivalent p_asymp_detect) should match.
N_STOOL = 243
K_INF   = 93    # biweekly-equivalent corrected (vs raw detected 65)
K_INF_RAW = 65  # for reference in plot


def _read_jsonl(path):
    rows = []
    for line in open(path, errors='ignore'):
        line = line.replace('\x00', '').strip()
        if line:
            try: rows.append(json.loads(line))
            except Exception: pass
    return rows


def _pois(r, phi):
    ll = 0.0
    for b, (c, PT) in IR_DATA.items():
        lam = r['ir_' + b] / 100.0 * PT
        if lam <= 0: return np.nan
        ll += c * np.log(lam) - lam
    return ll / phi


def _binom(r):
    p = min(max(r['repeat_frac'], 1e-6), 1 - 1e-6)
    return RO * np.log(p) + (RN - RO) * np.log(1 - p)


def _cumul_inf(r):
    """Binomial LL on biweekly-equivalent corrected cumulative first-infection fraction."""
    S = np.array(r['km_surv'], float)
    p = float(np.clip(1.0 - S[36], EPS, 1.0 - EPS))
    return K_INF * np.log(p) + (N_STOOL - K_INF) * np.log(1.0 - p)


def _cumul_frac_from_S(r):
    S = np.array(r['km_surv'], float)
    return float(1.0 - S[36])


def reweight(exp_dir, phi, rho, label):
    recs = _read_jsonl(exp_dir / 'outputs' / 'sir_results.jsonl')
    nroy = pd.read_csv(exp_dir / 'outputs' / 'nroy_draw.csv').reset_index(drop=True)

    scored = []
    for r in recs:
        pp = _pois(r, phi)
        if not np.isfinite(pp):
            ll = -np.inf
        else:
            ll = pp + _binom(r) + _cumul_inf(r)
        scored.append((r['idx'], ll, r))

    logL = np.array([x[1] for x in scored], float)
    fin  = np.isfinite(logL)
    w    = np.zeros(len(logL))
    if fin.any():
        w[fin] = np.exp(logL[fin] - np.nanmax(logL[fin])); w /= w.sum()
    ess = float(1.0 / np.sum(w**2)) if fin.any() else 0.0
    print(f"  {label}: finite={fin.sum()}/{len(logL)}  ESS={ess:.1f}  max_logL={np.nanmax(logL[fin]) if fin.any() else np.nan:.1f}")

    wpp = {}
    for b in BINS:
        wpp[f'ir_{b}'] = float(np.sum(w * np.array([r['ir_' + b] for _, _, r in scored])))
    wpp['repeat_frac']  = float(np.sum(w * np.array([r['repeat_frac'] for _, _, r in scored])))
    wpp['cumul_inf_36'] = float(np.sum(w * np.array([_cumul_frac_from_S(r) for _, _, r in scored])))

    dist = {b: np.array([r['ir_' + b] for _, _, r in scored]) for b in BINS}
    dist['repeat_frac']  = np.array([r['repeat_frac'] for _, _, r in scored])
    dist['cumul_inf_36'] = np.array([_cumul_frac_from_S(r) for _, _, r in scored])
    dist['w'] = w

    # report p_symp_1 if available (freed in exp 35)
    if 'p_symp_1' in nroy.columns:
        psymp_w = float(np.sum(w * nroy.loc[[r[0] for r in scored], 'p_symp_1'].values))
        print(f"    weighted p_symp_1 = {psymp_w:.3f}")

    return wpp, dist, ess


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi', type=float, default=3.0)
    ap.add_argument('--rho', type=float, default=0.10)
    a = ap.parse_args()

    print(f"\nIndia overdispersion reweighting: phi={a.phi}, rho={a.rho}")
    print(f"Targets: IR {[round(IR_DATA[b][0]/IR_DATA[b][1]*100,2) for b in BINS]} per 100 pm | "
          f"repeat={round(RO/RN,3)} ({RO}/{RN}) | "
          f"cumul_inf_36 (corrected)={K_INF}/{N_STOOL}={K_INF/N_STOOL:.3f}  "
          f"(raw detected={K_INF_RAW}/{N_STOOL}={K_INF_RAW/N_STOOL:.3f})")

    results = {}
    for label, exp_dir in EXPS.items():
        if not (exp_dir / 'outputs' / 'sir_results.jsonl').exists():
            print(f"  {label}: outputs not yet available — skipping")
            continue
        results[label] = reweight(exp_dir, a.phi, a.rho, label)

    if not results:
        print("No results available yet.")
        return

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    bin_labels = ['<6 mo', '6–11 mo', '12–23 mo']
    ir_targets = [IR_DATA[b][0] / IR_DATA[b][1] * 100 for b in BINS]
    colors = {'infnum\n(freed psymp)': '#1a5276', 'infnum\n(fixed psymp)': '#c0392b'}

    ax = axes[0]
    x = np.arange(len(BINS)); width = 0.30
    ax.bar(x, ir_targets, width=0.5, color='#aaaaaa', alpha=0.5, label='Target (Vellore)', zorder=2)
    for i, (label, (wpp, dist, ess)) in enumerate(results.items()):
        offset = (i - 0.5) * width
        ir_wpp = [wpp[f'ir_{b}'] for b in BINS]
        ir_lo  = [float(np.quantile(dist[b], 0.10)) for b in BINS]
        ir_hi  = [float(np.quantile(dist[b], 0.90)) for b in BINS]
        short  = label.split('\n')[0]
        ax.bar(x + offset, ir_wpp, width=width, color=colors[label], alpha=0.75,
               label=f'{short} (ESS={ess:.1f})', zorder=3)
        yerr_lo = np.maximum(np.array(ir_wpp) - np.array(ir_lo), 0.0)
        yerr_hi = np.maximum(np.array(ir_hi) - np.array(ir_wpp), 0.0)
        ax.errorbar(x + offset, ir_wpp,
                    yerr=[yerr_lo, yerr_hi],
                    fmt='none', color=colors[label], capsize=3, lw=1.5, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Symptomatic IR (per 100 child-months)')
    ax.set_title('A. Incidence by age\n(India/Vellore targets)')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[1]
    target_corr = K_INF / N_STOOL
    target_raw  = K_INF_RAW / N_STOOL
    ax.axhline(target_corr, color='#1a5276', lw=2, ls='--',
               label=f'Target corrected {target_corr:.3f} ({K_INF}/{N_STOOL})')
    ax.axhline(target_raw,  color='#aaaaaa', lw=1.5, ls=':',
               label=f'Target raw {target_raw:.3f} ({K_INF_RAW}/{N_STOOL})')
    for i, (label, (wpp, dist, ess)) in enumerate(results.items()):
        col = colors[label]; short = label.split('\n')[0]
        cf  = wpp['cumul_inf_36']
        lo  = float(np.quantile(dist['cumul_inf_36'], 0.10))
        hi  = float(np.quantile(dist['cumul_inf_36'], 0.90))
        ax.bar(i, cf, color=col, alpha=0.75, width=0.4,
               label=f'{short}: {cf:.3f} (ESS={ess:.1f})')
        ax.errorbar(i, cf, yerr=[[cf-lo], [hi-cf]],
                    fmt='none', color=col, capsize=5, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([l.split('\n')[0] for l in results.keys()], fontsize=9)
    ax.set_ylabel('Cumulative first-infection fraction by 36 mo')
    ax.set_title(f'B. Cumulative infection fraction\n(corrected target: {K_INF}/{N_STOOL}={target_corr:.3f})')
    ax.legend(frameon=False, fontsize=8)

    ax = axes[2]
    target_rep = RO / RN
    ax.axhline(target_rep, color='#aaaaaa', lw=2, ls='--', label=f'Target {target_rep:.3f} ({RO}/{RN})')
    for i, (label, (wpp, dist, ess)) in enumerate(results.items()):
        col = colors[label]; short = label.split('\n')[0]
        rep = wpp['repeat_frac']
        lo  = float(np.quantile(dist['repeat_frac'], 0.10))
        hi  = float(np.quantile(dist['repeat_frac'], 0.90))
        ax.bar(i, rep, color=col, alpha=0.75, width=0.4, label=f'{short}: {rep:.3f}')
        ax.errorbar(i, rep, yerr=[[rep-lo], [hi-rep]],
                    fmt='none', color=col, capsize=5, lw=2)
    ax.set_xticks([0, 1])
    ax.set_xticklabels([l.split('\n')[0] for l in results.keys()], fontsize=9)
    ax.set_ylabel('Repeat-detected fraction')
    ax.set_title('C. Repeat detection fraction')
    ax.legend(frameon=False, fontsize=8)

    fig.suptitle(
        f'India / Vellore — posterior predictive check (phi={a.phi}, rho={a.rho})\n'
        f'exp35: freed p_symp, corrected cumul target ({K_INF}/243)   vs   exp32: fixed p_symp',
        fontsize=10)
    fig.tight_layout()
    tag = f'_phi{int(a.phi)}_rho{int(a.rho*100)}'
    out = HERE / 'figures' / f'india_posterior_predictive{tag}.png'
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
