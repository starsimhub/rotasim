"""Posterior predictive comparison: exp39 (age_binned fixed psymp) vs baselines.

Compares four India models:
  exp39 — age_binned, p_symp per bin fixed at Vellore biweekly values
  exp40 — age_and_infection, extinction penalty + 6 waves
  exp35 — infnum, freed p_symp_1 lower bound + corrected K_INF=93
  exp32 — infnum, fixed p_symp at biweekly values (original baseline)

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
BINS = ['<6 m', '6-11 m', '12-23 m']
EPS  = 1e-9

EXPS = {
    'age_binned\n(fixed psymp)':      HERE / 'outputs' / 'ts',
    'age_and_inf\n(ext pen)':         HERE.parent / '40_india_age_inf_extpen' / 'outputs' / 'ts',
    'infnum\n(freed psymp)':          HERE.parent / '35_india_freed_psymp' / 'outputs',
    'infnum\n(fixed psymp)':          HERE.parent / '32_india_fixed_psymp' / 'outputs',
}

COLORS = {
    'age_binned\n(fixed psymp)':  '#1a5276',
    'age_and_inf\n(ext pen)':     '#117a65',
    'infnum\n(freed psymp)':      '#784212',
    'infnum\n(fixed psymp)':      '#c0392b',
}

# ── Load India targets ────────────────────────────────────────────────────────
_t      = P.load_targets(SITE)
_ir     = _t['ir_by_age']
_rf     = _t['repeat_frac']
IR_DATA = {b: (int(_ir.loc[b, 'cases']), float(_ir.loc[b, 'PT'])) for b in BINS}
RN = int(_rf['n']); RO = int(round(_rf['frac'] * RN))

N_STOOL = 243
K_INF   = 93    # biweekly-equivalent corrected
K_INF_RAW = 65  # raw detected, for reference


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
    S = np.array(r['km_surv'], float)
    p = float(np.clip(1.0 - S[36], EPS, 1.0 - EPS))
    return K_INF * np.log(p) + (N_STOOL - K_INF) * np.log(1.0 - p)


def _cumul_frac_from_S(r):
    S = np.array(r['km_surv'], float)
    return float(1.0 - S[36])


def reweight(exp_dir, phi, label):
    sir_path  = exp_dir / 'sir_results.jsonl'
    nroy_path = exp_dir / 'nroy_draw.csv'
    if not sir_path.exists():
        return None
    recs = _read_jsonl(sir_path)
    nroy = pd.read_csv(nroy_path).reset_index(drop=True) if nroy_path.exists() else None

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
    maxL = float(np.nanmax(logL[fin])) if fin.any() else np.nan
    print(f"  {label.replace(chr(10),' ')}: finite={fin.sum()}/{len(logL)}  ESS={ess:.1f}  max_logL={maxL:.1f}")

    wpp = {}
    for b in BINS:
        wpp[f'ir_{b}'] = float(np.sum(w * np.array([r['ir_' + b] for _, _, r in scored])))
    wpp['repeat_frac']  = float(np.sum(w * np.array([r['repeat_frac'] for _, _, r in scored])))
    wpp['cumul_inf_36'] = float(np.sum(w * np.array([_cumul_frac_from_S(r) for _, _, r in scored])))

    dist = {b: np.array([r['ir_' + b] for _, _, r in scored]) for b in BINS}
    dist['repeat_frac']  = np.array([r['repeat_frac'] for _, _, r in scored])
    dist['cumul_inf_36'] = np.array([_cumul_frac_from_S(r) for _, _, r in scored])
    dist['w'] = w

    return wpp, dist, ess, maxL


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--phi', type=float, default=3.0)
    ap.add_argument('--rho', type=float, default=0.10)
    a = ap.parse_args()

    print(f"\nIndia posterior predictive: phi={a.phi}, rho={a.rho}")
    ir_targets = [IR_DATA[b][0] / IR_DATA[b][1] * 100 for b in BINS]
    print(f"IR targets: {[round(x,3) for x in ir_targets]} per 100 child-months")
    print(f"Repeat: {RO}/{RN}={RO/RN:.3f}  |  cumul first-inf 36mo: {K_INF}/{N_STOOL}={K_INF/N_STOOL:.3f} (corrected)")

    results = {}
    for label, exp_dir in EXPS.items():
        out = reweight(exp_dir, a.phi, label)
        if out is not None:
            results[label] = out

    if not results:
        print("No results available."); return

    n_exp = len(results)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    bin_labels = ['<6 mo', '6–11 mo', '12–23 mo']

    # Panel A — IR by age
    ax = axes[0]
    x = np.arange(len(BINS))
    width = 0.18
    ax.bar(x, ir_targets, width=0.55, color='#aaaaaa', alpha=0.45, label='Target (Vellore)', zorder=2)
    for i, (label, (wpp, dist, ess, maxL)) in enumerate(results.items()):
        offset = (i - (n_exp - 1) / 2.0) * width
        ir_wpp = [wpp[f'ir_{b}'] for b in BINS]
        ir_lo  = [float(np.quantile(dist[b], 0.10)) for b in BINS]
        ir_hi  = [float(np.quantile(dist[b], 0.90)) for b in BINS]
        short  = label.split('\n')[0]
        ax.bar(x + offset, ir_wpp, width=width, color=COLORS[label], alpha=0.80,
               label=f'{short} ESS={ess:.1f}', zorder=3)
        yerr_lo = np.maximum(np.array(ir_wpp) - np.array(ir_lo), 0.0)
        yerr_hi = np.maximum(np.array(ir_hi) - np.array(ir_wpp), 0.0)
        ax.errorbar(x + offset, ir_wpp, yerr=[yerr_lo, yerr_hi],
                    fmt='none', color=COLORS[label], capsize=3, lw=1.5, zorder=4)
    ax.set_xticks(x); ax.set_xticklabels(bin_labels, fontsize=9)
    ax.set_ylabel('Symptomatic IR (per 100 child-months)')
    ax.set_title('A. Incidence by age')
    ax.legend(frameon=False, fontsize=7)

    # Panel B — cumulative first infection by 36 mo
    ax = axes[1]
    target_corr = K_INF / N_STOOL
    target_raw  = K_INF_RAW / N_STOOL
    ax.axhline(target_corr, color='#1a5276', lw=2, ls='--',
               label=f'Target corrected {target_corr:.3f} ({K_INF}/{N_STOOL})')
    ax.axhline(target_raw,  color='#aaaaaa', lw=1.5, ls=':',
               label=f'Target raw {target_raw:.3f} ({K_INF_RAW}/{N_STOOL})')
    bw = 0.35
    for i, (label, (wpp, dist, ess, maxL)) in enumerate(results.items()):
        col = COLORS[label]; short = label.split('\n')[0]
        cf  = wpp['cumul_inf_36']
        lo  = float(np.quantile(dist['cumul_inf_36'], 0.10))
        hi  = float(np.quantile(dist['cumul_inf_36'], 0.90))
        ax.bar(i, cf, color=col, alpha=0.80, width=bw, label=f'{short}: {cf:.3f}')
        ax.errorbar(i, cf, yerr=[[max(cf - lo, 0)], [max(hi - cf, 0)]],
                    fmt='none', color=col, capsize=5, lw=2)
    ax.set_xticks(range(n_exp))
    ax.set_xticklabels([l.split('\n')[0] for l in results], fontsize=7, rotation=15)
    ax.set_ylabel('Cumulative first-infection fraction by 36 mo')
    ax.set_title(f'B. Cumulative first infection\n(corrected target: {K_INF}/{N_STOOL}={target_corr:.3f})')
    ax.legend(frameon=False, fontsize=7)

    # Panel C — repeat detection fraction
    ax = axes[2]
    target_rep = RO / RN
    ax.axhline(target_rep, color='#aaaaaa', lw=2, ls='--', label=f'Target {target_rep:.3f} ({RO}/{RN})')
    for i, (label, (wpp, dist, ess, maxL)) in enumerate(results.items()):
        col = COLORS[label]; short = label.split('\n')[0]
        rep = wpp['repeat_frac']
        lo  = float(np.quantile(dist['repeat_frac'], 0.10))
        hi  = float(np.quantile(dist['repeat_frac'], 0.90))
        ax.bar(i, rep, color=col, alpha=0.80, width=bw, label=f'{short}: {rep:.3f}')
        ax.errorbar(i, rep, yerr=[[max(rep - lo, 0)], [max(hi - rep, 0)]],
                    fmt='none', color=col, capsize=5, lw=2)
    ax.set_xticks(range(n_exp))
    ax.set_xticklabels([l.split('\n')[0] for l in results], fontsize=7, rotation=15)
    ax.set_ylabel('Repeat-detected fraction')
    ax.set_title('C. Repeat detection fraction')
    ax.legend(frameon=False, fontsize=7)

    fig.suptitle(
        f'India / Vellore — posterior predictive (phi={a.phi}, rho={a.rho})\n'
        f'exp39: age_binned fixed psymp  |  exp40: age+inf extpen  |  '
        f'exp35: infnum freed psymp  |  exp32: infnum fixed psymp',
        fontsize=9)
    fig.tight_layout()
    tag = f'_phi{int(a.phi)}_rho{int(a.rho * 100)}'
    out = HERE / 'figures' / f'india_posterior_predictive{tag}.png'
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f'\nwrote {out}')


if __name__ == '__main__':
    main()
