"""
Exp 07 — inside-the-model mechanism figure, from the faithfully-reproduced posterior
trajectories (inside_model.json, re-run on capy with exact NROY params).

  fig_inside.png:
    A. susceptibility-by-age decomposition (top-weight trajectory): maternal protection
       wanes -> a susceptibility window opens ~6-11mo -> infection prevalence peaks there
       -> acquired protection builds -> susceptibility falls. THE mechanism behind the peak.
    B. mean prior infections vs age (the acquired-immunity ladder building up) -- all traj.
    C. prevalence over time by age band (top-weight) -- where infection sits & endemic dynamics.
    D. susceptibility-by-age for all top trajectories (mechanism is consistent across posterior).

Usage: uv run python experiments/07_history_matching/plot_inside.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import nbinom, betabinom

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
IR = [('<6 m', 27, 1415.0), ('6-11 m', 74, 1379.0), ('12-23 m', 59, 2515.0)]


def _logL(r, phi=2.0, rho=0.05):
    if (not r.get('ok')) or (r.get('frac_ever_detected') or 0) < 0.05: return -np.inf
    ll = 0.0
    for b, c, PT in IR:
        mu = r[f'ir_symp_{b}'] / 100.0 * PT
        if mu <= 0: return -np.inf
        rr = mu / (phi - 1.0); ll += nbinom.logpmf(c, rr, rr / (rr + mu))
    for k, n, p in [(58, 136, r['repeat_detected_frac']), (136, 213, r['frac_ever_detected'])]:
        p = min(max(p, 1e-6), 1 - 1e-6); M = (1 - rho) / rho
        ll += betabinom.logpmf(k, n, p * M, (1 - p) * M)
    return ll


def maternal_curve(am_months, med, gsd, hl_days, slope, eff, n=40000, seed=0):
    """Expected maternal protection vs age from the titer params (Monte-Carlo over the
    log-normal initial titer). Used because the connector doesn't persist per-infant titers
    (maternal_titer0 stays NaN) -- the population curve is still well-defined."""
    rng = np.random.default_rng(seed)
    sigma = np.log(max(gsd, 1 + 1e-9))
    t0 = med * np.exp(sigma * rng.standard_normal(n))
    hl_y = hl_days / 365.25
    out = []
    for a in np.asarray(am_months):
        titer = t0 * np.exp(-np.log(2) * (a / 12.0) / hl_y)
        th = np.power(np.maximum(titer, 0.0), slope)
        out.append(float(np.mean(eff * (th / (th + 1.0)))))
    return np.array(out)


def main():
    data = json.load(open(HERE / 'outputs' / 'inside_model.json'))
    # weight each trajectory (to order + pick the top one for the detailed panels)
    recs = {json.loads(l)['idx']: json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')}
    lls = {d['idx']: _logL(recs[d['idx']]) for d in data}
    mx = max(lls.values())
    for d in data:
        d['w'] = float(np.exp(lls[d['idx']] - mx))     # weight relative to the top trajectory
    data.sort(key=lambda d: -d['w'])
    top = data[0]; ins = top['inside']
    am = np.array(ins['age_months'])
    cols = plt.cm.viridis(np.linspace(0.15, 0.85, len(data)))

    fig, axs = plt.subplots(2, 2, figsize=(14, 9))

    # --- A. susceptibility-by-age decomposition (top trajectory) ---
    a = axs[0, 0]
    sus = np.array(ins['sus_by_age']); tot = np.array(ins['tot_by_age'])
    pr = recs[top['idx']]   # titer params for the expected maternal curve
    matc = maternal_curve(am, pr['par_titer_median'], pr['par_titer_gsd'],
                          pr['par_titer_half_life_days'], pr['par_hill_slope'], pr['par_maternal_efficacy'])
    a.fill_between(am, 0, matc, color='#4C72B0', alpha=0.22, label='maternal protection (expected)')
    a.plot(am, matc, color='#4C72B0', lw=1.5)
    a.plot(am, tot, color='k', lw=1.8, label='total protection (1 − susceptibility)')
    a.plot(am, sus, color='crimson', lw=2.5, label='susceptibility')
    a.set_xlim(0, 24); a.set_ylim(0, 1); a.set_xlabel('age (months)'); a.set_ylabel('protection / susceptibility')
    a.set_title(f'A. why the peak: protection vs age (idx {top["idx"]}, top weight)')
    a2 = a.twinx()
    a2.plot(am, ins['prev_by_age'], color='darkorange', lw=2.0, label='infection prevalence (right)')
    a2.set_ylabel('infection prevalence', color='darkorange'); a2.tick_params(axis='y', colors='darkorange')
    a2.set_ylim(0, max(ins['prev_by_age'][:24]) * 1.5 + 1e-6)
    h1, l1 = a.get_legend_handles_labels(); h2, l2 = a2.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, fontsize=8, loc='upper right')
    a.annotate('maternal wanes →\nsusceptibility window', xy=(8, 0.55), fontsize=7.5, color='crimson', ha='center')

    # --- B. acquired-immunity ladder: mean prior infections vs age ---
    b = axs[0, 1]
    for d, c in zip(data, cols):
        b.plot(d['inside']['age_months'], d['inside']['nrec_by_age'], color=c, lw=1.8,
               label=f"idx {d['idx']} (w_rel={d['w']:.2f})")
    b.set_xlim(0, 24); b.set_xlabel('age (months)'); b.set_ylabel('mean # prior infections')
    b.set_title('B. acquired-immunity ladder building with age'); b.legend(fontsize=8)

    # --- C. prevalence over time by age band (top trajectory), science window only ---
    c = axs[1, 0]; t = np.array(ins['t']); win = t >= 5.0
    c.plot(t[win], np.array(ins['prev_inf'])[win], label='infants <1y', color='crimson', lw=1)
    c.plot(t[win], np.array(ins['prev_yng'])[win], label='young 1-5y', color='#4C72B0', lw=1)
    c.plot(t[win], np.array(ins['prev_rest'])[win], label='rest 5y+', color='gray', lw=1)
    c.plot(t[win], np.array(ins['prev'])[win], label='overall', color='k', lw=1.5)
    c.set_xlim(5, 10); c.set_xlabel('years since start (2003)'); c.set_ylabel('infection prevalence')
    c.set_title(f'C. endemic prevalence by age band — science window (idx {top["idx"]})'); c.legend(fontsize=8)

    # --- D. susceptibility-by-age across trajectories (consistency) ---
    d = axs[1, 1]
    for dd, col in zip(data, cols):
        d.plot(dd['inside']['age_months'], dd['inside']['sus_by_age'], color=col, lw=1.8,
               label=f"idx {dd['idx']}")
    d.set_xlim(0, 24); d.set_ylim(0, 1); d.set_xlabel('age (months)'); d.set_ylabel('mean susceptibility')
    d.set_title('D. susceptibility-by-age (consistent across posterior)'); d.legend(fontsize=8)

    fig.suptitle('Inside the model — top posterior trajectories (exactly reproduced on capy)', fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(FIGDIR / 'fig_inside.png', dpi=120); plt.close(fig)
    print(f'wrote {FIGDIR}/fig_inside.png  (top idx {top["idx"]}, weight {top["w"]:.3f})')


if __name__ == '__main__':
    main()
