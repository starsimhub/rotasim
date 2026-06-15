"""
Compare the two maternal-immunity models' achievable protection-vs-age curves:
  - Erlang(n_stages) waning (Alicia's): protection(age) = eff * S_Erlang(age; n, mean_dur).
    A single DETERMINISTIC population curve (every same-age infant identical).
    Bounds (calibrate_maled.py): eff [0.5,0.99], mean_dur [30,300] d, n_stages {1..6}.
  - Titer model (ours): per-infant log-normal initial titer -> common exponential decay ->
    Hill protection. Population curve = eff * E_titer[Hill(titer(age))]; INDIVIDUALS differ.
    Bounds (run_wave.py): eff [0.7,0.99], median [4,60], gsd [1.3,3.5], half-life [25,70] d, slope [1.5,8].

Panels: A raw achievable envelopes; B efficacy-normalized SHAPE envelopes; C individual heterogeneity.

Usage: uv run python experiments/07_history_matching/plot_maternal_compare.py
"""
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
AGE_M = np.linspace(0, 24, 145); AGE_Y = AGE_M / 12.0
LN2 = np.log(2)
ERL = 'tab:blue'; TIT = 'tab:orange'


def erlang_survival(t_years, n, mean_years):
    n = max(1, int(n)); mean = max(float(mean_years), 1e-9); lam = n / mean
    x = lam * np.asarray(t_years, float); s = np.ones_like(x); term = np.ones_like(x)
    for k in range(1, n):
        term = term * x / k; s += term
    return np.exp(-x) * s


def erlang_curve(eff, mean_days, n):
    return eff * erlang_survival(AGE_Y, n, mean_days / 365.25)


def titer_curve(eff, median, gsd, hl_days, slope, n_titer=4000, rng=None):
    rng = rng or np.random.default_rng(0)
    t0 = median * np.exp(np.log(max(gsd, 1 + 1e-9)) * rng.standard_normal(n_titer))   # log-normal
    titer = t0[:, None] * np.exp(-LN2 * AGE_Y[None, :] / (hl_days / 365.25))           # decay, (n_titer, ages)
    th = np.power(np.maximum(titer, 0.0), slope)
    return eff * (th / (th + 1.0)).mean(axis=0)                                        # population mean


def envelope(curves):
    a = np.array(curves); return np.percentile(a, [2.5, 50, 97.5], axis=0)


def main():
    rng = np.random.default_rng(42)
    N = 600
    # --- sample each model's achievable population curves ---
    erl = [erlang_curve(rng.uniform(0.5, 0.99), rng.uniform(30, 300), rng.integers(1, 7)) for _ in range(N)]
    tit = [titer_curve(rng.uniform(0.7, 0.99), np.exp(rng.uniform(np.log(4), np.log(60))),
                       rng.uniform(1.3, 3.5), rng.uniform(25, 70), rng.uniform(1.5, 8.0),
                       rng=np.random.default_rng(i)) for i in range(N)]
    erl_q, tit_q = envelope(erl), envelope(tit)
    # shape-normalized (divide by age-0 value -> isolates waning shape, removes efficacy/height)
    erl_n = envelope([c / c[0] for c in erl]); tit_n = envelope([c / c[0] for c in tit])

    fig, axs = plt.subplots(1, 3, figsize=(17, 5))

    # A. raw achievable envelopes
    a = axs[0]
    a.fill_between(AGE_M, erl_q[0], erl_q[2], color=ERL, alpha=0.22, label='Erlang range (95%)')
    a.plot(AGE_M, erl_q[1], color=ERL, lw=2, label='Erlang median')
    a.fill_between(AGE_M, tit_q[0], tit_q[2], color=TIT, alpha=0.22, label='titer range (95%)')
    a.plot(AGE_M, tit_q[1], color=TIT, lw=2, label='titer median')
    a.set_title('A. achievable maternal protection (over each model\'s priors)')
    a.set_xlabel('age (months)'); a.set_ylabel('maternal protection'); a.set_xlim(0, 24); a.set_ylim(0, 1)
    a.legend(fontsize=8)

    # B. efficacy-normalized SHAPE
    b = axs[1]
    b.fill_between(AGE_M, erl_n[0], erl_n[2], color=ERL, alpha=0.22)
    b.plot(AGE_M, erl_n[1], color=ERL, lw=2, label='Erlang')
    b.fill_between(AGE_M, tit_n[0], tit_n[2], color=TIT, alpha=0.22)
    b.plot(AGE_M, tit_n[1], color=TIT, lw=2, label='titer')
    b.set_title('B. waning SHAPE only (normalized to age-0 = 1)')
    b.set_xlabel('age (months)'); b.set_ylabel('protection / protection(0)'); b.set_xlim(0, 24); b.set_ylim(0, 1)
    b.legend(fontsize=8)

    # C. individual-level: titer heterogeneity vs Erlang, with the POPULATION MEANS MATCHED
    # so the only visible difference is per-infant spread (not a location shift). Titer params
    # = exp-07 posterior medians; Erlang (same efficacy) is fit to the titer mean curve.
    c = axs[2]
    eff, med, gsd, hl, slope = 0.94, 26.3, 2.23, 49.3, 5.16     # ~ exp-07 posterior medians
    tmean = titer_curve(eff, med, gsd, hl, slope)
    best = None                                                 # fit Erlang(n, mean_dur) to titer mean
    for n in range(1, 7):
        for md in np.linspace(20, 365, 300):
            sse = float(((erlang_curve(eff, md, n) - tmean) ** 2).sum())
            if best is None or sse < best[0]:
                best = (sse, n, md)
    _, n_st, md_fit = best
    rngc = np.random.default_rng(1)
    for _ in range(30):
        t0 = med * np.exp(np.log(gsd) * rngc.standard_normal())
        titer = t0 * np.exp(-LN2 * AGE_Y / (hl / 365.25)); th = np.power(np.maximum(titer, 0), slope)
        c.plot(AGE_M, eff * th / (th + 1), color=TIT, alpha=0.28, lw=0.8)
    c.plot([], [], color=TIT, alpha=0.5, lw=1.0, label='titer: individual infants')
    c.plot(AGE_M, tmean, color=TIT, lw=2.8, label='titer: population mean')
    c.plot(AGE_M, erlang_curve(eff, md_fit, n_st), color=ERL, lw=2.5, ls='--',
           label=f'Erlang (n={n_st}, mean={md_fit:.0f}d): matched mean, all infants identical')
    c.set_title('C. SAME population mean — only difference is per-infant heterogeneity')
    c.set_xlabel('age (months)'); c.set_ylabel('maternal protection'); c.set_xlim(0, 24); c.set_ylim(0, 1)
    c.legend(fontsize=8)

    fig.suptitle('Maternal protection: Erlang waning vs log-normal-titer + Hill', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(FIGDIR / 'fig_maternal_compare.png', dpi=130); plt.close(fig)
    print(f'wrote {FIGDIR}/fig_maternal_compare.png')


if __name__ == '__main__':
    main()
