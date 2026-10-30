"""Maternal-antibody protection vs infant age (Bangladesh), CURRENT parameters.

Reproducible regeneration of the titer-vs-Erlang maternal comparison, using the maternal model's
actual formulas (rotasim.immunity) and the current identified params:
  - TITER (the model used by all corrected runs): FIXED_TITER_SHAPE (median 20, gsd 2.3,
    half-life 50d, Hill 4.7) for the SHAPE; efficacy from the exp27 (corrected infnum) posterior
    (median + 5-95%). Population curve = efficacy * E_t0[ Hill(titer(age)) ], t0 ~ lognormal.
  - ERLANG: efficacy * erlang_survival(age, n=6, mean_dur). No current Erlang *fit* exists (the
    corrected work used titer), so mean_dur is least-squares MATCHED to the titer shape, to show
    the two maternal STRUCTURES coincide (the original point) under current params.

Pure analytic curves -- no simulation, no VM. Run:
  python maternal_protection_by_age.py
"""
import sys, pathlib
import numpy as np
import pandas as pd
import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB = HERE.parents[1]
sys.path.insert(0, str(CALIB.parent))            # repo root, for `import rotasim`
from rotasim.immunity import erlang_survival     # the model's actual Erlang survival

# --- current params ---
TITER = dict(median=20.0, gsd=2.3, half_life_days=50.0, hill=4.7)   # FIXED_TITER_SHAPE
EXP27_POST = CALIB / 'experiments' / '27_infnum_titer_fixedshape_corrected' / 'outputs' / 'posterior_overdispersed_phi3_rho10.csv'
eff = pd.read_csv(EXP27_POST)['maternal_efficacy']
eff_med, eff_lo, eff_hi = eff.median(), eff.quantile(.05), eff.quantile(.95)

age_m = np.linspace(0, 18, 181)
age_y = age_m / 12.0

# titer SHAPE: population-mean Hill protection at each age (MC over lognormal initial titer)
rng = np.random.default_rng(0)
sigma = np.log(TITER['gsd']); t0 = TITER['median'] * np.exp(sigma * rng.standard_normal(200_000))
hl_y = TITER['half_life_days'] / 365.25
titer = t0[None, :] * np.exp(-np.log(2) * age_y[:, None] / hl_y)     # (age, draws)
th = np.power(np.maximum(titer, 0.0), TITER['hill'])
titer_shape = (th / (th + 1.0)).mean(axis=1)                        # E[Hill] by age (efficacy=1)

# Erlang(6): match mean_dur (years) to the titer shape by least squares
cands = np.linspace(0.2, 1.2, 401)
sse = [np.sum((titer_shape - erlang_survival(age_y, 6, m)) ** 2) for m in cands]
mean_dur = cands[int(np.argmin(sse))]
erlang_shape = erlang_survival(age_y, 6, mean_dur)

fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(age_m, eff_med * titer_shape, color='#2c7fb8', lw=2.5,
        label=f'titer maternal (FIXED_TITER_SHAPE; efficacy {eff_med:.2f})')
ax.fill_between(age_m, eff_lo * titer_shape, eff_hi * titer_shape, color='#2c7fb8', alpha=0.2,
                label='efficacy 5-95% (exp27 posterior)')
ax.plot(age_m, eff_med * erlang_shape, '--', color='#c0392b', lw=2.5,
        label=f'Erlang(6) maternal (mean {mean_dur*12:.1f} mo, matched)')
ax.set_xlabel('infant age (months)'); ax.set_ylabel('maternal protection against infection')
ax.set_title('Fitted maternal-antibody protection vs age (Bangladesh, current params)\n'
             'titer and Erlang structures coincide')
ax.set_xlim(0, 18); ax.set_ylim(0, 1); ax.legend(frameon=False); ax.grid(alpha=0.3)
fig.tight_layout()
out = HERE / 'figures' / 'maternal_protection_by_age_current.png'
fig.savefig(out, dpi=140)
print(f"efficacy (exp27): median {eff_med:.3f} [{eff_lo:.3f}, {eff_hi:.3f}]")
print(f"matched Erlang(6) mean duration: {mean_dur*12:.1f} months")
print(f"titer protection at birth/3mo/6mo/9mo: "
      + ", ".join(f"{age_m[i]:.0f}mo {eff_med*titer_shape[i]:.2f}" for i in (0, 30, 60, 90)))
print(f"wrote {out}")
