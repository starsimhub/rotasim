"""Post-hoc overdispersed reweighting of a trajectory-selection run -> posterior, WITHOUT
re-simulating. Recomputes each scored draw's composite log-likelihood from its saved
IR + repeat-fraction + KM survival curve, applies overdispersion (phi on the Poisson IR,
rho design-effect on the survival), and importance-resamples. Fixes the raw-likelihood ESS
collapse for infnum (Dan's recipe); for age it's run for completeness (still collapses --
age uses the emulator-MCMC posterior).

  python reweight_overdispersed.py --model infnum --phi 2 --rho 0.05
  python reweight_overdispersed.py --model age    --phi 2 --rho 0.05
"""
import sys, json, argparse, pathlib
import numpy as np, pandas as pd
THISDIR = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(THISDIR))
import process_incidence_maled as P

SITE = 'bangladesh'
BINS = ['<6 m', '6-11 m', '12-23 m']
EXP_DIR = {'age': '18_age_posterior', 'infnum': '19_infnum_posterior'}
EPS = 1e-9

_t = P.load_targets(SITE); _ir = _t['ir_by_age']; _rf = _t['repeat_frac']
IR_DATA = {b: (int(_ir.loc[b, 'cases']), float(_ir.loc[b, 'PT'])) for b in BINS}
RN = int(_rf['n']); RO = int(round(_rf['frac'] * RN))
_fi = pd.read_csv(THISDIR / 'maled_data' / f'first_infection_{SITE}.csv')
FIRSTINF = _fi[['age_event_months', 'event_observed']].dropna().values
N_REC = len(FIRSTINF)


def _read_jsonl(path):
    out = []
    for line in open(path, errors='ignore'):
        line = line.replace('\x00', '').strip()
        if not line:
            continue
        try:
            out.append(json.loads(line))
        except Exception:
            pass
    return out

def pois(r):
    ll = 0.0
    for b, (c, PT) in IR_DATA.items():
        lam = r['ir_' + b] / 100.0 * PT
        if lam <= 0:
            return np.nan
        ll += c * np.log(lam) - lam
    return ll
def binom(r):
    p = min(max(r['repeat_frac'], 1e-6), 1 - 1e-6); return RO * np.log(p) + (RN - RO) * np.log(1 - p)
def surv(r):
    S = np.array(r['km_surv'], float); ll = 0.0
    for a, ev in FIRSTINF:
        m = int(min(np.floor(a), 35))
        ll += np.log(max(S[m] - S[m + 1], EPS)) if ev == 1 else np.log(max(S[int(min(np.ceil(a), 36))], EPS))
    return ll
def med_from_S(r):
    S = np.array(r['km_surv'], float); below = np.where(S < 0.5)[0]
    if below.size == 0: return 36.0
    m = below[0]
    return 0.0 if m == 0 else float(m - 1 + (S[m - 1] - 0.5) / max(S[m - 1] - S[m], EPS))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', required=True, choices=['age', 'infnum'])
    ap.add_argument('--phi', type=float, default=2.0)    # quasi-Poisson overdispersion on IR counts
    ap.add_argument('--rho', type=float, default=0.05)   # intra-cohort correlation -> survival design-effect
    ap.add_argument('--exp-dir', default=None,
                    help="experiment folder under experiments/ (e.g. 24_infnum_titer_fixedshape); "
                         "defaults to the canonical titer-free posterior for --model")
    ap.add_argument('--tag', default='',
                    help="suffix for output filenames, e.g. _phi3_rho10 -> posterior_overdispersed_phi3_rho10.csv "
                         "(avoids clobbering prior runs)")
    a = ap.parse_args()
    DEFF = 1.0 + (N_REC - 1) * a.rho
    exp = a.exp_dir or EXP_DIR[a.model]
    out_dir = THISDIR / 'experiments' / exp / 'outputs'
    nroy = pd.read_csv(out_dir / 'nroy_draw.csv').reset_index(drop=True)
    recs = _read_jsonl(out_dir / 'sir_results.jsonl')
    print(f"{a.model} [{exp}]: {len(recs)} scored draws; phi={a.phi}, rho={a.rho} (N_rec={N_REC}, DEFF={DEFF:.1f})")

    rows = []
    for r in recs:
        if r.get('logL') is None or 'km_surv' not in r:
            continue
        pp = pois(r)
        if not np.isfinite(pp):
            ll = -np.inf
        else:
            ll = pp / a.phi + binom(r) + surv(r) / DEFF
        rows.append((r['idx'], ll, r))
    idx = np.array([x[0] for x in rows]); logL = np.array([x[1] for x in rows], float)
    fin = np.isfinite(logL)
    w = np.zeros(len(logL))
    if fin.any():
        w[fin] = np.exp(logL[fin] - np.nanmax(logL[fin])); w /= w.sum()
    ess = float(1.0 / np.sum(w ** 2)) if fin.any() else 0.0
    print(f"  finite={int(fin.sum())}/{len(logL)}  ESS={ess:.1f}  max_logL={np.nanmax(logL[fin]) if fin.any() else float('nan'):.1f}")

    rng = np.random.default_rng(0)
    pick = rng.choice(len(rows), size=len(rows), replace=True, p=w)
    post = nroy.iloc[[idx[i] for i in pick]].reset_index(drop=True)
    post.to_csv(out_dir / f'posterior_overdispersed{a.tag}.csv', index=False)

    # weighted posterior-predictive
    irw = {b: float(np.sum(w * np.array([r['ir_' + b] for _, _, r in rows]))) for b in BINS}
    repw = float(np.sum(w * np.array([r['repeat_frac'] for _, _, r in rows])))
    medw = float(np.sum(w * np.array([med_from_S(r) for _, _, r in rows])))
    stats = dict(model=a.model, phi=a.phi, rho=a.rho, deff=DEFF, finite=int(fin.sum()), ess=ess,
                 wpp_ir=irw, wpp_repeat=repw, wpp_first_inf_median=medw,
                 target_ir={b: IR_DATA[b] for b in BINS}, target_repeat=_rf['frac'], target_first_inf_median=12.12)
    json.dump(stats, (out_dir / f'overdispersed_stats{a.tag}.json').open('w'), indent=2)
    print(f"  wPP IR: {[round(irw[b],2) for b in BINS]} (target [1.91,5.37,2.35]) | repeat {repw:.3f} (0.403) | first-inf med {medw:.2f} (12.12)")
    print(f"  wrote posterior_overdispersed{a.tag}.csv ({len(post)} resamples) + overdispersed_stats{a.tag}.json")


if __name__ == '__main__':
    main()
