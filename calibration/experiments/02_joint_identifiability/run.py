"""
Exp 02 — joint identifiability on the OPTIMIZED fits (see README.md).

Pulls every completed trial from the existing study DBs (per-rep user_attrs)
and the single-phase logs, computes per-trial (gof_inc, gof_first) plus the
interpretable target features (6-11 mo IR, 24-35 mo IR, first-infection median),
and asks whether the achievable Pareto frontier reaches the MAL-ED data corner.
No new simulations. Run on the VM where the DBs + logs live:

  python run.py
"""
import sys
import re
import json
import glob
import pathlib
import argparse

import numpy as np
import pandas as pd
import yaml
import optuna
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR_LOCAL = HERE.parents[1]
sys.path.insert(0, str(CALIB_DIR_LOCAL))
import process_incidence_maled as P   # noqa: E402

BINS = P.MALED_AGE_BINS

# (label, family, study_name, db_file) — studies that store per-rep user_attrs.
USER_ATTR_STUDIES = [
    ("two-phase (symp_ir fit)",  "two-phase",        "rota_maled_bangladesh_symptomatic_ir",                "rota_maled_bangladesh_symptomatic_ir_2phase.db"),
    ("two-phase (first fit)",    "two-phase",        "rota_maled_bangladesh_first_infection",               "rota_maled_bangladesh_first_infection_2phase.db"),
    ("infnum exp",               "infnum",           "rota_maled_bangladesh_infnum_symptomatic_ir",         "rota_maled_bangladesh_infnum_symptomatic_ir.db"),
    ("infnum erlang6",           "infnum-erlang",    "rota_maled_bangladesh_infnum_erlang6_symptomatic_ir", "rota_maled_bangladesh_infnum_erlang6_symptomatic_ir.db"),
    ("offsets erlang6",          "offsets-erlang",   "rota_maled_bangladesh_ageinfoff_erlang6_symptomatic_ir", "rota_maled_bangladesh_ageinfoff_erlang6_symptomatic_ir.db"),
]
# (label, family, log_glob) — single-phase baselines (no user_attrs; parse logs).
LOG_STUDIES = [
    ("single-phase (symp_ir fit)", "single-phase", "calibrate_maled_bangladesh_worker103016_*.log"),
    ("single-phase (first fit)",   "single-phase", "calibrate_maled_bangladesh_worker103019_*.log"),
]

_RE_TRIAL = re.compile(r'Trial (\d+):\s*$')
_RE_GOF   = re.compile(r'median total = [\d.]+ \(inc=([\d.]+), first_inf=([\d.]+)\)')
_RE_IR    = re.compile(r'Median model IR \(per 100 PM\): <6 m=([\d.]+), 6-11 m=([\d.]+), 12-23 m=([\d.]+), 24-35 m=([\d.]+)')
_RE_FI    = re.compile(r'Median model first-inf \(mo\): Q25=([-\d.nan]+), med=([-\d.nan]+), Q75=([-\d.nan]+)')


def from_user_attrs(label, family, study_name, db_path):
    rows = []
    s = optuna.load_study(study_name=study_name, storage="sqlite:///" + str(db_path))
    for t in s.trials:
        ua = t.user_attrs
        inc, fir = ua.get('per_rep_gof_inc'), ua.get('per_rep_gof_first')
        irr, fii = ua.get('ir_by_age_per_rep'), ua.get('first_inf_per_rep')
        if not (inc and fir and irr and fii):
            continue
        ir = np.nanmedian(np.array(irr, dtype=float), axis=0)
        first_med = float(np.nanmedian([r[1] for r in fii]))
        rows.append(dict(label=label, family=family, trial=t.number,
                         gof_inc=float(np.median(inc)), gof_first=float(np.median(fir)),
                         ir_6_11=float(ir[1]), ir_24_35=float(ir[3]), first_med=first_med))
    return rows


def from_log(label, family, paths):
    rows = []
    for p in sorted(paths):
        cur = None
        for line in open(p):
            m = _RE_TRIAL.search(line)
            if m:
                cur = {'trial': int(m.group(1))}
                continue
            if cur is None:
                continue
            mg = _RE_GOF.search(line)
            if mg:
                cur['gof_inc'], cur['gof_first'] = float(mg.group(1)), float(mg.group(2))
            mi = _RE_IR.search(line)
            if mi:
                cur['ir_6_11'], cur['ir_24_35'] = float(mi.group(2)), float(mi.group(4))
            mf = _RE_FI.search(line)
            if mf:
                cur['first_med'] = float(mf.group(2))
                if {'gof_inc', 'gof_first', 'ir_6_11', 'first_med'} <= cur.keys():
                    rows.append(dict(label=label, family=family, **cur))
                cur = None
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--out-root', default=str(HERE))
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config))
    calib = pathlib.Path(cfg['calib_dir'])
    out_root = pathlib.Path(args.out_root)
    (out_root / 'outputs').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)

    rows = []
    for label, fam, study, db in USER_ATTR_STUDIES:
        try:
            r = from_user_attrs(label, fam, study, calib / db)
            print(f"  {label:<26} {len(r):>3} trials (user_attrs)")
            rows += r
        except Exception as e:
            print(f"  {label:<26} skipped ({type(e).__name__})")
    for label, fam, gpat in LOG_STUDIES:
        r = from_log(label, fam, glob.glob(str(calib / gpat)))
        print(f"  {label:<26} {len(r):>3} trials (log)")
        rows += r

    df = pd.DataFrame(rows)
    df.to_json(out_root / 'outputs' / 'trials.jsonl', orient='records', lines=True)
    print(f"\nPooled {len(df)} optimized trials across {df['family'].nunique()} model families.")

    # ---- Verdict: does any optimized fit reach the data corner (all 3 features)? ----
    tgt_p, tgt_f, tgt_o = cfg['target_peak_6_11'], cfg['target_first_med_mo'], cfg['target_oldest_24_35']
    peak_ok = (df['ir_6_11'] >= tgt_p * (1 - cfg['tol_peak_frac'])) & (df['ir_6_11'] <= tgt_p * (1 + cfg['tol_peak_frac']))
    first_ok = (df['first_med'] - tgt_f).abs() <= cfg['tol_first_med_mo']
    old_ok = df['ir_24_35'] <= cfg['tol_oldest_max']
    inside = df[peak_ok & first_ok & old_ok]
    print("\n=== Does any optimized fit reach the data corner (peak & first-inf & oldest all OK)? ===")
    print(f"  6-11mo peak in {tgt_p*(1-cfg['tol_peak_frac']):.2f}-{tgt_p*(1+cfg['tol_peak_frac']):.2f}: {int(peak_ok.sum())}")
    print(f"  first-inf med within +-{cfg['tol_first_med_mo']} mo of {tgt_f}: {int(first_ok.sum())}")
    print(f"  24-35mo IR <= {cfg['tol_oldest_max']}: {int(old_ok.sum())}")
    print(f"  ALL THREE at once: {len(inside)} / {len(df)}")
    print(f"  best GOF_inc={df['gof_inc'].min():.2f}  best GOF_first={df['gof_first'].min():.3f} "
          f"(no single fit achieves both minima -> L-shaped frontier = structural tension)")
    verdict = ("REACHES corner -- tension was search/local-optimum, not structural"
               if len(inside) else
               "FRONTIER EXCLUDES the data corner -- structural tension confirmed on optimized fits")
    print(f"  VERDICT: {verdict}")

    # ---- figure ----
    fams = sorted(df['family'].unique())
    cmap = dict(zip(fams, plt.cm.tab10(np.linspace(0, 1, len(fams)))))
    fig, (axA, axB) = plt.subplots(1, 2, figsize=(14, 6))

    for fam in fams:
        d = df[df['family'] == fam]
        axA.scatter(d['gof_inc'], np.maximum(d['gof_first'], 1e-3), s=26, color=cmap[fam], alpha=0.65, label=fam)
    axA.set_xscale('log'); axA.set_yscale('log')
    axA.set_xlabel(r'GOF$_{\mathrm{incidence}}$ (lower=better)')
    axA.set_ylabel(r'GOF$_{\mathrm{first\ infection}}$ (lower=better)')
    axA.set_title('(A) Achievable frontier in GOF space\n(ideal = lower-left corner)')
    axA.legend(frameon=False, fontsize=8); axA.grid(True, which='both', ls=':', alpha=0.3)

    for fam in fams:
        d = df[df['family'] == fam]
        axB.scatter(d['ir_6_11'], d['first_med'], s=26, color=cmap[fam], alpha=0.65, label=fam)
    # data target + tolerance box
    axB.scatter([tgt_p], [tgt_f], color='red', marker='*', s=320, edgecolors='k', zorder=6, label='MAL-ED data')
    from matplotlib.patches import Rectangle
    axB.add_patch(Rectangle((tgt_p * (1 - cfg['tol_peak_frac']), tgt_f - cfg['tol_first_med_mo']),
                            tgt_p * 2 * cfg['tol_peak_frac'], 2 * cfg['tol_first_med_mo'],
                            fill=False, ls='--', ec='red', lw=1.5))
    axB.set_xlabel('Model 6-11 mo symptomatic IR (peak; data=5.37)')
    axB.set_ylabel('Model first-infection median, mo (data=7.98)')
    axB.set_title(f'(B) Feature space: peak height vs first-inf timing\n'
                  f'{len(inside)} fit(s) also satisfy 24-35mo IR<= {cfg["tol_oldest_max"]} inside the box')
    axB.legend(frameon=False, fontsize=8); axB.grid(True, ls=':', alpha=0.3)

    fig.suptitle('Exp 02 — joint identifiability on optimized fits (MAL-ED Bangladesh)', fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_root / 'figures' / 'joint_identifiability.png', dpi=150)
    print(f"\nwrote {out_root / 'figures' / 'joint_identifiability.png'}")


if __name__ == '__main__':
    main()
