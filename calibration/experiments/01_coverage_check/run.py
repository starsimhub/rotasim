"""
Exp 01 — prior-predictive coverage check (see README.md).

Draws independent prior samples (NOT Optuna), 1 replicate each, and asks whether
any SINGLE draw jointly reaches both MAL-ED Bangladesh targets (symptomatic
IR-by-age and age-at-first-infection). Writes each draw's raw summary to
outputs/results.jsonl incrementally (so a spot-VM eviction loses nothing and a
re-run resumes), then computes marginal + joint coverage and plots to figures/.

Run on the VM:
  python run.py                 # uses config.yaml
  python run.py --n-draws 2 --n-agents 4000 --out-root /tmp/smoke   # quick smoke
"""
import sys
import json
import pathlib
import argparse

import numpy as np
import pandas as pd
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = pathlib.Path(__file__).resolve().parent
CALIB_DIR = HERE.parents[1]          # .../calibration
sys.path.insert(0, str(CALIB_DIR))

from calibrate_maled import (        # noqa: E402
    _run_one_replicate, SITE_DEMOGRAPHICS,
    FIXED_REPORTING_RATE, FIXED_CONSTANT_SEVERITY,
    process_incidence_maled, thisdir,
)
from coverage_check_maled import sample_prior, BINS, QUARTS, INC_OK, FI_OK, CAL_WINDOW  # noqa: E402


def _indexed_worker(task):
    """Spawn worker: run one prior draw, return a small JSON-able summary."""
    idx, sim_config, params, seed, cal_window = task
    mo = _run_one_replicate((sim_config, params, seed, cal_window))
    return {
        'idx': idx,
        'params': params,
        'ir': [float(mo['ir_by_age'].loc[b, 'IR']) for b in BINS],
        'first_inf': {k: float(mo['first_infection'][k]) for k in QUARTS},
    }


def _row_gof(row, targets):
    mo = {'ir_by_age': pd.DataFrame({'IR': row['ir']}, index=BINS),
          'first_infection': row['first_inf']}
    g = process_incidence_maled.gof(mo, targets, fit_target='joint')
    return g['gof_incidence'], g['gof_first_infection']


def analyse_and_plot(rows, targets, cfg, fig_path):
    rows = sorted(rows, key=lambda r: r['idx'])
    ir = np.array([r['ir'] for r in rows])
    fi = np.array([[r['first_inf'][k] for k in QUARTS] for r in rows])
    gof = np.array([_row_gof(r, targets) for r in rows])
    gof_inc, gof_fi = gof[:, 0], gof[:, 1]
    tgt_ir = np.array([targets['ir_by_age'].loc[b, 'IR'] for b in BINS])
    tgt_fi = np.array([targets['first_infection'][k] for k in QUARTS])
    n = len(rows)

    print(f"\n=== MARGINAL coverage ({n} draws) ===")
    marg = True
    for j, b in enumerate(BINS):
        lo, hi = ir[:, j].min(), ir[:, j].max(); inside = lo <= tgt_ir[j] <= hi; marg &= inside
        print(f"  IR {b:<8}: data={tgt_ir[j]:6.2f}  draws[{lo:6.2f},{hi:6.2f}]  {'IN ' if inside else 'OUT'}")
    for j, k in enumerate(QUARTS):
        lo, hi = fi[:, j].min(), fi[:, j].max(); inside = lo <= tgt_fi[j] <= hi; marg &= inside
        print(f"  first-inf {k:<6}: data={tgt_fi[j]:6.2f}  draws[{lo:6.2f},{hi:6.2f}]  {'IN ' if inside else 'OUT'}")

    joint = np.where((gof_inc <= INC_OK) & (gof_fi <= FI_OK))[0]
    norm = (gof_inc / INC_OK) ** 2 + (gof_fi / FI_OK) ** 2
    bi = int(np.argmin(norm))
    print("\n=== JOINT coverage (single draw near BOTH targets) ===")
    print(f"  draws with gof_inc<={INC_OK} AND gof_first<={FI_OK}: {len(joint)} / {n}")
    print(f"  best single-target: min gof_inc={gof_inc.min():.2f}, min gof_first={gof_fi.min():.3f}")
    print(f"  closest-to-both draw: gof_inc={gof_inc[bi]:.2f}, gof_first={gof_fi[bi]:.3f}, "
          f"IR={[round(float(v),2) for v in ir[bi]]}")
    covered = len(joint) > 0
    verdict = ("JOINTLY COVERED -- a draw reaches both (tension is search/likelihood, not structural)"
               if covered else
               "NOT JOINTLY COVERED -- no single draw reaches both (Pareto tension is STRUCTURAL)")
    print(f"\n  VERDICT: {verdict}")
    if marg and not covered:
        print("  NOTE: marginal passes but joint fails -> classic structural-tension signature.")

    fig, (axA, axB) = plt.subplots(1, 2, figsize=(13, 5.5))
    rng = np.random.default_rng(0)
    x = np.arange(len(BINS))
    for j in range(len(BINS)):
        axA.scatter(x[j] + rng.uniform(-0.12, 0.12, n), ir[:, j], s=10, color='#bbbbbb', alpha=0.55)
    axA.scatter(x, tgt_ir, color='red', marker='_', s=600, lw=3, zorder=5, label='MAL-ED data')
    axA.set_yscale('symlog', linthresh=0.1); axA.set_xticks(x); axA.set_xticklabels(BINS)
    axA.set_ylabel('Symptomatic IR (per 100 PM)')
    axA.set_title(f'(A) Marginal coverage\n{n} prior draws (grey) vs data (red)')
    axA.legend(frameon=False); axA.spines[['top', 'right']].set_visible(False)

    axB.scatter(gof_inc, np.maximum(gof_fi, 1e-3), s=22, color='#2ca02c', alpha=0.6, label='prior draws')
    axB.axvline(INC_OK, color='k', ls=':', alpha=0.5); axB.axhline(FI_OK, color='k', ls=':', alpha=0.5)
    axB.scatter([gof_inc[bi]], [max(gof_fi[bi], 1e-3)], s=160, marker='*', color='#2ca02c',
                edgecolors='k', zorder=6, label='closest to both')
    axB.set_xscale('log'); axB.set_yscale('log')
    axB.set_xlabel(r'GOF$_{\mathrm{incidence}}$ (lower=better)')
    axB.set_ylabel(r'GOF$_{\mathrm{first\ infection}}$ (lower=better)')
    axB.set_title(f'(B) Joint coverage\nlower-left box = both acceptable; {len(joint)} draw(s) inside')
    axB.legend(frameon=False, loc='upper right'); axB.grid(True, which='both', ls=':', alpha=0.3)
    fig.suptitle(f"Coverage check — MAL-ED Bangladesh — {cfg['symptom_model']} (Erlang n={cfg['maternal_n_stages']})",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(fig_path, dpi=150)
    print(f'\nwrote {fig_path}')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=str(HERE / 'config.yaml'))
    ap.add_argument('--out-root', default=str(HERE), help='dir holding outputs/ and figures/')
    ap.add_argument('--n-draws', type=int, default=None)    # CLI overrides for smoke
    ap.add_argument('--n-agents', type=int, default=None)
    ap.add_argument('--n-workers', type=int, default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    for k in ('n_draws', 'n_agents', 'n_workers'):
        v = getattr(args, k)
        if v is not None:
            cfg[k] = v

    out_root = pathlib.Path(args.out_root)
    (out_root / 'outputs').mkdir(parents=True, exist_ok=True)
    (out_root / 'figures').mkdir(parents=True, exist_ok=True)
    jsonl = out_root / 'outputs' / 'results.jsonl'

    demo = SITE_DEMOGRAPHICS[cfg['site']]
    sim_config = dict(
        n_agents=cfg['n_agents'], start='2003-01-01', stop='2013-01-01', n_contacts=7,
        birth_rate=demo['birth_rate'], death_rate=demo['death_rate'],
        constant_severity=FIXED_CONSTANT_SEVERITY, reporting_rate=FIXED_REPORTING_RATE,
        age_data_path=str(thisdir / 'uk_age_data.csv'), p_asymp_detect=0.4,
        symptom_model=cfg['symptom_model'], maternal_n_stages=cfg['maternal_n_stages'],
    )
    targets = process_incidence_maled.load_targets(cfg['site'])

    # Deterministic draws + seeds (so a resumed run reproduces the same tasks).
    rng = np.random.default_rng(cfg['seed'])
    draws = [sample_prior(rng, cfg['symptom_model']) for _ in range(cfg['n_draws'])]
    sim_seeds = rng.integers(0, 1_000_000, cfg['n_draws']).tolist()

    done = set()
    if jsonl.exists():
        with jsonl.open() as f:
            done = {json.loads(line)['idx'] for line in f if line.strip()}
    todo = [i for i in range(cfg['n_draws']) if i not in done]
    print(f"Coverage check: {cfg['n_draws']} draws ({len(done)} done, {len(todo)} to run) | "
          f"symptom_model={cfg['symptom_model']} Erlang n={cfg['maternal_n_stages']} n_agents={cfg['n_agents']}")

    if todo:
        from multiprocessing import get_context
        tasks = [(i, sim_config, draws[i], int(sim_seeds[i]), CAL_WINDOW) for i in todo]
        nw = min(cfg['n_workers'], len(tasks))
        with get_context('spawn').Pool(processes=nw) as pool:
            for res in pool.imap_unordered(_indexed_worker, tasks):
                with jsonl.open('a') as f:
                    f.write(json.dumps(res) + '\n')
                print(f"  draw {res['idx']:>3} done  IR={[round(v,2) for v in res['ir']]} "
                      f"first-inf med={res['first_inf']['median']:.1f}")

    rows = [json.loads(line) for line in jsonl.open() if line.strip()]
    analyse_and_plot(rows, targets, cfg, str(out_root / 'figures' / 'coverage_check.png'))


if __name__ == '__main__':
    main()
