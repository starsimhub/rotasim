"""
Exp 07 — "what the MAL-ED trial looks like in the model": a swimmer plot of individual
enrollee life courses from one posterior trajectory (reproduced exactly on capy).

Top panel : population maternal protection (expected) + susceptibility vs age (the window).
Bottom    : one lane per sampled enrollee, age 0-24mo. Each infection is a marker:
              red star  = detected symptomatic  (a MAL-ED symptomatic case)
              blue dot  = detected asymptomatic  (surveillance-detected silent infection)
              gray x    = undetected infection   (the model has it; the trial misses it)
            lane = follow-up (birth -> dropout); '|' marks early dropout; faint verticals
            are the surveillance-stool schedule (monthly to 12mo, then quarterly).
  --per-infant: shade each lane's individual maternal-protected window (illustrative -- the
                model redraws titers each step so per-infant titer isn't persisted; we draw
                each infant a titer from the same log-normal, seeded by UID).

Usage:
  uv run python experiments/07_history_matching/plot_swimmer.py                       # v1
  uv run python experiments/07_history_matching/plot_swimmer.py --infile swimmer2.json \
        --out fig_swimmer2.png --per-infant                                            # v2 (representative)
"""
import json, argparse
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
VISITS = list(range(1, 13)) + [15, 18, 21, 24]


def maternal_curve(am, med, gsd, hl_days, slope, eff, n=40000, seed=0):
    rng = np.random.default_rng(seed); sigma = np.log(max(gsd, 1 + 1e-9))
    t0 = med * np.exp(sigma * rng.standard_normal(n)); hl_y = hl_days / 365.25
    out = []
    for a in np.asarray(am):
        t = np.maximum(t0 * np.exp(-np.log(2) * (a / 12) / hl_y), 0) ** slope
        out.append(float(np.mean(eff * (t / (t + 1)))))
    return np.array(out)


def infant_window(uid, med, gsd, hl_days, slope, eff, thresh=0.5):
    """Illustrative per-infant maternal-protected window: age (months) where one titer draw
    (seeded by uid) drops below `thresh` protection."""
    rng = np.random.default_rng(int(uid)); sigma = np.log(max(gsd, 1 + 1e-9))
    t0 = med * np.exp(sigma * rng.standard_normal()); hl_y = hl_days / 365.25
    for a in np.arange(0, 24.01, 0.25):
        t = max(t0 * np.exp(-np.log(2) * (a / 12) / hl_y), 0) ** slope
        if eff * (t / (t + 1)) < thresh:
            return a
    return 24.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--infile', default='swimmer.json')
    ap.add_argument('--out', default='fig_swimmer.png')
    ap.add_argument('--per-infant', action='store_true', help='shade each lane''s maternal window')
    args = ap.parse_args()

    sw = json.load(open(HERE / 'outputs' / args.infile))
    idx = sw['idx']; cohort = sw['cohort']; n_enr = sw.get('n_enrolled')
    rec = {json.loads(l)['idx']: json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')}[idx]
    pr = {k[4:]: v for k, v in rec.items() if k.startswith('par_')}
    am = np.arange(0, 25)
    matc = maternal_curve(am, pr['titer_median'], pr['titer_gsd'], pr['titer_half_life_days'],
                          pr['hill_slope'], pr['maternal_efficacy'])
    ins = [x for x in json.load(open(HERE / 'outputs' / 'inside_model.json')) if x['idx'] == idx][0]['inside']
    sus = np.array(ins['sus_by_age'])[:25]
    nlane = len(cohort)

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 2.4 + 0.13 * nlane), height_ratios=[1, 0.55 * nlane / 10 + 1], sharex=True)
    ax0.fill_between(am, 0, matc, color='#4C72B0', alpha=0.22)
    ax0.plot(am, matc, color='#4C72B0', lw=2, label='maternal protection (expected)')
    ax0.plot(am, sus, color='crimson', lw=2, label='susceptibility')
    ax0.set_ylim(0, 1); ax0.set_ylabel('protection /\nsusceptibility'); ax0.legend(fontsize=8, loc='center right')
    enr = f'{len(cohort)} of {n_enr} enrolled' if n_enr else f'{len(cohort)} enrollees'
    ax0.set_title(f'What the MAL-ED trial looks like in the model — idx {idx} ({enr}; reproduced on capy)')

    msf = min(1.0, 40.0 / nlane)   # marker scale: full size for few lanes, smaller when dense
    for v in VISITS:
        ax1.axvline(v, color='0.88', lw=0.8, zorder=0)
    for lane, c in enumerate(cohort):
        if args.per_infant:
            aw = infant_window(c['uid'], pr['titer_median'], pr['titer_gsd'], pr['titer_half_life_days'],
                               pr['hill_slope'], pr['maternal_efficacy'])
            ax1.barh(lane, aw, height=0.85, left=0, color='#4C72B0', alpha=0.16, zorder=0)
        ax1.plot([0, c['exit_m']], [lane, lane], color='0.78', lw=1.0, zorder=1)
        if c['exit_m'] < 23.5:
            ax1.plot(c['exit_m'], lane, marker='|', color='0.4', ms=7, mew=1.3, zorder=2)
        for e in c['events']:
            if e['age_m'] > c['exit_m']:
                continue
            if e['detected'] and e['symp']:
                ax1.scatter(e['age_m'], lane, marker='*', s=150 * msf, color='crimson', zorder=4, edgecolor='k', linewidth=0.3)
            elif e['detected']:
                ax1.scatter(e['age_m'], lane, marker='o', s=42 * msf, color='steelblue', zorder=4, edgecolor='k', linewidth=0.3)
            else:
                ax1.scatter(e['age_m'], lane, marker='x', s=28 * msf, color='0.55', alpha=0.8, zorder=3)
    ax1.set_xlim(0, 24); ax1.set_ylim(-1, nlane); ax1.set_xticks(range(0, 25, 3))
    ax1.set_xlabel('age (months)'); ax1.set_ylabel('enrollee (sorted by follow-up length)')
    handles = [Line2D([], [], marker='*', color='crimson', ls='', ms=12, label='detected symptomatic (MAL-ED case)'),
               Line2D([], [], marker='o', color='steelblue', ls='', ms=8, label='detected asymptomatic (surveillance)'),
               Line2D([], [], marker='x', color='0.55', ls='', ms=8, label='undetected infection (trial misses)'),
               Line2D([], [], marker='|', color='0.4', ls='', ms=10, label='dropout')]
    if args.per_infant:
        handles.append(Line2D([], [], marker='s', color='#4C72B0', alpha=0.4, ls='', ms=10,
                              label='per-infant maternal window (illustrative)'))
    ax1.legend(handles=handles, fontsize=7.5, loc='lower right', ncol=2)
    fig.tight_layout()
    fig.savefig(FIGDIR / args.out, dpi=130); plt.close(fig)
    ndet = sum(e['detected'] for c in cohort for e in c['events'])
    ntot = sum(len(c['events']) for c in cohort)
    print(f'wrote {FIGDIR}/{args.out}  ({len(cohort)} lanes, {ntot} infections, {ndet} detected, enrolled={n_enr})')


if __name__ == '__main__':
    main()
