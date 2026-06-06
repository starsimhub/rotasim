"""
Exp 07 — "what the MAL-ED trial looks like in the model": a swimmer plot of individual
enrollee life courses from one posterior trajectory (swimmer.json, exactly reproduced on capy).

Top panel : population maternal protection (expected) + susceptibility vs age (the window).
Bottom    : one lane per sampled enrollee, age 0-24mo. Each infection is a marker:
              red star  = detected symptomatic  (a MAL-ED symptomatic case)
              blue dot  = detected asymptomatic  (surveillance-detected silent infection)
              gray x    = undetected infection   (the model has it; the trial misses it)
            lane = follow-up (birth -> dropout); '|' marks early dropout; faint verticals
            are the surveillance-stool schedule (monthly to 12mo, then quarterly).

Usage: uv run python experiments/07_history_matching/plot_swimmer.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

HERE = Path(__file__).resolve().parent
FIGDIR = HERE / 'figures'; FIGDIR.mkdir(exist_ok=True)
VISITS = list(range(1, 13)) + [15, 18, 21, 24]      # surveillance-stool schedule (months)


def maternal_curve(am, med, gsd, hl_days, slope, eff, n=40000, seed=0):
    rng = np.random.default_rng(seed); sigma = np.log(max(gsd, 1 + 1e-9))
    t0 = med * np.exp(sigma * rng.standard_normal(n)); hl_y = hl_days / 365.25
    return np.array([float(np.mean(eff * ((np.maximum(t0 * np.exp(-np.log(2) * (a / 12) / hl_y), 0) ** slope) /
                     (np.maximum(t0 * np.exp(-np.log(2) * (a / 12) / hl_y), 0) ** slope + 1)))) for a in am])


def main():
    sw = json.load(open(HERE / 'outputs' / 'swimmer.json'))
    idx = sw['idx']; cohort = sw['cohort']
    rec = {json.loads(l)['idx']: json.loads(l) for l in open(HERE / 'outputs' / 'sir_results.jsonl')}[idx]
    pr = {k[4:]: v for k, v in rec.items() if k.startswith('par_')}
    am = np.arange(0, 25)
    matc = maternal_curve(am, pr['titer_median'], pr['titer_gsd'], pr['titer_half_life_days'],
                          pr['hill_slope'], pr['maternal_efficacy'])
    ins = [x for x in json.load(open(HERE / 'outputs' / 'inside_model.json')) if x['idx'] == idx][0]['inside']
    sus = np.array(ins['sus_by_age'])[:25]

    fig, (ax0, ax1) = plt.subplots(2, 1, figsize=(12, 9), height_ratios=[1, 3.2], sharex=True)

    # top: population curves
    ax0.fill_between(am, 0, matc, color='#4C72B0', alpha=0.22)
    ax0.plot(am, matc, color='#4C72B0', lw=2, label='maternal protection (expected)')
    ax0.plot(am, sus, color='crimson', lw=2, label='susceptibility')
    ax0.set_ylim(0, 1); ax0.set_ylabel('protection /\nsusceptibility'); ax0.legend(fontsize=8, loc='center right')
    ax0.set_title(f'What the MAL-ED trial looks like in the model — idx {idx} (top posterior trajectory, reproduced on capy)')

    # bottom: swimmer lanes
    for v in VISITS:
        ax1.axvline(v, color='0.85', lw=0.8, zorder=0)
    for lane, c in enumerate(cohort):
        ax1.plot([0, c['exit_m']], [lane, lane], color='0.75', lw=1.2, zorder=1)
        if c['exit_m'] < 23.5:                                   # dropped out before 24mo
            ax1.plot(c['exit_m'], lane, marker='|', color='0.4', ms=9, mew=1.5, zorder=2)
        for e in c['events']:
            if e['age_m'] > c['exit_m']:
                continue
            if e['detected'] and e['symp']:
                ax1.scatter(e['age_m'], lane, marker='*', s=150, color='crimson', zorder=4, edgecolor='k', linewidth=0.3)
            elif e['detected']:
                ax1.scatter(e['age_m'], lane, marker='o', s=42, color='steelblue', zorder=4, edgecolor='k', linewidth=0.3)
            else:
                ax1.scatter(e['age_m'], lane, marker='x', s=28, color='0.55', alpha=0.8, zorder=3)
    ax1.set_xlim(0, 24); ax1.set_ylim(-1, len(cohort)); ax1.set_xticks(range(0, 25, 3))
    ax1.set_xlabel('age (months)'); ax1.set_ylabel('enrollee (sorted by follow-up length)')
    handles = [Line2D([], [], marker='*', color='crimson', ls='', ms=12, label='detected symptomatic (MAL-ED case)'),
               Line2D([], [], marker='o', color='steelblue', ls='', ms=8, label='detected asymptomatic (surveillance)'),
               Line2D([], [], marker='x', color='0.55', ls='', ms=8, label='undetected infection (trial misses)'),
               Line2D([], [], marker='|', color='0.4', ls='', ms=10, label='dropout')]
    ax1.legend(handles=handles, fontsize=8, loc='lower right', ncol=2)
    ax1.text(0.2, len(cohort) - 1.5, 'faint verticals = surveillance-stool schedule (monthly→12mo, then quarterly)',
             fontsize=7, color='0.5')
    fig.tight_layout()
    fig.savefig(FIGDIR / 'fig_swimmer.png', dpi=130); plt.close(fig)
    ndet = sum(e['detected'] for c in cohort for e in c['events'])
    ntot = sum(len(c['events']) for c in cohort)
    print(f'wrote {FIGDIR}/fig_swimmer.png  ({len(cohort)} enrollees, {ntot} infections, {ndet} detected)')


if __name__ == '__main__':
    main()
