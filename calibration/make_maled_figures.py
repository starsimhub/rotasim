"""
Make two MAL-ED Bangladesh calibration figures from existing calibration logs
(NO sims re-run). Both figures are built purely by parsing the per-trial lines
that calibrate_maled.py already wrote to its log files:

  - "Median model IR (per 100 PM): <6 m=.., 6-11 m=.., 12-23 m=.., 24-35 m=.."
  - "median total = T (inc=I, first_inf=F)"

Figure 1 (incidence_by_age_bangladesh.png):
  Grouped bars of symptomatic incidence by age bin: MAL-ED data vs the
  best symptomatic-IR-fit trial vs the best first-infection-fit trial.
  (No error bars per request.)

Figure 2 (pareto_frontier_bangladesh.png):
  Scatter of every completed trial in (GOF_incidence, GOF_first_infection)
  space, colored by which target it was calibrated against, with the
  non-dominated (Pareto) frontier drawn. Shows the two objectives trade off.

Only the two code-consistent studies (both run with the detection model,
p_asymp_detect=0.40) are used: worker103016 (symptomatic_ir) and
worker103019 (first_infection).
"""
import re
import argparse

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

BINS = ['<6 m', '6-11 m', '12-23 m', '24-35 m']

# Best trial in each study (from study.best_trial).
BEST_SYMP_IR = 19
BEST_FIRST   = 17

_RE_TARGET = re.compile(r'(<6 m|6-11 m|12-23 m|24-35 m)\s+cases=\s*\d+\s+PT=\s*\d+\s+IR=([\d.]+)')
_RE_TRIAL  = re.compile(r'Trial (\d+):\s*$')
_RE_GOF    = re.compile(r'median total = [\d.]+ \(inc=([\d.]+), first_inf=([\d.]+)\)')
_RE_MODEL_IR = re.compile(
    r'Median model IR \(per 100 PM\): '
    r'<6 m=([\d.]+), 6-11 m=([\d.]+), 12-23 m=([\d.]+), 24-35 m=([\d.]+)')


def parse_log(path):
    """Return (target_ir dict, trials dict).

    target_ir: {bin: IR}
    trials: {trial_num: {'ir': {bin: IR}, 'gof_inc': float, 'gof_first': float}}
    """
    target_ir = {}
    trials = {}
    cur = None
    with open(path) as f:
        for line in f:
            mt = _RE_TARGET.search(line)
            if mt and mt.group(1) not in target_ir:
                target_ir[mt.group(1)] = float(mt.group(2))
                continue
            mtr = _RE_TRIAL.search(line)
            if mtr:
                cur = int(mtr.group(1))
                trials.setdefault(cur, {})
                continue
            mg = _RE_GOF.search(line)
            if mg and cur is not None:
                trials[cur]['gof_inc'] = float(mg.group(1))
                trials[cur]['gof_first'] = float(mg.group(2))
                continue
            mi = _RE_MODEL_IR.search(line)
            if mi and cur is not None:
                trials[cur]['ir'] = dict(zip(BINS, map(float, mi.groups())))
    return target_ir, trials


def figure_incidence(target_ir, symp_trials, first_trials, out):
    fig, ax = plt.subplots(figsize=(9, 5.5))
    x = np.arange(len(BINS))
    w = 0.27

    tgt = [target_ir[b] for b in BINS]
    symp = [symp_trials[BEST_SYMP_IR]['ir'][b] for b in BINS]
    frst = [first_trials[BEST_FIRST]['ir'][b] for b in BINS]

    ax.bar(x - w, tgt,  w, label='MAL-ED data', color='#444444')
    ax.bar(x,     symp, w, label=f'Symptomatic-IR fit (trial #{BEST_SYMP_IR})',
           color='#1f77b4')
    ax.bar(x + w, frst, w, label=f'First-infection fit (trial #{BEST_FIRST})',
           color='#d62728')

    for xi, vals in zip(x, zip(tgt, symp, frst)):
        for dx, v in zip((-w, 0, w), vals):
            ax.text(xi + dx, v + 0.15, f'{v:.1f}', ha='center', va='bottom',
                    fontsize=7.5)

    ax.set_xticks(x)
    ax.set_xticklabels(BINS)
    ax.set_xlabel('Age bin')
    ax.set_ylabel('Symptomatic incidence (per 100 person-months)')
    ax.set_title('Bangladesh MAL-ED: symptomatic incidence by age\n'
                 'data vs. model fits to the two separate targets')
    ax.legend(frameon=False)
    ax.spines[['top', 'right']].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


def pareto_front(points):
    """points: list of (inc, first). Return indices on the non-dominated front
    (minimizing both), sorted by inc ascending."""
    order = sorted(range(len(points)), key=lambda i: (points[i][0], points[i][1]))
    front, best_first = [], np.inf
    for i in order:
        if points[i][1] < best_first - 1e-12:
            front.append(i)
            best_first = points[i][1]
    return front


def figure_pareto(symp_trials, first_trials, out):
    fig, ax = plt.subplots(figsize=(8, 6))

    series = [
        ('symptomatic_ir study', symp_trials, '#1f77b4', BEST_SYMP_IR),
        ('first_infection study', first_trials, '#d62728', BEST_FIRST),
    ]
    all_pts, all_meta = [], []
    for label, trials, color, best in series:
        pts = [(t['gof_inc'], t['gof_first']) for n, t in sorted(trials.items())
               if 'gof_inc' in t and 'gof_first' in t]
        nums = [n for n, t in sorted(trials.items())
                if 'gof_inc' in t and 'gof_first' in t]
        xs = [p[0] for p in pts]
        ys = [max(p[1], 1e-3) for p in pts]  # floor for log axis
        ax.scatter(xs, ys, s=28, color=color, alpha=0.65, label=label,
                   edgecolors='none')
        for (x, y), n in zip(zip(xs, ys), nums):
            if n == best:
                ax.scatter([x], [y], s=180, marker='*', color=color,
                           edgecolors='k', zorder=5)
                ax.annotate(f'#{n}', (x, y), textcoords='offset points',
                            xytext=(8, 6), fontsize=9, fontweight='bold')
        all_pts.extend(pts)
        all_meta.extend(nums)

    # Pareto frontier across the pooled trials.
    fronti = pareto_front(all_pts)
    fx = [all_pts[i][0] for i in fronti]
    fy = [max(all_pts[i][1], 1e-3) for i in fronti]
    ax.plot(fx, fy, 'k--', lw=1.3, label='Pareto frontier', zorder=4)

    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlabel(r'GOF$_{\mathrm{incidence}}$  (sum sq. log-IR diff over age bins)')
    ax.set_ylabel(r'GOF$_{\mathrm{first\ infection}}$  (normalised quartile diff)')
    ax.set_title('Bangladesh MAL-ED: the two calibration targets trade off\n'
                 '(each point = one completed trial; lower-left is better)')
    ax.legend(frameon=False, loc='upper right')
    ax.grid(True, which='both', ls=':', alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'wrote {out}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--symp-log', default='/tmp/calibrate_maled_bangladesh_worker103016_20260602_103016.log')
    p.add_argument('--first-log', default='/tmp/calibrate_maled_bangladesh_worker103019_20260602_103019.log')
    p.add_argument('--outdir', default='.')
    args = p.parse_args()

    target_ir, symp_trials = parse_log(args.symp_log)
    target_ir2, first_trials = parse_log(args.first_log)
    # Targets are identical across both logs; sanity check.
    assert all(abs(target_ir[b] - target_ir2[b]) < 1e-6 for b in BINS), \
        'target IR differs between logs?'

    print('Target IR by age:', {b: target_ir[b] for b in BINS})
    print(f'symptomatic_ir trials parsed: {len(symp_trials)}; '
          f'first_infection trials parsed: {len(first_trials)}')

    figure_incidence(target_ir, symp_trials, first_trials,
                     f'{args.outdir}/incidence_by_age_bangladesh.png')
    figure_pareto(symp_trials, first_trials,
                  f'{args.outdir}/pareto_frontier_bangladesh.png')


if __name__ == '__main__':
    main()
