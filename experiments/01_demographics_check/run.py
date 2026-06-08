"""
Exp 01 — Demographics Equilibrium Check

Runs demographics-only sims (no disease) for Bangladesh and Pakistan.
Snapshots age distribution at years 0-5, 7, 10 to check:
  1. How fast does the population equilibrate from the UK initial distribution?
  2. Is the <6m fraction at equilibrium consistent with the site birth rate?
  3. Does the calibration person-months denominator (steady-state snapshot at sim end)
     match the theoretically expected value?

Usage:
  uv run python experiments/01_demographics_check/run.py
"""

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import starsim as ss
import sciris as sc
from pathlib import Path

OUTDIR = Path(__file__).parent / 'outputs'
FIGDIR = Path(__file__).parent / 'figures'
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

# Match what the calibration uses
AGE_DATA = Path(__file__).parent.parent.parent / 'calibration' / 'uk_age_data.csv'
N_AGENTS = 20_000
SNAPSHOT_YEARS = [0, 1, 2, 3, 4, 5, 7, 10]

SITES = {
    'bangladesh': dict(birth_rate=19, death_rate=6),
    'pakistan':   dict(birth_rate=27, death_rate=7),
}

# MAL-ED bins (months) — same as calibration code
MALED_BINS = {'<6m': (0, 6), '6-12m': (6, 12), '12-24m': (12, 24), '24-36m': (24, 36)}


class AgeDemographicsAnalyzer(ss.Analyzer):
    """Snapshot age distribution at specified year offsets from sim start."""

    def __init__(self, snapshot_years, **kwargs):
        super().__init__(**kwargs)
        self.snapshot_years = list(snapshot_years)
        self.snapshots = []
        self._target_tis = {}
        self._recorded = set()

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        # Build map: timestep_index -> year_offset
        dt_years = self.dt.years  # freq object → float years
        for y in self.snapshot_years:
            ti = round(y / dt_years)
            self._target_tis[ti] = y

    def step(self):
        if self.ti in self._target_tis and self.ti not in self._recorded:
            self._recorded.add(self.ti)
            year_offset = self._target_tis[self.ti]
            ages = self.sim.people.age.values.copy()
            self.snapshots.append({
                'year_offset': year_offset,
                'n_alive': int(len(ages)),
                'ages': ages,
            })


def bin_fractions(ages_years):
    """Fraction of population in each MAL-ED age bin and selected wider bins."""
    bins = dict(**MALED_BINS, **{'3-5y': (36, 60), '5-10y': (60, 120), '10+y': (120, 99999)})
    total = len(ages_years)
    out = {}
    for label, (lo_m, hi_m) in bins.items():
        lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
        count = int(((ages_years >= lo_y) & (ages_years < hi_y)).sum())
        out[label] = count / total if total > 0 else 0.0
    return out


def theory_lt6m(birth_rate_per1000):
    """Expected fraction of population in the <6m bin at stable demographic equilibrium.
    Approximation: births per person per year × 0.5 years in the bin.
    Ignores mortality over the 6-month window (small for these rates).
    """
    return (birth_rate_per1000 / 1000.0) * 0.5


def run_site(site, birth_rate, death_rate):
    t0 = sc.tic()
    analyzer = AgeDemographicsAnalyzer(snapshot_years=SNAPSHOT_YEARS)
    sim = ss.Sim(
        n_agents=N_AGENTS,
        start='2003-01-01',
        stop='2013-01-01',
        dt=ss.days(1),
        people=ss.People(n_agents=N_AGENTS, age_data=str(AGE_DATA)),
        demographics=[
            ss.Births(birth_rate=ss.peryear(birth_rate)),
            ss.Deaths(death_rate=ss.peryear(death_rate)),
        ],
        analyzers=[analyzer],
        verbose=False,
    )
    sim.run()
    elapsed = sc.toc(t0, output=True)
    print(f'  {site}: {elapsed:.1f}s, n_agents final={sim.people.n_agents}')
    # Starsim deep-copies modules on init — read snapshots from the live sim object.
    return sim.analyzers['agedemographicsanalyzer'].snapshots


def main():
    all_records = []

    for site, demo in SITES.items():
        print(f'\n=== {site.upper()} (birth={demo["birth_rate"]}, death={demo["death_rate"]} per 1000/yr) ===')
        theory = theory_lt6m(demo['birth_rate'])
        print(f'  Theoretical equilibrium <6m fraction: {theory*100:.3f}%')

        snapshots = run_site(site, **demo)

        print(f'  {"Year":>4}  {"n_alive":>8}  {"<6m%":>7}  {"ratio":>6}  {"<12m%":>7}  {"<36m%":>7}')
        for snap in snapshots:
            ages = snap['ages']
            fracs = bin_fractions(ages)
            lt6m  = fracs['<6m']
            lt12m = fracs['<6m'] + fracs['6-12m']
            lt36m = lt12m + fracs['12-24m'] + fracs['24-36m']
            ratio = lt6m / theory if theory > 0 else float('nan')
            print(f'  {snap["year_offset"]:>4}  {snap["n_alive"]:>8}  '
                  f'{lt6m*100:>6.3f}%  {ratio:>6.2f}  {lt12m*100:>6.3f}%  {lt36m*100:>6.3f}%')

            # Write one JSONL record per snapshot
            rec = {
                'site': site,
                'year_offset': snap['year_offset'],
                'n_alive': snap['n_alive'],
                'theory_lt6m_pct': round(theory * 100, 4),
                'ratio_to_theory': round(ratio, 4),
            }
            rec.update({f'frac_{k}': round(v, 6) for k, v in bin_fractions(ages).items()})
            all_records.append(rec)

        # Person-months denominator check: simulate what the calibration code does
        # (snapshot age dist at sim end × calibration window in months = 5 years × 12)
        final_snap = [s for s in snapshots if s['year_offset'] == 10][0]
        ages_final = final_snap['ages']
        cal_window_months = 5.0 * 12.0  # years 5-10
        for bin_label, (lo_m, hi_m) in MALED_BINS.items():
            lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
            count = int(((ages_final >= lo_y) & (ages_final < hi_y)).sum())
            model_pt = count * cal_window_months
            # Theoretical PT: theory fraction × N_AGENTS × cal_window_months
            theory_frac = (demo['birth_rate'] / 1000.0) * (hi_m - lo_m) / 12.0
            theory_pt   = theory_frac * N_AGENTS * cal_window_months
            print(f'  PT denominator {bin_label}: model={model_pt:.0f} pm  theory≈{theory_pt:.0f} pm  '
                  f'ratio={model_pt/theory_pt:.2f}' if theory_pt > 0 else '')

    # Save outputs
    with open(OUTDIR / 'results.jsonl', 'w') as f:
        for rec in all_records:
            f.write(json.dumps(rec) + '\n')
    pd.DataFrame(all_records).to_csv(OUTDIR / 'demographics_trajectory.csv', index=False)
    print(f'\nSaved outputs/demographics_trajectory.csv and results.jsonl')

    # --- Figure: age-distribution trajectory ---
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=False)

    for ax, (site, demo) in zip(axes, SITES.items()):
        site_recs = [r for r in all_records if r['site'] == site]
        theory = theory_lt6m(demo['birth_rate'])

        years  = [r['year_offset'] for r in site_recs]
        lt6m   = [r['frac_<6m']    * 100 for r in site_recs]
        lt12m  = [(r['frac_<6m'] + r['frac_6-12m']) * 100 for r in site_recs]
        lt36m  = [(r['frac_<6m'] + r['frac_6-12m'] + r['frac_12-24m'] + r['frac_24-36m']) * 100
                  for r in site_recs]

        ax.plot(years, lt6m,  'o-', color='tab:blue',   label='<6 months (model)')
        ax.plot(years, lt12m, 's--', color='tab:orange', label='<12 months (model)')
        ax.plot(years, lt36m, '^:',  color='tab:green',  label='<36 months (model)')
        ax.axhline(theory * 100, color='tab:blue', linestyle=':', alpha=0.5,
                   label=f'<6m equilibrium ({theory*100:.2f}%)')
        ax.axvspan(5, 10, alpha=0.07, color='green', label='Calibration window')
        ax.axvline(5, color='gray', linestyle='--', alpha=0.5)

        ax.set_xlabel('Simulation year')
        ax.set_ylabel('% of population in age group')
        ax.set_title(f'{site.title()}\nbirth={demo["birth_rate"]}, death={demo["death_rate"]} per 1000/yr')
        ax.legend(fontsize=8, loc='upper right')
        ax.set_xticks(years)
        ax.grid(True, alpha=0.25)

    plt.suptitle('Age Distribution Trajectory — Initialized from UK Age Data', fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGDIR / 'age_distribution_trajectory.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/age_distribution_trajectory.png')


if __name__ == '__main__':
    main()
