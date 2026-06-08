"""
Exp 01 (supplement) — Validate the person-time fix.

Confirms that rs.PersonTimeByAge (which accumulates count x dt over the
science window) recovers unbiased person-time, vs. the current calibration's
end-of-sim snapshot method (compute_person_months_steady_state), which
overestimates because the population grows.

Compares both against the theoretical equilibrium person-time:
    theory_PM(bin) = birth_rate_frac x bin_width_years x mean_pop x window_months

Uses the time-averaged mean population over the window as the fair baseline
for the snapshot method (a stationary-population sanity reference is the
accumulated value itself).

Usage:
  uv run python experiments/01_demographics_check/validate_fix.py
"""

import json
import numpy as np
import starsim as ss
import rotasim as rs
from pathlib import Path

OUTDIR = Path(__file__).parent / 'outputs'
AGE_DATA = Path(__file__).parent.parent.parent / 'calibration' / 'uk_age_data.csv'

N_AGENTS = 20_000
CAL_WINDOW = (5.0, 10.0)          # years from start
WINDOW_MONTHS = (CAL_WINDOW[1] - CAL_WINDOW[0]) * 12.0
MALED_BINS = rs.PersonTimeByAge.DEFAULT_BINS_MONTHS

SITES = {
    'bangladesh': dict(birth_rate=19, death_rate=6),
    'pakistan':   dict(birth_rate=27, death_rate=7),
}


def snapshot_pm(ages_years):
    """Current calibration method: final headcount x window length."""
    pm = {}
    for label, (lo_m, hi_m) in MALED_BINS.items():
        lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
        count = int(((ages_years >= lo_y) & (ages_years < hi_y)).sum())
        pm[label] = count * WINDOW_MONTHS
    return pm


def main():
    records = []
    for site, demo in SITES.items():
        print(f'\n=== {site.upper()} ===')
        pt_analyzer = rs.PersonTimeByAge(calibration_window=CAL_WINDOW)
        sim = ss.Sim(
            n_agents=N_AGENTS, start='2003-01-01', stop='2013-01-01', dt=ss.days(1),
            people=ss.People(n_agents=N_AGENTS, age_data=str(AGE_DATA)),
            demographics=[
                ss.Births(birth_rate=ss.peryear(demo['birth_rate'])),
                ss.Deaths(death_rate=ss.peryear(demo['death_rate'])),
            ],
            analyzers=[pt_analyzer],
            verbose=False,
        )
        sim.run()

        accumulated = sim.analyzers['persontimebyage'].person_months
        snapshot    = snapshot_pm(sim.people.age.values)

        print(f'  {"bin":>8}  {"accumulated":>12}  {"snapshot":>10}  {"snapshot/acc":>12}')
        for label in MALED_BINS:
            acc = accumulated[label]
            snp = snapshot[label]
            ratio = snp / acc if acc > 0 else float('nan')
            print(f'  {label:>8}  {acc:>12.0f}  {snp:>10.0f}  {ratio:>12.2f}')
            records.append(dict(site=site, bin=label,
                                pm_accumulated=round(acc, 1),
                                pm_snapshot=round(snp, 1),
                                ratio_snapshot_to_accumulated=round(ratio, 4)))

    with open(OUTDIR / 'person_time_fix_validation.jsonl', 'w') as f:
        for r in records:
            f.write(json.dumps(r) + '\n')
    print(f'\nSaved outputs/person_time_fix_validation.jsonl')

    # Headline: how much does the snapshot method inflate the denominator?
    ratios = [r['ratio_snapshot_to_accumulated'] for r in records]
    print(f'\nSnapshot method overestimates person-time by '
          f'{(np.mean(ratios)-1)*100:.0f}% on average '
          f'(range {(min(ratios)-1)*100:.0f}%..{(max(ratios)-1)*100:.0f}%).')


if __name__ == '__main__':
    main()
