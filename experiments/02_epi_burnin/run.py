"""
Exp 02 — Epidemiological Burn-in Check

Runs the full disease model (mirroring the MAL-ED calibration worker setup)
for the full 10-year span and tracks, over time:
  - overall prevalence
  - number of co-circulating strains
  - population immunity: fraction with any immunity, mean lifetime exposures,
    mean susceptibility (1 - protection)
  - age-specific prevalence (MAL-ED bins) -- is the age gradient stationary?

Question: is the transmission system at endemic steady state by year 5 (when
the calibration window opens), or still in a transient?

Usage:
  uv run python experiments/02_epi_burnin/run.py
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import starsim as ss
import sciris as sc
import rotasim as rs
from pathlib import Path

HERE = Path(__file__).parent
OUTDIR = HERE / 'outputs'
FIGDIR = HERE / 'figures'
OUTDIR.mkdir(exist_ok=True)
FIGDIR.mkdir(exist_ok=True)

REPO = HERE.parent.parent
AGE_DATA = REPO / 'calibration' / 'uk_age_data.csv'
PARAM_JSON = REPO / 'calibration' / 'calibration_maled_bangladesh_worker093205.json'

N_AGENTS = 20_000
CAL_WINDOW_START = 5.0  # years
RECORD_EVERY = 7        # record every ~week of sim steps to keep series light
DEMO = dict(birth_rate=19, death_rate=6)  # Bangladesh

MALED_BINS = {'<6 m': (0, 6), '6-11 m': (6, 12), '12-23 m': (12, 24), '24-35 m': (24, 36)}


class BurnInAnalyzer(ss.Analyzer):
    """Record endemic-state summaries over time to assess stationarity."""

    def __init__(self, record_every=7, **kwargs):
        super().__init__(**kwargs)
        self.record_every = record_every
        self.rows = []
        self._ic = None

    def init_pre(self, sim, force=False):
        super().init_pre(sim, force)
        self._dt_years = self.dt.years

    def step(self):
        if self.ti % self.record_every != 0:
            return
        sim = self.sim
        ic = sim.connectors.rotaimmunityconnector
        ages = sim.people.age.values
        alive = sim.people.alive.values
        n_alive = int(alive.sum())
        if n_alive == 0:
            return

        # Infection status across all diseases (any strain) + per-strain counts.
        any_infected = np.zeros(len(ages), dtype=bool)
        n_strains = 0
        for disease in sim.diseases.values():
            inf = disease.infected.values
            if inf.sum() > 0:
                n_strains += 1
            any_infected |= inf
        # Restrict to alive
        any_infected &= alive

        # Mean susceptibility across diseases (lower = more immune).
        sus_vals = [d.rel_sus.values[alive] for d in sim.diseases.values()]
        mean_sus = float(np.mean([s.mean() for s in sus_vals])) if sus_vals else np.nan

        # Immunity summaries from the connector.
        has_imm = ic.has_immunity.values[alive]
        n_exposures = ic.num_current_infections.values[alive]

        row = {
            'year': round(self.ti * self._dt_years, 4),
            'n_alive': n_alive,
            'prevalence': float(any_infected.sum() / n_alive),
            'n_strains': n_strains,
            'frac_any_immunity': float(has_imm.mean()),
            'mean_lifetime_exposures': float(n_exposures.mean()),
            'mean_susceptibility': mean_sus,
        }

        # Age-specific prevalence (MAL-ED bins).
        for label, (lo_m, hi_m) in MALED_BINS.items():
            lo_y, hi_y = lo_m / 12.0, hi_m / 12.0
            in_bin = (ages >= lo_y) & (ages < hi_y) & alive
            n_bin = int(in_bin.sum())
            row[f'prev_{label}'] = float((any_infected & in_bin).sum() / n_bin) if n_bin > 0 else np.nan
        self.rows.append(row)


def build_sim(params, analyzers):
    """Mirror the MAL-ED calibration worker sim setup (Bangladesh)."""
    immunity_connector = rs.RotaImmunityConnector(use_fixed_susceptibility=False)
    people = ss.People(n_agents=N_AGENTS, age_data=str(AGE_DATA))
    sim = rs.Sim(
        n_agents=N_AGENTS,
        start='2003-01-01', stop='2013-01-01',
        dt=ss.days(1),
        verbose=False,
        scenario='single',
        people=people,
        analyzers=analyzers,
        networks=ss.RandomNet(n_contacts=7),
        demographics=[
            ss.Births(birth_rate=ss.peryear(DEMO['birth_rate'])),
            ss.Deaths(death_rate=ss.peryear(DEMO['death_rate'])),
        ],
        connectors=[immunity_connector],
        rand_seed=1,
    )
    sim.pars.base_beta = params['base_beta']
    for disease in sim.pars.diseases:
        if isinstance(disease, rs.Rotavirus):
            disease.pars.beta = ss.perday(sim.pars.base_beta * disease.pars.fitness)
    sim.init()
    ic = sim.connectors.rotaimmunityconnector
    ic.pars['use_fixed_susceptibility'] = True
    ic.pars['sus_after_1']     = params['sus_after_1']
    ic.pars['sus_after_2']     = params['sus_after_2']
    ic.pars['sus_after_3plus'] = params['sus_after_3plus']
    ic.pars['maternal_immunity_efficacy']  = params.get('maternal_immunity_efficacy', 0.0)
    ic.pars['maternal_immunity_half_life'] = ss.days(params.get('maternal_immunity_half_life_days', 90.0))
    ic.initialize_immunity(min_age=18, max_age=125, min_exposures=5, max_exposures=15)
    return sim


def main():
    params = json.loads(PARAM_JSON.read_text())['best_params']
    print(f'Using params from {PARAM_JSON.name}:')
    for k, v in params.items():
        print(f'  {k}: {v:.4f}')

    t0 = sc.tic()
    analyzer = BurnInAnalyzer(record_every=RECORD_EVERY)
    sim = build_sim(params, analyzers=[analyzer])
    sim.run()
    print(f'\nRun: {sc.toc(t0, output=True):.1f}s')

    rows = sim.analyzers['burninanalyzer'].rows
    with open(OUTDIR / 'burnin_timeseries.jsonl', 'w') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    print(f'Saved outputs/burnin_timeseries.jsonl ({len(rows)} records)')

    years = np.array([r['year'] for r in rows])

    # --- Figure: stationarity panels ---
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))

    ax = axes[0, 0]
    ax.plot(years, [r['prevalence'] * 100 for r in rows], color='tab:red', lw=1)
    ax.set_title('Overall prevalence')
    ax.set_ylabel('% infected')

    ax = axes[0, 1]
    ax.plot(years, [r['frac_any_immunity'] * 100 for r in rows], color='tab:blue', lw=1,
            label='% with any immunity')
    ax.plot(years, [r['mean_susceptibility'] * 100 for r in rows], color='tab:green', lw=1,
            label='mean susceptibility %')
    ax.set_title('Population immunity')
    ax.set_ylabel('%')
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.plot(years, [r['mean_lifetime_exposures'] for r in rows], color='tab:purple', lw=1)
    ax.set_title('Mean lifetime exposures per (living) agent')
    ax.set_ylabel('infections')
    ax.set_xlabel('Simulation year')

    ax = axes[1, 1]
    for label in MALED_BINS:
        ax.plot(years, [r[f'prev_{label}'] * 100 for r in rows], lw=1, label=label)
    ax.set_title('Age-specific prevalence (MAL-ED bins)')
    ax.set_ylabel('% infected in bin')
    ax.set_xlabel('Simulation year')
    ax.legend(fontsize=8)

    for ax in axes.flat:
        ax.axvline(CAL_WINDOW_START, color='gray', linestyle='--', alpha=0.6)
        ax.axvspan(CAL_WINDOW_START, 10, alpha=0.06, color='green')
        ax.grid(True, alpha=0.25)

    plt.suptitle('Exp 02 — Epidemiological Burn-in (Bangladesh, dashed = calibration window start)',
                 fontsize=12)
    plt.tight_layout()
    fig.savefig(FIGDIR / 'burnin_timeseries.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('Saved figures/burnin_timeseries.png')

    # --- Quick stationarity numeric: compare mean over yrs 4-5 vs yrs 9-10 ---
    def window_mean(key, lo, hi):
        vals = [r[key] for r in rows if lo <= r['year'] < hi]
        return float(np.mean(vals)) if vals else float('nan')

    print('\nStationarity check (pre-window yr 4-5 vs end yr 9-10):')
    for key in ['prevalence', 'frac_any_immunity', 'mean_susceptibility', 'mean_lifetime_exposures']:
        early = window_mean(key, 4, 5)
        late  = window_mean(key, 9, 10)
        drift = (late - early) / early * 100 if early not in (0, float('nan')) else float('nan')
        print(f'  {key:>26}: yr4-5={early:.4f}  yr9-10={late:.4f}  drift={drift:+.1f}%')


if __name__ == '__main__':
    main()
