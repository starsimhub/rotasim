# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project overview

Rotasim is an agent-based model of rotavirus transmission and strain diversity, built on Starsim 4. Each strain (a G,P combination) is a `Rotavirus` disease of its own, and connectors handle what happens between strains. `docs/model.md` describes the model; `docs/getting_started.md` shows how to use it.

## Commands

```bash
pip install -e .                    # Install
cd tests && pytest -n auto          # Run the tests, which include every code block in the docs
```

## Structure

- `rotasim/rotasim.py`: `Sim`, which builds the diseases from a scenario
- `rotasim/utils.py`: scenarios, and generating G,P combinations
- `rotasim/rotavirus.py`: `Rotavirus`, the natural history of one strain (an `ss.Infection`, written as code)
- `rotasim/immunity.py`: `RotaImmunityConnector`, which sets each strain's `rel_sus` every step
- `rotasim/reassortment.py`: `RotaReassortmentConnector`
- `rotasim/interventions.py`: `RotaVaccination`
- `rotasim/age_networks.py`: `AgeAssortativeNet`
- `rotasim/analyzers.py`: general analyzers
- `rotasim/surveillance.py`: observation models, and `SymptomModel`
- `planning/v4_port.md`: the work plan for the move to Starsim 4
- `starsim_issues.md`: what Starsim could do better for this model

## Conventions

- Write the model as Starsim does: `define_pars()`, `define_states()`, and `update_pars()` in `__init__()`; `ss.Dist` objects for all random numbers (never `np.random`); `ss.TimeArr` with `due()` for scheduled events, set at `self.ti + 1 + dur` for an infection.
- Durations are given with units (`ss.days(13)`) and drawn in steps, so the model works at any timestep.
- Keep per-step code vectorized.
- In tests, follow the Starsim style: short test names, related checks together with commented asserts, `@sc.timer()`, and a `__main__` block.
- Headings use sentence case, and Markdown isn't hard-wrapped.

## Calibration

The calibration experiments in `calibration/` came from `dec_calibration_akraay`, and were run on Starsim 3. They are not tested and have not been updated; to run one on this code, see the Removed and Changes sections of `CHANGELOG.md`.
