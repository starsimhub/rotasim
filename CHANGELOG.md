# Changelog

## Version 3.0.0 (2026-10-30)

Rotasim now runs on Starsim 4, and includes the model developed for calibration to the MAL-ED study and to surveillance data.

### Highlights

- Runs on Starsim 4. Sims are about 2.5 times faster for the single-strain calibration setup.
- New model features: an asymptomatic phase with reduced shedding, maternal immunity (by age or from antibody titers), protection that builds up with the number of infections, and an age-assortative network.
- New observation models: `MALEDTargets`, `MALEDCohort`, and `Surveillance`, with `SymptomModel` for the probability of symptoms.
- New documentation: a [getting started guide](docs/getting_started.md) and a [model description](docs/model.md).

### Bug fixes

- Protection against one strain no longer carries over into the calculation for the next, which favored whichever strain came first. Results with several strains change substantially: in the `simple` scenario, strains that used to differ tenfold now have about equal numbers of infections, as they should with equal fitness.
- `rs.Sim()` with the default scenario, and every other scenario except `simple` and `single`, no longer fails.
- Scenarios' `default_fitness` is used again, so reassortants no longer all have a fitness of 1.
- `override_fitness` can now set the fitness of a reassortant, rather than silently ignoring it.
- With `use_preferred_partners=True`, a scenario's own strains are always kept.
- `RotaVaccination` runs again; it failed with recent versions of Starsim.
- `RotaVaccination` offers each dose once. Before, anyone who had not had a dose was offered one every day, so 98% of infants were vaccinated even at an uptake of 20%.
- `RotaVaccination` no longer makes the susceptibility of the unvaccinated undefined.
- `AgeAssortativeNet` uses `child_age_threshold`, which an error in operator precedence had made about 1 year whatever its value. Children also now get their full `n_contacts`.
- Waning uses the right units with a timestep other than one day.
- Random numbers now all come from Starsim, so results depend only on the sim's seed.

### Changes

- Each infection now lasts half a day longer on average, because of how Starsim 4 schedules events. That is about a tenth of the symptomatic phase, so transmission is higher for the same parameters: over 10 seeds, prevalence in a single-strain endemic setup rose from 0.224 to 0.247, and the median age at first infection fell by about 12%. Calibrated values of `beta` will need refitting.
- `generate_gp_reassortments()` returns strains in sorted order, and scenarios use `init_prevalence` throughout.
- A sim runs for 10 years by default, and agents age even with no demographics.
- With no `age_data`, agents take the world's age distribution, which is Starsim 4's default, rather than ages uniform from 0 to 60.
- `StrainStats` results are named like `G1P8_count`, and it no longer has its own `to_df()`: use its results, or `sim.to_df()`.
- `InfectedStrainStats` gets its infections from each strain's `ti_infected`, so it no longer misses anyone reinfected in the step they recovered.
- Analyzers that took `seed=` no longer do, and use the sim's random numbers.
- `Sim` no longer prints as it is set up.

### Removed

- `rs.Aging`: Starsim ages agents. It added days to ages in years.
- `RotaVax` and `StrainStatistics`, which were aliases.
- `get_strain_summary()`, `print_strain_summary()`, `print_vaccination_summary()`, `get_infection_summary()`, and `Sim.get_connector_by_type()`.
- Parameters that had no effect: `cotransmission_prob`, the `long_term_immunity_*` parameters, and `immunity_init_dist`; and the `min_exposures` and `max_exposures` arguments of `initialize_immunity()`.
- The heatmap of `UidTracker` by age. It now shows one row per agent.
- Scripts that read the output files of Rotasim 1 (`data_analysis/`, `scripts/`), and `examples/`.

### Calibration

- The `calibration/` folder has the experiments from the `dec_calibration_akraay` branch. They were run on Starsim 3 and have not been updated, so they are a record of that work rather than scripts that run on this version.
