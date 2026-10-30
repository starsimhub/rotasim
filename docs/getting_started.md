# Getting started

This page shows the main things you can do with Rotasim. The [model description](model.md) explains what the model does. Rotasim is built on Starsim, so everything in the [Starsim documentation](https://starsim.org/v4) applies too.

## Run a scenario

A scenario says which strains are present at the start. Each strain has a fitness, which scales how transmissible it is, and an initial prevalence.

```python
import rotasim as rs

rs.list_scenarios()  # The built-in scenarios, with descriptions

sim = rs.Sim(scenario='baseline', n_agents=2000, start='2000-01-01', stop='2002-01-01')
sim.run()
```

A sim has a daily timestep, and has a disease for every combination of the G and P genotypes in the scenario. The `baseline` scenario starts with G1P8, G2P4, and G3P8, which gives six diseases: the rest are reassortants, which start with no infections.

Each strain's results are under its name:

```python
g1p8 = sim.results.G1P8
g1p8.prevalence       # Share of people infected, over time
g1p8.new_infections   # New infections, over time
g1p8.cum_infections   # Cumulative infections, over time
```

## Change a scenario

To change the transmission rate, or the fitness or initial prevalence of the strains:

```python
sim = rs.Sim(
    scenario        = 'baseline',
    base_beta       = 0.2,                            # Transmission rate per day, at a fitness of 1
    override_fitness = {(1, 8): 0.95, (2, 4): 0.8},   # Fitness by strain
    n_agents        = 2000,
)
```

Or write a scenario of your own. `default_fitness` is the fitness of the strains that aren't listed, which here are G1P4 and G2P8:

```python
scenario = dict(
    strains = {
        (1, 8): dict(fitness=1.0, init_prevalence=0.015),
        (2, 4): dict(fitness=0.8, init_prevalence=0.010),
    },
    default_fitness = 0.5,
)
sim = rs.Sim(scenario=scenario, n_agents=2000)
```

## Add births and deaths

Without demographics, nobody new enters the population, so infection tends to die out. Age-related results also depend on a realistic population, so most analyses need births and deaths.

```python
import starsim as ss

sim = rs.Sim(
    scenario     = 'single',
    n_agents     = 2000,
    start        = '2000-01-01',
    stop         = '2002-01-01',
    networks     = ss.RandomNet(n_contacts=7),
    demographics = [ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(7))],
)
sim.run()
```

The network can be any Starsim network. `rs.AgeAssortativeNet` has children and adults mix preferentially with their own group.

## Change immunity

The immunity parameters belong to `rs.RotaImmunityConnector`. Giving `connectors` replaces the default connectors, so include `rs.RotaReassortmentConnector` as well if there is more than one strain:

```python
immunity = rs.RotaImmunityConnector(
    use_fixed_susceptibility   = True,    # Baseline protection set by the number of infections
    sus_after_1                = 0.7,     # Susceptibility after one infection...
    sus_after_2                = 0.5,     # ...two...
    sus_after_3plus            = 0.3,     # ...and three or more
    maternal_immunity_efficacy = 0.8,     # Protection at birth
)
reassortment = rs.RotaReassortmentConnector(reassortment_prob=0.01)
sim = rs.Sim(scenario='simple', connectors=[immunity, reassortment], n_agents=2000)
```

A sim starts with no one having had an infection. To start adults with the immunity they would have from earlier infections, call `initialize_immunity()` after initializing the sim:

```python
sim = rs.Sim(scenario='single', n_agents=2000, start='2000-01-01', stop='2001-01-01')
sim.init()
sim.connectors.rotaimmunityconnector.initialize_immunity(min_age=5, max_age=100)
sim.run()
```

## Add vaccination

`rs.RotaVaccination` vaccinates infants in an age range, by default with two doses at an uptake of 80% each. Vaccine antigens are given separately for G and P:

```python
vaccine = rs.RotaVaccination(start_date='2001-01-01', G_antigens=[1], P_antigens=[8], uptake_dist=0.9)

scenarios = {
    'No vaccine': None,
    'Vaccine': vaccine,
}
sims = {}
for label, intervention in scenarios.items():
    sims[label] = rs.Sim(
        scenario      = 'baseline',
        n_agents      = 2000,
        start         = '2000-01-01',
        stop          = '2003-01-01',
        demographics  = [ss.Births(birth_rate=ss.peryear(20)), ss.Deaths(death_rate=ss.peryear(7))],
        interventions = intervention,
    ).run()

summary = sims['Vaccine'].interventions.rotavaccination.get_vaccination_summary()
```

## Look at results

Analyzers collect more detail, as results or, for some, a dataframe from `to_df()`:

```python
analyzers = [rs.StrainStats(), rs.AgeStats(), rs.InfectedStrainStats()]
sim = rs.Sim(scenario='simple', n_agents=2000, start='2000-01-01', stop='2001-01-01', analyzers=analyzers)
sim.run()

strains = sim.analyzers.strainstats.results
strains.G1P8_proportion  # G1P8's share of infections, over time
infections = sim.analyzers.infectedstrainstats.to_df()  # One row per infection, with age and strain
```

| Analyzer | What it gives |
|---|---|
| `StrainStats` | People infected with each strain, and its share of infections |
| `EventStats` | Births, deaths, recoveries, infections, reassortments, and co-infections at each step |
| `AgeStats` | People in each age group |
| `InfectedStrainStats` | A log of every infection |
| `PersonTimeByAge` | Person-time by age, as the denominator of incidence rates |
| `UidTracker` | States of chosen agents over time |
| `MALEDTargets`, `MALEDCohort` | What the MAL-ED birth cohort study would observe |
| `Surveillance` | The age distribution of symptomatic cases, as surveillance would see it |

The last three turn infections into symptomatic cases with `rs.SymptomModel`, and their outputs come from `results_dict()`.

## Run several sims

Starsim's tools for multiple runs work as usual. This runs a sim with five different seeds, and shows the median and range of the results:

```python
msim = ss.parallel(rs.Sim(scenario='single', n_agents=2000, start='2000-01-01', stop='2001-01-01'), n_runs=5, parallel=False)
msim.plot()
```
