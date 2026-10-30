# Rotasim

Rotasim is an agent-based model of rotavirus transmission, built on [Starsim](https://github.com/starsimhub/starsim). It tracks several strains at once, each a combination of G and P genotypes, so it can be used to study how strains compete, how new ones arise by reassortment, and how vaccines with different antigens change both.

## Main features

- Strains as G,P combinations, with co-infection and reassortment
- Immunity that depends on infection history, how closely a strain matches earlier ones, and time since recovery
- Maternal immunity that wanes with age, either by age alone or from each infant's antibody titer
- Symptomatic and asymptomatic phases, with reduced shedding in the latter
- Multi-dose vaccination with protection specific to G and P
- Analyzers for strains, ages, and infection events, and models of what cohort studies and surveillance observe

## Installation

Rotasim needs Python 3.11 or later and Starsim 4. While Starsim 4 is in development, install it from its `v4` branch:

```bash
pip install git+https://github.com/starsimhub/starsim@v4
pip install -e .
```

## Quick start

```python
import rotasim as rs

sim = rs.Sim(scenario='simple', n_agents=2000, start='2000-01-01', stop='2001-01-01')
sim.run()
sim.plot()
```

This runs one year of two strains, G1P8 and G2P4. All four combinations of their genotypes are in the sim, and G1P4 and G2P8 appear through reassortment.

## Documentation

- [Getting started](docs/getting_started.md): how to set up and run scenarios, add demographics and vaccination, and look at results
- [Model description](docs/model.md): how the model works, and its parameters
- [Changelog](CHANGELOG.md)

## Tests

```bash
pip install -r tests/requirements.txt
cd tests
pytest -n auto
```

## License

MIT; see [LICENSE](LICENSE).
