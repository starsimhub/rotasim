# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

RotaABM is an agent-based model for simulating rotavirus genetic diversity and vaccination interventions. The project is built on top of the `starsim` framework and uses genetic modeling to track pathogen evolution and immune responses.

**Current Status**: Undergoing major v2 architecture refactoring on `rotasim_architecture` branch to follow traditional Starsim design patterns.

## V2 Architecture (In Development)

**Key Design Change**: Transition from monolithic `Rota(ss.Module)` to individual `Rotavirus(ss.Infection)` instances per strain, with cross-strain interactions handled by `ss.Connector` classes.

### Core V2 Components
- **Rotavirus(ss.Infection)**: Individual strain-specific disease class with G,P as first-class parameters
- **RotaImmunityConnector(ss.Connector)**: High-performance vectorized cross-strain immunity using bitmasks
- **RotaReassortment(ss.Connector)**: Genetic reassortment mechanism (pending dynamic strain creation support)
- **Multiple Analyzers**: Maintain compatibility with existing CSV output formats

### V2 Usage Pattern
```python
# G,P as first-class parameters, backbone for non-antigenic segments
rota_g1p8 = Rotavirus(G=1, P=8, backbone=(1,1))  # Auto-named "G1P8_11"
rota_g2p4 = Rotavirus(G=2, P=4, backbone=(1,1))  # Auto-named "G2P4_11"

# Connectors auto-detect Rotavirus instances
immunity = RotaImmunityConnector()
reassortment = RotaReassortment()

sim = ss.Sim(
    diseases=[rota_g1p8, rota_g2p4],
    connectors=[immunity, reassortment],
    networks='random'
)
```

## Development Commands

### Installation
```bash
pip install -e .
```
This installs RotaABM as an importable module in development mode.

### Running Tests
```bash
python -m pytest tests/
```
Run individual test files:
```bash
python -m pytest tests/test_rotasim.py
python -m pytest tests/test_analyzers.py
python -m pytest tests/test_calibration.py
```

### Running Simple Simulation
```bash
cd tests
python simple.py
```
This writes results to a `results` folder with file names based on parameter values.

### Plotting Results
After running simulations:
```bash
python data_analysis/plot_results.py
```

### Performance Testing
```bash
python tests/test_performance.py
```

## Architecture Overview

### Core Components

1. **Main Simulation (`rotasim/rotasim.py`)**
   - `Sim` class extends `starsim.Sim`
   - Entry point for running simulations
   - Default parameters: 10,000 agents, 10-year timelimit, annual timestep

2. **Genetic Model (`rotasim/rotasim_genetics.py`)**
   - `Rota` class: Main disease model with genetic diversity
   - `RotaPathogen`: Handles pathogen genetics and evolution
   - `PathogenMatch`: Defines strain similarity (homotypic, partial hetero, complete hetero)
   - Tracks G, P, A, B segments for genetic diversity

3. **Interventions (`rotasim/interventions.py`)**
   - `RotaVax`: Vaccination product class
   - `RotaVaxProg`: Vaccination program implementation
   - Supports multi-dose vaccines with dose-specific efficacy
   - Strain-specific vaccine efficacy based on genetic matching

4. **Analysis (`rotasim/analyzers.py`)**
   - `StrainStats`: Tracks strain proportions and counts over time
   - Built on `starsim.Analyzer` framework

### Key Dependencies
- `starsim`: Core simulation framework
- `sciris`: Utilities and data structures
- `numpy`: Numerical computations
- `numba`: Performance optimization
- `pandas`: Data analysis
- `matplotlib`: Plotting
- `optuna`: Calibration optimization (in calibration module)

### Directory Structure
- `rotasim/`: Core simulation modules
- `tests/`: Unit tests and simple simulation scripts
- `data_analysis/`: Plotting and analysis scripts
- `calibration/`: Model calibration tools and data
- `archive/`: Legacy code versions

## Development Patterns

### Simulation Setup
```python
import rotasim as rs

sim = rs.Sim(
    n_agents=50000,
    timelimit=10,
    verbose=True,
    rota_kwargs={"vaccination_time": 5, "time_to_equilibrium": 2}
)
sim.run()
```

### Accessing Results
```python
events = sim.connectors["rota"].event_dict
strain_count = sim.connectors.rota.strain_count
```

### Adding Analyzers
Analyzers should extend `starsim.Analyzer` and implement:
- `init_results()`: Initialize result storage
- `step()`: Collect data at each timestep

### Current Immunity System (V1)
**Immunity Portfolio**: Each person maintains `immunity[uid] = {strain: recovery_time}` dict
**Waning**: Poisson process removes entire immunity portfolio based on time since first infection  
**Protection Logic**: When exposed to new strain:
- Same G,P → Homotypic protection (50%)
- Shared G OR P → Partial heterotypic protection (50%) 
- No shared G,P → Complete heterotypic protection (50%)

### V2 Immunity Performance Optimization
**Problem**: Current system requires triple loops (uid × current_disease × prior_disease)
**Solution**: Bitmask vectorization using `ss.FloatArr` with bitwise operations
- `exposed_G_bitmask`: Single integer per agent tracking all G genotypes exposed to
- `exposed_P_bitmask`: Single integer per agent tracking all P genotypes exposed to
- Zero UID loops using numpy vectorized bitwise operations (`&`, `|`, `<<`)

### Genetic Matching
The model uses a 4-segment genetic system (G, P, A, B):
- **Antigenic segments (G,P)**: Drive immunity and transmission
- **Non-antigenic segments (A,B)**: "Backbone" inherited during reassortment
- **Match types**: HOMOTYPIC (same G,P), PARTIAL_HETERO (shared G or P), COMPLETE_HETERO (no shared G,P)

### Vaccination Parameters
Vaccines are configured with:
- `vaccine_efficacy_match_factor`: Efficacy by genetic match type
- `vaccine_efficacy_dose_factor`: Relative efficacy by dose number
- `mean_dur_protection`: Duration of protection per dose

## Common Development Tasks

### Calibration
Use `calibration/calibration.py` for model calibration with:
- Optuna optimization
- Custom goodness-of-fit metrics
- Data processing utilities

### Performance Analysis
- Use `tests/test_performance.py` for benchmarking
- Profile with `tests/profile_rotaABM.py`
- Performance targets stored in `tests/test_performance.json`

### Result Analysis
- Raw results typically saved as CSV files
- Use `data_analysis/plot_results.py` for standard plots
- Strain diversity analysis with pie charts and line plots available

## Testing Strategy

Tests are structured as:
- Unit tests for individual components
- Integration tests using small populations (N=2,000)
- Performance regression tests
- Baseline comparison tests using saved JSON events

The project uses pytest for testing but can be run with simple Python execution of test files.