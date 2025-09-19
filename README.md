# Rotasim

An agent-based model for simulating rotavirus genetic diversity and vaccination interventions. Rotasim is built on the Starsim v3 framework and uses genetic modeling to track pathogen evolution and immune responses across multiple strains.

## Features

- **Multi-strain architecture**: Each rotavirus G,P combination is modeled as a separate disease instance
- **Cross-strain immunity**: High-performance vectorized immunity tracking with bitmask operations
- **Genetic reassortment**: Realistic genetic recombination between co-infecting strains
- **Population dynamics**: Birth, death, and aging processes
- **Comprehensive analysis**: Built-in analyzers for strain statistics, events, and demographics

## Installation

```bash
pip install -e .
```

This installs Rotasim as an importable module in development mode.

## Model Architecture

### Strain Representation

Rotasim uses a multi-strain architecture where each unique G,P combination is represented as a separate `Rotavirus` disease instance:

- **G genotype**: Major outer capsid protein determining antigenic properties
- **P genotype**: Protease-sensitive protein determining antigenic properties  
- **Backbone**: Non-antigenic segments (A,B) inherited during reassortment

Each strain is automatically named using the format `G{G}P{P}` (e.g., "G1P8", "G2P4").

#### Initial Strains and Reassortment

The model starts with a set of initial strains and generates all possible G,P combinations through reassortment:

```python
# 3 initial strains create 9 total combinations
initial_strains = [(1, 8), (2, 4), (3, 6)]  
# Results in: G1P8, G1P4, G1P6, G2P8, G2P4, G2P6, G3P8, G3P4, G3P6
```

#### Active vs Dormant Strains

- **Active strains**: Present in the initial population with non-zero prevalence
- **Dormant strains**: Created through reassortment but initially absent from the population

### Immunity Representation

Rotasim tracks immunity using a high-performance bitmask system for computational efficiency:

#### Exposure Tracking

Each agent maintains exposure history using bitmasks:
- `exposed_G_bitmask`: Integer tracking all G genotypes the agent has been exposed to
- `exposed_P_bitmask`: Integer tracking all P genotypes the agent has been exposed to
- `exposed_GP_bitmask`: Integer tracking all specific G,P combinations the agent has been exposed to

#### Cross-Strain Protection Logic

When an agent is exposed to a new strain, protection is determined by previous exposures using a hierarchical matching system with **maximum protection principle**:

1. **Homotypic Protection**: Exact G and P match with previous infection
   - **Default protection**: 90% (`homotypic_immunity_efficacy = 0.9`)
   - Example: Previous G1P8 infection protects against new G1P8 exposure

2. **Partial Heterotypic Protection**: Shared G OR P genotype (but not both)
   - **Default protection**: 50% (`partial_heterotypic_immunity_efficacy = 0.5`)
   - **Key feature**: Uses the **most protective partial match** available
   - Example: Agent previously infected with G1P8 and G2P4, now exposed to G1P6
     - G1P6 shares G=1 with G1P8 infection → partial G-type protection
     - G1P6 has P=6, no previous P=6 exposure → no P-type protection
     - **Result**: Receives G1-based partial protection (the available match)

3. **Complete Heterotypic Protection**: No shared G or P genotypes but has prior immunity
   - **Default protection**: 30% (`complete_heterotypic_immunity_efficacy = 0.3`)
   - Example: Previous G1P8 infection, exposed to G2P4 (no shared antigens)

4. **Naive Susceptibility**: No prior immunity to any strain
   - **Default protection**: 0% (`naive_immunity_efficacy = 0.0`)
   - Fully susceptible to first rotavirus infection

#### Immunity Waning Mechanics

Immunity wanes through exponential decay based on time since recovery from each strain:

- **Waning process**: Individual strain immunity decays exponentially over time
- **Default waning rate**: ~273 days (`full_waning_rate = ss.freqperyear(365/273)`)
- **No waning delay**: Decay begins immediately after recovery (`immunity_waning_delay = ss.days(0)`)
- **Gradual waning**: Unlike complete portfolio loss, immunity decays gradually for each strain
- **Maximum protection principle**: For partial heterotypic protection, uses the strongest remaining immunity from either G-type or P-type matches

#### Mathematical Implementation

The cross-strain protection calculation uses bitwise operations for efficiency:

```python
# Check for homotypic match (exact G,P match)
has_exact_match = (exposed_GP_bitmask & disease_GP_mask) != 0

# Check for partial heterotypic match (G match OR P match, but not both)
has_G_match = (exposed_G_bitmask & disease_G_mask) != 0 & ~has_exact_match
has_P_match = (exposed_P_bitmask & disease_P_mask) != 0 & ~has_exact_match

# For partial matches, use maximum decay factor from G or P immunity
max_decay_factor = max(G_max_decayed_immunity_factor, P_max_decayed_immunity_factor)

# Calculate final protection
protection_factor = strain_match_immunity_efficacy × decay_factor
```

### Exposure Effects on Susceptibility (rel_sus)

The `rel_sus` (relative susceptibility) parameter controls infection probability and is modified by:

1. **Cross-strain immunity**: Reduced susceptibility based on previous exposures and their decay
2. **Infection history**: Scaling factor based on total number of previous infections

When an agent is exposed to a strain, their `rel_sus` is calculated as:
```
rel_sus = (1 - strain_immunity_efficacy × decay_factor) × infection_history_factor
```

Where:
- `strain_immunity_efficacy` comes from the hierarchical matching system (0.0-0.9)
- `decay_factor` represents time-based immunity waning (0.0-1.0)
- `infection_history_factor` scales based on total previous infections

## Usage Examples

### Basic Simulation

```python
import rotasim as rs

# Simple simulation with default parameters
sim = rs.Sim()
sim.run()

# Access results
print(f"Simulation completed: {sim.ti} timesteps")
```

### Multi-Strain Simulation with Custom Immunity

```python
import rotasim as rs
from rotasim import RotaImmunityConnector

# Custom immunity parameters
immunity = RotaImmunityConnector(
    homotypic_immunity_efficacy=0.95,        # Very strong same-strain protection
    partial_heterotypic_immunity_efficacy=0.7,  # Strong cross-protection
    complete_heterotypic_immunity_efficacy=0.4, # Moderate heterotypic protection
    full_waning_rate=rs.freqperyear(365/365)    # 1 year waning duration
)

# Specify initial strains (creates 9 total combinations)  
initial_strains = [(1, 8), (2, 4), (3, 6)]
sim = rs.Sim(
    initial_strains=initial_strains,
    connectors=[immunity],
    n_agents=50000,
    timelimit=10,
    verbose=True
)
sim.run()

# View strain summary
summary = sim.get_strain_summary()
print(f"Active strains: {len(summary['active_strains'])}")
print(f"Dormant strains: {len(summary['dormant_strains'])}")
```

### Testing Cross-Strain Protection

```python
import rotasim as rs
from rotasim import RotaImmunityConnector

# Test different protection scenarios
high_cross_protection = RotaImmunityConnector(
    homotypic_immunity_efficacy=0.9,
    partial_heterotypic_immunity_efficacy=0.8,  # High cross-protection
    complete_heterotypic_immunity_efficacy=0.6
)

low_cross_protection = RotaImmunityConnector(
    homotypic_immunity_efficacy=0.9,
    partial_heterotypic_immunity_efficacy=0.3,  # Low cross-protection
    complete_heterotypic_immunity_efficacy=0.1
)

# Compare strain diversity outcomes
for immunity, label in [(high_cross_protection, "High"), (low_cross_protection, "Low")]:
    sim = rs.Sim(
        initial_strains=[(1, 8), (2, 4), (3, 6), (4, 8)],
        connectors=[immunity],
        timelimit=15,
        verbose=0
    )
    sim.run()
    print(f"{label} cross-protection: simulation completed")
```

### Analysis and Data Export

```python
import rotasim as rs
from rotasim.analyzers import StrainStats, EventStats

# Add analyzers to track detailed statistics
analyzers = [
    StrainStats(),    # Strain proportions and counts
    EventStats()      # Births, deaths, infections, waning events
]

sim = rs.Sim(
    initial_strains=[(1, 8), (2, 4), (3, 8)],
    analyzers=analyzers,
    timelimit=10
)
sim.run()

# Export results
strain_stats = analyzers[0].to_df()
event_stats = analyzers[1].to_df()

strain_stats.to_csv("strain_proportions.csv", index=False)
event_stats.to_csv("simulation_events.csv", index=False)

# Examine immunity waning events
waning_events = event_stats['wanings'].sum() 
print(f"Total immunity waning events: {waning_events}")
```

## Running Examples

### Quick Test

```bash
# Run from tests directory
cd tests
python simple.py
```

### Integration Tests

```bash
# Test multi-strain architecture
cd tests
python -m pytest test_integration.py -v
```

### Performance Benchmarks

```bash
# Test performance with large strain numbers
python tests/test_performance.py
```

## Advanced Configuration

### Custom Protection Factors

```python
import rotasim as rs
from rotasim import RotaImmunityConnector

# Asymmetric protection scenarios
strong_homotypic = RotaImmunityConnector(
    homotypic_immunity_efficacy=0.95,       # Very strong same-strain
    partial_heterotypic_immunity_efficacy=0.4,   # Moderate cross-protection
    complete_heterotypic_immunity_efficacy=0.1,  # Minimal heterotypic
    full_waning_rate=rs.freqperyear(365/180)     # 6-month waning
)

sim = rs.Sim(
    initial_strains=[(1, 8), (2, 4), (3, 6), (4, 8)],
    connectors=[strong_homotypic]
)
```

### Monitoring Cross-Strain Interactions

```python
import rotasim as rs
from rotasim.analyzers import EventStats

# Track immunity and reassortment events
analyzer = EventStats()
sim = rs.Sim(
    initial_strains=[(1, 8), (2, 4), (3, 8)],
    analyzers=[analyzer],
    timelimit=15,
    verbose=1  # Show summary information
)

sim.run()

# Analyze cross-strain dynamics
events = analyzer.to_df()
print(f"Immunity waning events: {events['wanings'].sum()}")
print(f"Reassortment events: {events['reassortments'].sum()}")
print(f"Peak coinfected agents: {events['coinfected_agents'].max()}")
```

### Parameter Sensitivity Analysis

```python
import rotasim as rs
import numpy as np

# Test immunity parameter sensitivity
cross_protection_levels = [0.2, 0.4, 0.6, 0.8]
results = []

for cross_prot in cross_protection_levels:
    immunity = rs.RotaImmunityConnector(
        partial_heterotypic_immunity_efficacy=cross_prot
    )
    
    sim = rs.Sim(
        initial_strains=[(1, 8), (2, 4), (3, 8)],
        connectors=[immunity],
        timelimit=10,
        verbose=0
    )
    sim.run()
    
    results.append({
        'cross_protection': cross_prot,
        'total_strains': len(sim.diseases)
    })

print("Cross-protection sensitivity:")
for result in results:
    print(f"Protection: {result['cross_protection']:.1f}, Total strains: {result['total_strains']}")
```

## Contributing

This project follows standard development practices:

1. **Testing**: Run `python -m pytest tests/` before submitting changes
2. **Performance**: Use `python tests/test_performance.py` to check performance  
3. **Style**: Follow existing code patterns and documentation standards
4. **Verbosity**: Use `sim.pars.verbose` for debug output control

## Dependencies

- `starsim`: Core simulation framework
- `numpy`: Numerical computations and bitmask operations  
- `pandas`: Data analysis and export
- `matplotlib`: Plotting and visualization
- `numba`: Performance optimization (optional)
- `sciris`: Utilities and parameter management