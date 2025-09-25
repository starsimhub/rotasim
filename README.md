# Rotasim

An agent-based model for simulating rotavirus genetic diversity and vaccination interventions. Rotasim is built on the Starsim v3 framework and uses genetic modeling to track pathogen evolution and immune responses across multiple strains.

**Current Status**: V2 architecture complete with unified scenario API for easy strain configuration and improved performance.

## Features

- **Unified scenario API**: Easy-to-use predefined scenarios with strain definitions, fitness, and prevalence
- **Multi-strain architecture**: Each rotavirus G,P combination is modeled as a separate disease instance
- **Cross-strain immunity**: High-performance vectorized immunity tracking with bitmask operations
- **Genetic reassortment**: Realistic genetic recombination between co-infecting strains
- **Population dynamics**: Birth, death, and aging processes
- **Comprehensive analysis**: Built-in analyzers for strain statistics, events, and demographics
- **Clean codebase**: Recently refactored to remove 435+ lines of deprecated code

## Installation

```bash
pip install -e .
```

This installs Rotasim as an importable module in development mode.

### Quick Start

```python
import rotasim as rs

# Run a simple simulation
sim = rs.Sim(scenario='simple')  # G1P8 and G2P4 strains
sim.run()
print(f"Simulation completed with {len(sim.diseases)} diseases")
```

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

### Basic Simulation with Unified Scenarios

```python
import rotasim as rs

# Simple simulation using predefined scenario
sim = rs.Sim(scenario='simple')  # Two-strain scenario (G1P8, G2P4)
sim.run()

# Access results
print(f"Simulation completed with {len(sim.diseases)} total diseases")
print(f"Scenario: {sim.scenario}")
print(f"Initial strains: {sim.initial_strains}")
```

### Available Built-in Scenarios

```python
import rotasim as rs

# List all available scenarios
scenarios = rs.list_scenarios()
for name, description in scenarios.items():
    print(f"{name}: {description}")

# Use different scenarios
sim_baseline = rs.Sim(scenario='baseline')           # 3 common global strains
sim_diverse = rs.Sim(scenario='high_diversity')      # 12 strains with varied fitness
sim_competition = rs.Sim(scenario='realistic_competition')  # G1P8 dominant with competition
```

### Scenario Customization with Overrides

```python
import rotasim as rs

# Override scenario parameters
sim = rs.Sim(
    scenario='baseline',
    override_prevalence=0.02,  # Set all strains to 2% prevalence
    override_fitness={(1,8): 0.95, (2,4): 0.8},  # Override specific strain fitness
    base_beta=0.15,  # Adjust base transmission rate
    n_agents=50000
)
sim.run()

# Add new strain to existing scenario
sim = rs.Sim(
    scenario='baseline', 
    override_strains={(9,6): {'fitness': 0.7, 'prevalence': 0.003}},  # Add G9P6
    verbose=True
)

# View strain summary
summary = sim.get_strain_summary()
print(f"Total diseases: {summary['total_diseases']}")
print(f"Active strains: {len(summary['active_strains'])}")
print(f"Dormant reassortants: {len(summary['dormant_strains'])}")
```

### Custom Scenarios

```python
import rotasim as rs

# Define custom scenario
custom_scenario = {
    'strains': {
        (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
        (2, 4): {'fitness': 0.8, 'prevalence': 0.010},
        (3, 6): {'fitness': 0.9, 'prevalence': 0.005}
    },
    'default_fitness': 0.3  # For dormant reassortants
}

sim = rs.Sim(
    scenario=custom_scenario,
    base_beta=0.1,
    n_agents=25000
)
sim.run()
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
        scenario='high_diversity',  # Use built-in scenario
        connectors=[immunity],
        n_agents=25000,
        dt=rs.days(1),  # Daily timesteps
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
    scenario='realistic_competition',  # Use built-in scenario
    analyzers=analyzers,
    dt=rs.days(1),  # Daily timesteps
    stop='2030-01-01',  # 10-year simulation
    verbose=1
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
print(f"Final scenario: {sim.final_scenario}")
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
# Test multi-strain architecture and unified scenarios
cd tests
python -m pytest test_integration.py -v
python test_utils.py  # Test scenario system
```

### Performance Benchmarks

```bash
# Test performance with large strain numbers
python tests/test_performance.py
```

## Advanced Configuration

### Custom Protection Factors with Scenarios

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
    scenario='balanced_competition',  # Use built-in 4-strain scenario
    connectors=[strong_homotypic],
    base_beta=0.12,
    verbose=2  # Detailed output
)
```

### Monitoring Cross-Strain Interactions

```python
import rotasim as rs
from rotasim.analyzers import EventStats

# Track immunity and reassortment events
analyzer = EventStats()
sim = rs.Sim(
    scenario='emergence_scenario',  # Weak background for emergence studies
    analyzers=[analyzer],
    stop='2035-01-01',  # 15-year simulation
    dt=rs.days(1),     # Daily timesteps
    verbose=1  # Show summary information
)

sim.run()

# Analyze cross-strain dynamics
events = analyzer.to_df()
print(f"Immunity waning events: {events['wanings'].sum()}")
print(f"Reassortment events: {events['reassortments'].sum()}")
print(f"Peak coinfected agents: {events['coinfected_agents'].max()}")
print(f"Final scenario used: {sim.final_scenario['strains']}")
```

### Parameter Sensitivity Analysis

```python
import rotasim as rs
import numpy as np

# Test scenario and immunity parameter sensitivity
scenarios = ['simple', 'baseline', 'realistic_competition']
cross_protection_levels = [0.2, 0.4, 0.6, 0.8]
results = []

for scenario in scenarios:
    for cross_prot in cross_protection_levels:
        immunity = rs.RotaImmunityConnector(
            partial_heterotypic_immunity_efficacy=cross_prot
        )
        
        sim = rs.Sim(
            scenario=scenario,
            connectors=[immunity],
            dt=rs.days(1),
            stop='2030-01-01',
            verbose=0
        )
        sim.run()
        
        results.append({
            'scenario': scenario,
            'cross_protection': cross_prot,
            'total_diseases': len(sim.diseases),
            'initial_strains': len(sim.initial_strains)
        })

print("Scenario and cross-protection sensitivity:")
for result in results:
    print(f"Scenario: {result['scenario']}, Protection: {result['cross_protection']:.1f}, "
          f"Total diseases: {result['total_diseases']}, Initial: {result['initial_strains']}")
```

## Recent Changes (V2 Architecture)

### Unified Scenario System
- **Built-in scenarios**: 8 predefined scenarios from simple 2-strain to complex 12-strain setups
- **Easy customization**: Override fitness, prevalence, or add new strains to existing scenarios
- **Clean API**: Single `scenario` parameter replaces multiple configuration options
- **Backward compatibility**: All existing functionality preserved

### Code Cleanup
- **Removed 435+ lines** of deprecated code after unified scenario implementation
- **Eliminated legacy dictionaries**: `INITIAL_STRAIN_SCENARIOS`, `FITNESS_HYPOTHESES` 
- **Streamlined imports**: Clean import structure with only necessary functions
- **Updated examples**: All documentation updated to use new unified API

### Available Scenarios
```python
import rotasim as rs

# List all built-in scenarios
print(rs.list_scenarios())
# Output:
# {
#   'simple': 'Simple two-strain scenario - G1P8 and G2P4 with equal fitness and prevalence',
#   'baseline': 'Baseline scenario - common global strains with equal fitness',
#   'realistic_competition': 'G1P8 dominant with realistic strain competition',
#   'balanced_competition': 'G1P8 dominant with moderate balanced competition',
#   'high_diversity': 'High diversity with 12 strains and varied fitness',
#   'low_diversity': 'Low diversity with 4 main competitive strains',
#   'emergence_scenario': 'Scenario for studying strain emergence with weak background'
# }
```

## Contributing

This project follows standard development practices:

1. **Testing**: Run `python -m pytest tests/` and `python tests/test_utils.py` before submitting changes
2. **Performance**: Use `python tests/test_performance.py` to check performance  
3. **Style**: Follow existing code patterns and documentation standards
4. **Scenarios**: Use built-in scenarios when possible, create custom scenarios for specific needs
5. **Verbosity**: Use `verbose=1` for summary info, `verbose=2` for detailed strain creation output

## Dependencies

- `starsim`: Core simulation framework
- `numpy`: Numerical computations and bitmask operations  
- `pandas`: Data analysis and export
- `matplotlib`: Plotting and visualization
- `numba`: Performance optimization (optional)
- `sciris`: Utilities and parameter management