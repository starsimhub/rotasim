# Rotasim v2 Architecture Refactoring Plan

## Overview
Refactor current monolithic Rotasim implementation to follow traditional Starsim design patterns by treating each rotavirus strain as a separate disease instance, with reassortment handled via connectors.

## Architectural Vision

### Current Architecture Issues
- `Rota(ss.Module)` is monolithic, handling all strains in one class
- Genetic diversity, transmission, and immunity logic tightly coupled  
- Reassortment embedded within single module
- Networks not cleanly separated from disease logic
- Interventions must handle all strain combinations internally

### New Architecture Design

#### Core Components
1. **Rotavirus(ss.Infection)** - Individual strain-specific disease class
2. **RotaImmunityConnector(ss.Connector)** - Cross-strain immunity interactions  
3. **RotaReassortment(ss.Connector)** - Genetic reassortment mechanism
4. **Multiple Analyzers** - Replicate current output functionality
5. **Enhanced Interventions** - Multi-strain aware vaccination

#### Key Design Principles
- **One strain = One disease instance**: Each G/P/A/B strain combination becomes a separate `Rotavirus(ss.Infection)` instance
- **Native Starsim multi-disease support**: Use `diseases=[rota_g1p8, rota_g2p4, ...]` - no custom Sim modifications needed
- **Standard networks**: Leverage existing Starsim network classes (ss.RandomNet, ss.MFNet, etc.)
- **Connector-based interactions**: All cross-strain effects handled by ss.Connector classes
- **Auto-detection**: Connectors automatically detect Rotavirus instances in simulation
- **Modular analyzers**: Separate analyzer classes aggregate data across strain instances

#### Example Usage
```python
# Create individual strain instances
rota_g1p8 = Rotavirus(strain=(1,8,1,1))  # Auto-named "G1P8A1B1"
rota_g2p4 = Rotavirus(strain=(2,4,1,1))  # Auto-named "G2P4A1B1"
rota_g3p6 = Rotavirus(strain=(3,6,1,1))  # Auto-named "G3P6A1B1"

# Connectors auto-detect Rotavirus instances - no manual listing needed
immunity = RotaImmunityConnector()  # Finds all Rotavirus diseases automatically
reassortment = RotaReassortment()   # Finds all Rotavirus diseases automatically

# Standard Starsim simulation - no custom Sim class needed
sim = ss.Sim(
    diseases=[rota_g1p8, rota_g2p4, rota_g3p6],
    connectors=[immunity, reassortment],
    networks='random',
    n_agents=10000
)
```

## Initial Design Phase Implementation

### Phase 1.1: Core Rotavirus Disease Class

**Rotavirus(ss.Infection) Design:**
```python
class Rotavirus(ss.Infection):
    def __init__(self, strain=(1,8,1,1), name=None, **kwargs):
        # Auto-generate name if not provided
        if name is None:
            name = f"G{strain[0]}P{strain[1]}A{strain[2]}B{strain[3]}"
        
        super().__init__(name=name, **kwargs)
        self.strain = strain
        
    @property
    def G(self):
        """G genotype"""
        return self.strain[0]
        
    @property  
    def P(self):
        """P genotype"""
        return self.strain[1]
        
    @property
    def antigenic_segments(self):
        """Antigenic segments for immunity matching (G,P)"""
        return self.strain[:2]
        
    def match_strain(self, other_strain):
        """Return HOMOTYPIC/PARTIAL_HETERO/COMPLETE_HETERO"""
        # Implementation based on current PathogenMatch logic
```

**Key Features:**
- Each instance represents one G/P/A/B strain combination
- Automatic strain naming (e.g., "G1P8A1B1")
- Convenient G and P genotype getters
- Strain fitness handled through different `beta` values per instance
- Standard ss.Infection transmission mechanics

### Phase 1.2: Cross-Strain Immunity Connector

**RotaImmunityConnector Design:**
```python
class RotaImmunityConnector(ss.Connector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.define_pars(
            homotypic_immunity_rate=0.5,
            partial_heterotypic_immunity_rate=0.5, 
            complete_heterotypic_immunity_rate=0.5,
        )
        
    def init_post(self, sim):
        """Auto-detect all Rotavirus disease instances in simulation"""
        super().init_post(sim)
        self.rota_diseases = []
        for disease in sim.diseases.values():
            if isinstance(disease, Rotavirus):
                self.rota_diseases.append(disease)
        
    def step(self):
        # For each person with prior infections:
        # - Check infection history across all detected Rotavirus strains
        # - Calculate cross-immunity protection level
        # - Modify rel_sus for other strains accordingly
```

**Auto-Detection Benefits:**
- No need to manually specify disease instances
- Automatically adapts when new strains are added via reassortment
- Simplifies user configuration

### Phase 1.3: Reassortment Connector

**Key Question for Starsim Core Team:**
> **CRITICAL DESIGN QUESTION**: Can new disease instances be dynamically added to a running simulation, or must we start with all possible strain combinations pre-defined?
> 
> This affects whether reassortment can create truly novel strains during simulation or if we need to pre-populate all theoretically possible G/P/A/B combinations at initialization.

**RotaReassortment Design Concept:**
```python
class RotaReassortment(ss.Connector):
    def init_post(self, sim):
        """Auto-detect all Rotavirus disease instances"""
        super().init_post(sim)
        self.rota_diseases = []
        for disease in sim.diseases.values():
            if isinstance(disease, Rotavirus):
                self.rota_diseases.append(disease)
    
    def step(self):
        # Identify hosts infected with multiple strains
        # Generate new strain combinations (G/P/A/B reassortment)
        # Either:
        #   A) Dynamically create new Rotavirus instances (if Starsim supports)
        #   B) Activate pre-existing dormant strain instances
```

**Implementation Options:**
1. **Dynamic Creation**: Create new Rotavirus instances during simulation (preferred)
2. **Pre-population**: Start with all possible combinations, activate as needed

## Benefits of New Architecture

### For Researchers
- **Modular strain modeling**: Easy to add/remove specific strains
- **Clear separation of concerns**: Transmission, immunity, and reassortment logic separated
- **Experimental flexibility**: Test hypotheses about specific strain interactions
- **Standard Starsim patterns**: Familiar framework for Starsim users
- **Auto-configuration**: Connectors automatically detect and adapt to strain instances

### For Model Development
- **Maintainable code**: Following established Starsim design patterns
- **Extensible framework**: Easy to add new features (seasonality, age-specific effects, etc.)
- **Performance optimization**: Parallel processing opportunities for strain instances
- **Testing isolation**: Test individual components separately
- **No backwards compatibility burden**: Clean slate implementation

### For Calibration & Analysis
- **Granular parameter control**: Tune parameters for individual strains
- **Detailed output tracking**: Monitor each strain's dynamics separately
- **Flexible intervention modeling**: Strain-specific vaccination strategies
- **Scenario analysis**: Compare different strain composition scenarios

## Implementation Timeline

**Phase 1 (Weeks 1-2)**: Core Rotavirus class + basic immunity connector with auto-detection  
**Phase 2 (Weeks 3-4)**: Dynamic reassortment mechanism (pending Starsim core team input)  
**Phase 3 (Weeks 4-5)**: Enhanced analyzers and interventions  
**Phase 4 (Weeks 5-6)**: Integration, testing, and optimization  

## Questions for Starsim Core Development Team

1. **Dynamic Disease Creation**: Can new disease instances be added to a running simulation after initialization?
2. **Connector Auto-Detection**: Best practices for connectors to automatically discover relevant disease instances?
3. **Performance Considerations**: Any recommendations for managing large numbers of disease instances (potential 100+ strain combinations)?

---

## Appendix: Current System Features (For Reference)

### Key Current Features to Preserve
- G/P/A/B strain combinations with antigenic segments (typically G,P)
- 3-level immunity: HOMOTYPIC > PARTIAL_HETERO > COMPLETE_HETERO  
- Strain fitness based on G/P combinations
- Age-structured population with specific age bins
- Reassortment creating new strain combinations
- Multi-dose vaccination with strain-specific efficacy

### Current CSV Outputs (Maintain Compatibility)
1. **rota_strain_count_*.csv** - Count of each strain over time
2. **rota_strains_sampled_*.csv** - Sample of infected individuals with strains
3. **rota_strains_infected_all_*.csv** - All infected individuals with strains  
4. **rota_vaccinecount_*.csv** - Vaccination data over time
5. **rota_agecount_*.csv** - Age distribution data
6. **rota_vaccine_efficacy_*.csv** - Vaccine efficacy metrics
7. **event_counts_*.csv** - Event counts and statistics
8. **immunity_counts_*.csv** - Immunity tracking data

---

**Document Status:** Draft for review by rotavirus research and model development teams  
**Branch:** `rotasim_architecture`  
**Date:** 2025-01-28