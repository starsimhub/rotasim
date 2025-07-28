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
# Create individual strain instances with G,P as first-class parameters
rota_g1p8 = Rotavirus(G=1, P=8, backbone=(1,1))    # Auto-named "G1P8_11"
rota_g2p4 = Rotavirus(G=2, P=4, backbone=(1,1))    # Auto-named "G2P4_11" 
rota_g3p6 = Rotavirus(G=3, P=6, backbone=(2,2))    # Auto-named "G3P6_22"

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
    def __init__(self, G, P, backbone=(1, 1), name=None, **kwargs):
        # G, P: first-class antigenic genotypes (required)
        # backbone: flexible tuple for non-antigenic segments (A, B, ...)
        # name: auto-generated if not provided
        
        if name is None:
            backbone_str = ''.join([f'{seg}' for seg in backbone])
            name = f"G{G}P{P}_{backbone_str}"  # e.g., "G1P8_11" or "G2P4_22"
        
        super().__init__(name=name, **kwargs)
        self.G = G
        self.P = P
        self.backbone = backbone
        
    @property
    def strain(self):
        """Complete strain tuple (G, P, *backbone) for compatibility"""
        return (self.G, self.P) + self.backbone
        
    @property
    def antigenic_segments(self):
        """Antigenic segments for immunity matching"""
        return (self.G, self.P)
        
    def match_strain(self, other):
        """Return HOMOTYPIC/PARTIAL_HETERO/COMPLETE_HETERO based on G,P"""
        if isinstance(other, Rotavirus):
            other_gp = (other.G, other.P)
        else:
            other_gp = other[:2]  # assume tuple format
            
        if self.antigenic_segments == other_gp:
            return PathogenMatch.HOMOTYPIC
        elif self.G == other_gp[0] or self.P == other_gp[1]:
            return PathogenMatch.PARTIAL_HETERO
        else:
            return PathogenMatch.COMPLETE_HETERO
```

**Key Features:**
- **G and P as first-class parameters**: `Rotavirus(G=1, P=8)` is biologically intuitive
- **Flexible backbone**: `backbone=(1,1)` for A1B1, extensible for future segments
- **Auto-naming**: "G1P8_11" clearly shows antigenic type and genetic background
- **Strain fitness**: Handled through different `beta` values per instance
- **Backward compatibility**: `.strain` property returns full tuple for existing code

### Phase 1.2: Cross-Strain Immunity Connector

**Current Immunity System (to preserve exactly):**
1. **Immunity Portfolio**: Each person maintains `immunity[uid] = {strain: recovery_time}` dict
2. **Waning**: Poisson process removes entire immunity portfolio based on time since first infection  
3. **Protection Logic**: When exposed to new strain, check if person has ever had:
   - Same G,P → Homotypic protection (50%)
   - Shared G OR P → Partial heterotypic protection (50%) 
   - No shared G,P → Complete heterotypic protection (50%)

**RotaImmunityConnector Design (High-Performance Vectorized):**
```python
class RotaImmunityConnector(ss.Connector):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        self.define_pars(
            homotypic_immunity_rate=0.5,
            partial_heterotypic_immunity_rate=0.5, 
            complete_heterotypic_immunity_rate=0.5,
            waning_rate=365/273,  # omega parameter
        )
        
    def init_post(self, sim):
        """Auto-detect Rotavirus diseases and create bitmask mappings"""
        super().init_post(sim)
        self.rota_diseases = [d for d in sim.diseases.values() 
                            if isinstance(d, Rotavirus)]
        
        # Create bit mappings for unique G,P types in simulation
        unique_G = sorted(set(d.G for d in self.rota_diseases))
        unique_P = sorted(set(d.P for d in self.rota_diseases))
        
        self.G_to_bit = {g: i for i, g in enumerate(unique_G)}
        self.P_to_bit = {p: i for i, p in enumerate(unique_P)}
        
        # Pre-compute disease-specific bitmasks for fast lookup
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        for disease in self.rota_diseases:
            self.disease_G_masks[disease.name] = 1 << self.G_to_bit[disease.G]
            self.disease_P_masks[disease.name] = 1 << self.P_to_bit[disease.P]
        
    def define_states(self):
        self.define_states(
            ss.FloatArr('exposed_G_bitmask', default=0.0),  # Bitmask of exposed G types
            ss.FloatArr('exposed_P_bitmask', default=0.0),  # Bitmask of exposed P types
            ss.FloatArr('oldest_infection', default=np.nan),
            ss.BoolArr('has_immunity', default=False),
        )
        
    def step(self):
        # 1. Vectorized immunity waning
        self._apply_waning()
        
        # 2. Vectorized cross-immunity protection calculation
        self._update_cross_immunity()
        
    def _apply_waning(self):
        """Vectorized waning - clear entire immunity portfolios"""
        n_waning = np.random.poisson(self.pars.waning_rate * self.dt * self.has_immunity.sum())
        if n_waning > 0:
            immune_uids = self.has_immunity.uids
            oldest_times = self.oldest_infection[immune_uids]
            waning_indices = np.argpartition(oldest_times, n_waning)[:n_waning]
            waning_uids = immune_uids[waning_indices]
            
            # Clear all immunity bitmasks
            self.exposed_G_bitmask[waning_uids] = 0.0
            self.exposed_P_bitmask[waning_uids] = 0.0
            self.has_immunity[waning_uids] = False
            self.oldest_infection[waning_uids] = np.nan
            
    def _update_cross_immunity(self):
        """Fully vectorized using bitwise operations - NO UID LOOPS"""
        for disease in self.rota_diseases:
            disease_G_mask = self.disease_G_masks[disease.name]
            disease_P_mask = self.disease_P_masks[disease.name]
            
            # Convert FloatArr to int for bitwise ops, then back to bool
            G_bits = self.exposed_G_bitmask.astype(int)
            P_bits = self.exposed_P_bitmask.astype(int)
            
            has_G_match = (G_bits & disease_G_mask) != 0
            has_P_match = (P_bits & disease_P_mask) != 0
            
            has_homotypic = has_G_match & has_P_match
            has_partial = (has_G_match | has_P_match) & ~has_homotypic
            
            # Vectorized protection assignment
            protection = np.where(
                has_homotypic, self.pars.homotypic_immunity_rate,
                np.where(has_partial, self.pars.partial_heterotypic_immunity_rate,
                        self.pars.complete_heterotypic_immunity_rate)
            )
            
            # Apply only to people with immunity (performance optimization)
            mask = self.has_immunity
            disease.rel_sus[mask] = 1.0 - protection[mask]
    
    def record_recovery(self, disease, recovered_uids):
        """Update bitmasks when people recover from infections"""
        if len(recovered_uids) == 0:
            return
            
        G_bit = 1 << self.G_to_bit[disease.G]  
        P_bit = 1 << self.P_to_bit[disease.P]
        
        # Update bitmasks with type conversion
        self.exposed_G_bitmask[recovered_uids] = (
            self.exposed_G_bitmask[recovered_uids].astype(int) | G_bit
        ).astype(float)
        
        self.exposed_P_bitmask[recovered_uids] = (
            self.exposed_P_bitmask[recovered_uids].astype(int) | P_bit  
        ).astype(float)
        
        self.has_immunity[recovered_uids] = True
        
        # Track oldest infection time
        first_infections = np.isnan(self.oldest_infection[recovered_uids])
        self.oldest_infection[recovered_uids[first_infections]] = self.sim.ti
```

**Bitmask Vectorization Benefits:**
- **True vectorization**: Zero UID loops, pure numpy array operations on entire population
- **Memory efficient**: 2 floats per agent vs. potentially dozens of boolean arrays  
- **Starsim compatible**: Uses standard ss.FloatArr with automatic birth/death handling
- **Scalable**: Handles up to 32/64 unique G,P types (sufficient for rotavirus)
- **Fast bitwise operations**: `&`, `|`, `<<` are very efficient

**Extensibility:**
- **Modular design**: DetailedRotaImmunityConnector can be separate ss.Connector
- **Clean interface**: `record_recovery()` method for easy integration with disease instances
- **Future-proof**: Bitmask approach can handle dynamic strain addition during reassortment

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