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
# Define initial G,P strain combinations (backbone removed for performance)
initial_strains = [(1,8), (2,4), (3,6)]

# Auto-generate ALL possible G,P reassortments at simulation start
# (Required: Starsim cannot add new diseases mid-simulation)
all_gp_pairs = generate_gp_reassortments(initial_strains)  # ~20-70 combinations

# Create Rotavirus instances for all possible G,P combinations
base_beta = 0.1  # Global transmission scaling
diseases = []
for G, P in all_gp_pairs:
    # Most reassortants start with zero prevalence
    init_prev = 0.01 if (G,P) in initial_strains else 0.0
    diseases.append(Rotavirus(G=G, P=P, init_prev=init_prev))

# Strain-specific fitness multipliers for easy parameter tuning
x_beta_lookup = {
    (1,8): 1.0,   # G1P8 baseline fitness
    (2,4): 0.8,   # G2P4 reduced fitness  
    (3,6): 1.2,   # G3P6 enhanced fitness
}
for disease in diseases:
    x_beta = x_beta_lookup.get((disease.G, disease.P), 1.0)
    disease.pars.beta = base_beta * x_beta

# Connectors auto-detect Rotavirus instances - no manual listing needed
immunity = RotaImmunityConnector()  # Finds all Rotavirus diseases automatically
reassortment = RotaReassortment()   # Uses set_prognoses() for co-infected hosts

# Standard Starsim simulation - no custom Sim class needed
sim = ss.Sim(
    diseases=diseases,  # All possible G,P combinations
    connectors=[immunity, reassortment],
    networks='random',
    n_agents=10000
)
```

## Initial Design Phase Implementation

### Phase 1.1: Core Rotavirus Disease Class

**Rotavirus(ss.Infection) Design (Simplified for Performance):**
```python
class Rotavirus(ss.Infection):
    def __init__(self, G, P, name=None, **kwargs):
        # G, P: first-class antigenic genotypes (required)
        # name: auto-generated if not provided
        
        if name is None:
            name = f"G{G}P{P}"  # e.g., "G1P8", "G2P4"
        
        super().__init__(name=name, **kwargs)
        self.G = G
        self.P = P
        
    @property
    def strain(self):
        """Strain tuple for compatibility with existing code"""
        return (self.G, self.P)
        
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
- **G,P only for initial implementation**: Massive performance improvement (16x fewer disease instances)
- **Biologically meaningful**: G,P drive all immunity and transmission dynamics
- **Simple naming**: "G1P8" clearly identifies antigenic type
- **Strain fitness**: Handled through `base_beta * x_beta_lookup[(G,P)]` multipliers
- **Future extensible**: Can add backbone parameter later once core architecture is validated

**Performance Benefits:**
- **Reduced complexity**: ~70 G,P combinations vs. 1,000+ with backbone
- **Faster simulation**: Fewer disease instances for Starsim to manage
- **Easier debugging**: Simpler architecture to validate and optimize

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

**Key Insight from Starsim Limitations:**
> **CRITICAL**: Starsim cannot add new disease instances mid-simulation. All possible reassortant G,P combinations must be created at initialization.

**RotaReassortment Design:**
```python
class RotaReassortment(ss.Connector):
    def init_post(self, sim):
        """Auto-detect all Rotavirus disease instances and create lookup"""
        super().init_post(sim)
        self.rota_diseases = [d for d in sim.diseases.values() 
                            if isinstance(d, Rotavirus)]
        
        # Create fast lookup: (G,P) -> disease instance
        self.strain_lookup = {(d.G, d.P): d for d in self.rota_diseases}
        
    def step(self):
        # 1. Identify hosts infected with multiple Rotavirus strains
        co_infected_hosts = self._find_co_infected_hosts()
        
        # 2. Generate reassortment events (Poisson process)
        n_reassortments = np.random.poisson(self.pars.reassortment_rate * len(co_infected_hosts))
        
        # 3. For each reassortment event:
        for host_uid in np.random.choice(co_infected_hosts, n_reassortments):
            # Get current strain infections
            current_strains = self._get_host_infections(host_uid)
            
            # Generate novel G,P combination from parents
            new_gp = self._reassort_genotypes(current_strains)
            
            # Use set_prognoses() to infect with pre-existing disease instance
            reassortant_disease = self.strain_lookup[new_gp]
            reassortant_disease.set_prognoses([host_uid])
            
    def _find_co_infected_hosts(self):
        """Find hosts infected with 2+ different Rotavirus strains"""
        infection_counts = np.zeros(self.sim.n_agents)
        for disease in self.rota_diseases:
            infection_counts += disease.infected.astype(int)
        return np.where(infection_counts >= 2)[0]
```

**Implementation Requirements:**
1. **Pre-populate all G,P combinations**: Generate all possible reassortants at sim start
2. **Most start dormant**: `init_prev=0.0` for reassortant strains  
3. **Use set_prognoses()**: Activate dormant strains via existing Starsim mechanism

### Phase 1.4: Performance Optimization for Many Diseases

**Challenge**: 50-100 disease instances with mostly zero prevalence
**Key Insight**: Most reassortant strains remain dormant (`infected.sum() == 0`)

**Optimization Strategies:**
```python
# In disease transmission loops
for disease in sim.diseases.values():
    if isinstance(disease, Rotavirus) and not disease.infected.any():
        continue  # Skip dormant strains
    
    # Only process active diseases
    disease.step_transmission()

# In immunity connector
def _update_cross_immunity(self):
    # Only update rel_sus for diseases with potential exposure
    active_diseases = [d for d in self.rota_diseases if d.infected.any()]
    for disease in active_diseases:
        # Vectorized immunity calculation
        ...

# In analyzers  
def step(self):
    # Batch operations for strain families
    active_strains = {d.strain: d.infected.sum() 
                     for d in self.rota_diseases if d.infected.any()}
```

**Expected Performance Impact:**
- **Dormant strain overhead**: Minimal with proper `infected.any()` checks
- **Active strain processing**: Same as current system
- **Memory usage**: Moderate increase (50-100 disease instances vs. 1)
- **Network transmission**: Only active diseases participate

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