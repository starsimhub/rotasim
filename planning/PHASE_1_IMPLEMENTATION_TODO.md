# Phase 1 Implementation TODO - Rotasim v2 Architecture

## Overview
Implement core infrastructure for strain-specific Rotavirus disease instances with vectorized cross-strain immunity.

## Branch: `rotasim_architecture`
## Architecture Plan: `planning/rotasim_v2_architecture_plan.md`

---

## TASK 1: Core Rotavirus Disease Class
**File**: `rotasim/diseases/rotavirus.py` (new file)
**Estimated Time**: 2-3 hours

### 1.1 Basic Class Structure
```python
class Rotavirus(ss.Infection):
    def __init__(self, G, P, backbone=(1, 1), name=None, **kwargs):
        # Auto-generate name: "G1P8_11"
        # Store G, P, backbone attributes
        # Call super().__init__()
```

### 1.2 Required Properties
- `@property def strain(self)`: Return `(self.G, self.P) + self.backbone` for compatibility
- `@property def antigenic_segments(self)`: Return `(self.G, self.P)`
- `def match_strain(self, other)`: Return HOMOTYPIC/PARTIAL_HETERO/COMPLETE_HETERO

### 1.3 Parameters to Define
- `beta`: Strain-specific transmission rate
- `init_prev`: Initial prevalence for this strain
- Standard `ss.Infection` parameters

### 1.4 Testing
- Create simple test: single Rotavirus instance transmits correctly
- Verify name auto-generation works
- Test strain matching logic

---

## TASK 2: High-Performance Immunity Connector  
**File**: `rotasim/connectors/immunity.py` (new file)
**Estimated Time**: 4-5 hours

### 2.1 Connector Structure
```python
class RotaImmunityConnector(ss.Connector):
    def __init__(self, **kwargs):
        # Define immunity parameters
        
    def init_post(self, sim):
        # Auto-detect Rotavirus instances
        # Create G,P to bit mappings
        # Pre-compute disease-specific bitmasks
        
    def define_states(self):
        # ss.FloatArr('exposed_G_bitmask', default=0.0)
        # ss.FloatArr('exposed_P_bitmask', default=0.0) 
        # ss.FloatArr('oldest_infection', default=np.nan)
        # ss.BoolArr('has_immunity', default=False)
```

### 2.2 Vectorized Immunity Logic
- `_apply_waning()`: Vectorized Poisson waning, clear oldest immunity first
- `_update_cross_immunity()`: Bitwise operations, no UID loops
- `_calculate_protection_vectorized()`: numpy.where for homotypic/partial/complete

### 2.3 Integration Method
- `record_recovery(disease, recovered_uids)`: Update bitmasks when people recover
- Called by Rotavirus instances when infections resolve

### 2.4 Testing
- Test bitmask operations work correctly
- Verify immunity rates match current system
- Performance test vs. current implementation

---

## TASK 3: Multi-Strain Simulation Integration
**File**: `rotasim/rotasim.py` (update existing)
**Estimated Time**: 2 hours

### 3.1 Update Sim Class
- Modify to accept list of Rotavirus instances
- Add RotaImmunityConnector to connectors
- Update example usage patterns

### 3.2 Integration Testing
- Create 2-3 strain system: G1P8, G2P4, G3P6
- Verify independent transmission
- Test cross-immunity protection
- Compare results with current monolithic model

---

## TASK 4: Result Compatibility & Validation
**File**: `rotasim/analyzers/compatibility.py` (new file)  
**Estimated Time**: 3-4 hours

### 4.1 Strain Count Analyzer
- Aggregate infection counts across all Rotavirus instances
- Output CSV format matching current `rota_strain_count_*.csv`
- Handle strain naming consistency

### 4.2 Validation Tests
- Run parallel simulations: current v1 vs. new v2
- Compare key metrics: prevalence curves, immunity dynamics
- Verify CSV outputs match format and content

### 4.3 Performance Benchmarks
- Time comparison: monolithic vs. multi-instance + bitmask vectorization
- Memory usage analysis
- Scalability test with increasing number of strains

---

## TASK 5: Documentation & Examples
**Estimated Time**: 1-2 hours

### 5.1 Update Documentation
- Add Phase 1 examples to `CLAUDE.md`
- Create simple usage examples
- Document performance improvements

### 5.2 Test Scripts
- Create `tests/test_v2_basic.py`: Basic functionality tests
- Create `examples/phase1_demo.py`: Demonstration script
- Update existing test suite to run on both v1 and v2

---

## KEY IMPLEMENTATION NOTES

### Performance Critical Points
1. **Bitmask Operations**: Use `astype(int)` for bitwise ops, `astype(float)` to store in FloatArr
2. **Vectorization**: Avoid all UID loops in immunity connector
3. **Memory**: 2 floats per agent vs. potentially many boolean arrays

### Integration Points
1. **Recovery Integration**: Rotavirus instances must call `immunity_connector.record_recovery()`
2. **Auto-Detection**: Connectors find Rotavirus instances automatically via `isinstance()`
3. **State Management**: Leverage Starsim's automatic birth/death handling for arrays

### Validation Criteria
1. **Exact Behavioral Match**: New system produces identical results to current system
2. **Performance Improvement**: Faster execution, especially with many strains
3. **CSV Compatibility**: Existing analysis scripts work unchanged

### Questions for Starsim Core Team
1. **Dynamic Disease Creation**: Can new Rotavirus instances be added during simulation for reassortment?
2. **Connector Best Practices**: Optimal patterns for auto-detecting disease instances?
3. **Performance Guidelines**: Recommended approaches for large-scale multi-disease systems?

---

## SUCCESS CRITERIA FOR PHASE 1
- [ ] Single Rotavirus strain transmits correctly with proper naming
- [ ] Multi-strain system (2-3 strains) runs without interference  
- [ ] Cross-immunity protection matches current immunity rates exactly
- [ ] Bitmask vectorization shows performance improvement over current system
- [ ] CSV outputs maintain exact format compatibility
- [ ] All tests pass for both individual components and integrated system

## NEXT PHASES (Preview)
- **Phase 2**: Dynamic strain creation via RotaReassortment connector
- **Phase 3**: Enhanced vaccination system integration
- **Phase 4**: Complete analyzer suite for all CSV outputs
- **Phase 5**: Migration utilities and advanced features

---

**Created**: 2025-01-28  
**Branch**: `rotasim_architecture`  
**Dependencies**: Updated architecture plan in `planning/rotasim_v2_architecture_plan.md`