"""
High-performance immunity connector for Rotavirus v2 architecture
Uses bitmask vectorization for cross-strain immunity calculations
"""
# Standard library imports
# (none needed for this module)

# Third-party imports
import numpy as np
import starsim as ss

# Local imports
from .rotavirus import Rotavirus

class PathogenMatch:
    """Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic"""
    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3


class RotaImmunityConnector(ss.Connector):
    """
    High-performance cross-strain immunity connector using bitmask vectorization
    
    This connector automatically detects all Rotavirus disease instances in the
    simulation and manages cross-strain immunity using efficient bitwise operations
    on entire population arrays, avoiding all UID loops.
    """
    
    def __init__(self, pars=None, **kwargs):
        """
        Initialize immunity connector with default parameters
        
        Args:
            pars (dict, optional): Parameters dict to override defaults
            **kwargs: Additional parameters
            
        Default parameters:
            homotypic_immunity_efficacy (0.9): Protection from same G,P strain
            partial_heterotypic_immunity_efficacy (0.5): Protection from shared G or P
            complete_heterotypic_immunity_efficacy (0.3): Protection from different G,P  
            naive_immunity_efficacy (0.0): Baseline immunity for naive individuals
            # full_waning_rate: Immunity waning rate (~273 days mean, 365/273 per year)
            immunity_waning_delay (0 days): Delay before immunity decay starts
            cotransmission_prob (2%): Probability of co-transmitting multiple strains
        """
        super().__init__()
        
        # Define immunity parameters
        self.define_pars(
            homotypic_immunity_efficacy = 0.9,        # Protection from same G,P strain
            partial_heterotypic_immunity_efficacy = 0.5,  # Protection from shared G or P
            complete_heterotypic_immunity_efficacy = 0.3, # Protection from different G,P (or no prior exposure to any strain)
            naive_immunity_efficacy = 0.0,               # Baseline immunity for naive individuals (0.0 = fully susceptible)
            immunity_waning_delay = ss.days(0),               # Time delay before immunity decay starts (years)
            # infection_history_susceptibility_factors = {0: 1, 1: 1, 2: 1, 3: 1},  # Susceptibility scaling based on total infection history. We may want to remove this feature later.
            cotransmission_prob = ss.bernoulli(p=0.02),  # Probability of transmitting all strains instead of dominant strain selection (2%). We may want to remove this feature later.
        )
        
        # Update with user parameters
        self.update_pars(pars=pars, **kwargs)
        
        # Define immunity state arrays

        #  name=name, dtype=ss_int, nan=int_nan, **kwargs)
        self.define_states(
            ss.Arr('exposed_GP_bitmask', dtype=np.int64, nan=ss.dtypes.int_nan, default=0),    # Bitmask of exposed (G,P) pairs
            ss.Arr('exposed_G_bitmask', dtype=np.int64, nan=ss.dtypes.int_nan, default=0),     # Bitmask of exposed G types
            ss.Arr('exposed_P_bitmask', dtype=np.int64, nan=ss.dtypes.int_nan, default=0),     # Bitmask of exposed P types
            ss.FloatArr('oldest_infection', default=np.nan), # Time of first infection (for waning)
            ss.BoolArr('has_immunity', default=False),       # Whether agent has any immunity
            ss.FloatArr('num_recovered_infections', default=0.0),   # Total number of prior infections (for scaling susceptibility)
            ss.FloatArr('num_current_infections', default=0.0),  # Current number of active infections (for coinfection logic)
            ss.FloatArr('homotypic_immunity_decay_factor', default=0.0),  # Decay factor for immunity over time (1.0 = full immunity, 0.0 = none)
        )
        
        # Will be populated during init_post
        self.rota_diseases = []
        self.G_to_bit = {}
        self.P_to_bit = {}
        self.GP_to_bit = {}
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}
        
        # G and P max decay tracking (populated in init_post)
        self.G_max_decayed_immunity_factors = {}  # Will store ss.FloatArr for each G type
        self.P_max_decayed_immunity_factors = {}  # Will store ss.FloatArr for each P type
        
        # Reusable array for decay factor calculations to reduce allocations
        self._temp_decay_array = None

    def init_results(self):
        """Initialize results storage for immunity-related outputs"""
        super().init_results()

        return

    def init_pre(self, sim, force=False):
        """Auto-detect Rotavirus diseases and create states before initialization"""
        # Auto-detect all Rotavirus disease instances
        self.rota_diseases = [d for d in sim.diseases.values() 
                            if isinstance(d, Rotavirus)]
        
        if len(self.rota_diseases) == 0:
            if sim.pars.verbose:
                print("Warning: No Rotavirus diseases found in simulation")
            super().init_pre(sim)
            return
            
        if sim.pars.verbose:
            print(f"RotaImmunityConnector: Found {len(self.rota_diseases)} Rotavirus strains")
        
        # Calculate unique genotypes once and store for use in init_post()
        self.unique_G = sorted(set(d.G for d in self.rota_diseases))
        self.unique_P = sorted(set(d.P for d in self.rota_diseases))
        self.unique_GP = sorted(set((d.G, d.P) for d in self.rota_diseases))
        
        # Create G and P max decay tracking arrays BEFORE calling super().init_pre()
        G_states = []
        P_states = []
        for g in self.unique_G:
            state_name = f'G{g}_max_decayed_immunity_factor'
            state = ss.FloatArr(state_name, default=1.0)
            G_states.append(state)
            self.G_max_decayed_immunity_factors[g] = state
            
        for p in self.unique_P:
            state_name = f'P{p}_max_decayed_immunity_factor'
            state = ss.FloatArr(state_name, default=1.0)
            P_states.append(state)
            self.P_max_decayed_immunity_factors[p] = state
            
        # Define all states at once
        self.define_states(*G_states, *P_states)
        
        
        # NOW call super().init_pre() with states already defined
        super().init_pre(sim)

    def init_post(self):
        """Create bitmask mappings after initialization"""
        super().init_post()
        
        if len(self.rota_diseases) == 0:
            return
        
        # Use the genotypes calculated in init_pre() - no duplication!
        
        # Ensure we don't exceed bitwise limits based on actual datatype
        max_bits = np.iinfo(self.exposed_GP_bitmask.dtype).bits
        if len(self.unique_G) > max_bits or len(self.unique_P) > max_bits or len(self.unique_GP) > max_bits:
            raise ValueError(f"Too many unique genotypes: {len(self.unique_G)} G types, {len(self.unique_P)} P types, {len(self.unique_GP)} GP pairs. Max {max_bits} each. Either increase bitmask dtype size or reduce number of strains.")
        
        # Create mappings: genotype -> bit position
        self.G_to_bit = {g: i for i, g in enumerate(self.unique_G)}
        self.P_to_bit = {p: i for i, p in enumerate(self.unique_P)}
        self.GP_to_bit = {gp: i for i, gp in enumerate(self.unique_GP)}
        
        if self.sim.pars.verbose > 1:
            print(f"  - G genotypes: {self.unique_G} -> bits {list(self.G_to_bit.values())}")
            print(f"  - P genotypes: {self.unique_P} -> bits {list(self.P_to_bit.values())}")
            print(f"  - GP pairs: {self.unique_GP} -> bits {list(self.GP_to_bit.values())}")
        
        # Pre-compute disease-specific bitmasks for fast lookup
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}
        for disease in self.rota_diseases:
            self.disease_G_masks[disease.name] = 1 << self.G_to_bit[disease.G]
            self.disease_P_masks[disease.name] = 1 << self.P_to_bit[disease.P]
            self.disease_GP_masks[disease.name] = 1 << self.GP_to_bit[(disease.G, disease.P)]
        
        if self.sim.pars.verbose > 1:
            print(f"  - Pre-computed bitmasks for {len(self.rota_diseases)} diseases")
        
        
    def step(self):
        """Main connector step: apply waning and update cross-immunity"""
        if len(self.rota_diseases) == 0:
            return
            
        # Update cross-immunity protection for all diseases
        self._update_cross_immunity()
        
    def _update_cross_immunity(self):
        """Fully vectorized cross-immunity using bitwise operations - NO UID LOOPS"""
        # Reset decay factors and update immunity for all diseases
        self._reset_decay_factors()
        self._update_immunity_decay_factors()
        self._calculate_disease_susceptibilities()
        
    def _reset_decay_factors(self):
        """Reset G and P max decay factors to 0.0 each timestep for fresh calculation"""
        for g in self.unique_G:
            self.G_max_decayed_immunity_factors[g][:] = 0.0
        for p in self.unique_P:
            self.P_max_decayed_immunity_factors[p][:] = 0.0
            
    def _update_immunity_decay_factors(self):
        """Update immunity decay factors for all diseases based on recovery times"""
        for disease in self.rota_diseases:
            # Update max decay factors for agents recovered from this specific strain
            recovered_from_strain = (disease.infected==False) & (disease.ti_recovered > 0)
            recovered_uids = recovered_from_strain.uids
            
            if recovered_from_strain.any():
                # Time since recovery from this strain
                time_since_recovery = (disease.ti - disease.ti_recovered[recovered_from_strain]) * disease.dt
                
                # Apply delayed exponential decay
                waning_started = time_since_recovery > disease.pars.waning_delay

                if waning_started.any():
                    waning_started_uids = recovered_uids[waning_started]
                    # Calculate decay factor for agents past the delay period
                    decay_time = time_since_recovery[waning_started] - self.pars.immunity_waning_delay
                    # Use pre-computed decay rates stored when agents recovered
                    decay_rate = disease.waning_rate[waning_started_uids]
                    decay_factor = np.exp(-decay_rate * decay_time) # todo verify decay rate in days, decay time in days when dt is different
                    
                    # Update per-strain decay factor (for homotypic immunity)
                    self.homotypic_immunity_decay_factor[waning_started_uids] = decay_factor
                    
                    # Update G and P max decay factors (take maximum with existing values for strongest immunity)
                    current_G_decay = self.G_max_decayed_immunity_factors[disease.G][waning_started_uids]
                    current_P_decay = self.P_max_decayed_immunity_factors[disease.P][waning_started_uids]
                    
                    self.G_max_decayed_immunity_factors[disease.G][waning_started_uids] = np.maximum(current_G_decay, decay_factor)
                    self.P_max_decayed_immunity_factors[disease.P][waning_started_uids] = np.maximum(current_P_decay, decay_factor)
                    
    def _calculate_disease_susceptibilities(self):
        """Calculate disease susceptibilities based on immunity matching and decay factors"""
        for disease in self.rota_diseases:
            disease_G_mask = self.disease_G_masks[disease.name]
            disease_P_mask = self.disease_P_masks[disease.name]
            disease_GP_mask = self.disease_GP_masks[disease.name]
            
            # Extract raw numpy arrays for bitwise operations
            G_bits = self.exposed_G_bitmask.values
            P_bits = self.exposed_P_bitmask.values
            GP_bits = self.exposed_GP_bitmask.values
            
            # Vectorized matching using bitwise operations
            has_exact_match = (GP_bits & disease_GP_mask) != 0
            has_G_match = (G_bits & disease_G_mask) != 0 & ~has_exact_match
            has_P_match = (P_bits & disease_P_mask) != 0 & ~has_exact_match

            # Determine immunity type and assign protection levels
            has_partial = (has_G_match | has_P_match)
            has_immunity_mask = self.has_immunity[:]  # Agents with any prior immunity
            
            # Separate naive agents (no prior immunity) from true heterotypic matches
            has_complete_hetero = ~has_partial & has_immunity_mask
            
            # Vectorized strain match efficacy using numpy.where with 4 categories. This assigns the correct immunity efficacy based on match type.
            strain_match_immunity_efficacy = np.where(
                has_exact_match, self.pars.homotypic_immunity_efficacy,
                np.where(has_partial, self.pars.partial_heterotypic_immunity_efficacy,
                        np.where(has_complete_hetero, self.pars.complete_heterotypic_immunity_efficacy,
                                self.pars.naive_immunity_efficacy))
            )

            # Calculate appropriate decay factor based on immunity type. Default is 0.0 (no immunity).
            # Optimized: Reuse array to reduce allocations
            if self._temp_decay_array is None or len(self._temp_decay_array) != len(self.sim.people):
                self._temp_decay_array = np.zeros(len(self.sim.people), dtype=float)
            else:
                self._temp_decay_array.fill(0.0)  # Reset to zero faster than creating new array
            final_decayed_immunity_factor = self._temp_decay_array
            
            # Homotypic: use per-strain decay
            final_decayed_immunity_factor[has_exact_match] = self.homotypic_immunity_decay_factor.values[has_exact_match]
            
            # Find least decayed immunity for partial matches
            partial_match_G = has_G_match & ~has_P_match
            partial_match_P = has_P_match & ~has_G_match
            final_decayed_immunity_factor[partial_match_G] = self.G_max_decayed_immunity_factors[disease.G].values[partial_match_G]
            final_decayed_immunity_factor[partial_match_P] = self.P_max_decayed_immunity_factors[disease.P].values[partial_match_P]

            # For partial matches coming from both G and P, take the maximum decay from either G or P.
            g_and_p_match = has_G_match & has_P_match # homotypic has already been filtered out, so we can safely use & here
            final_decayed_immunity_factor[g_and_p_match] = np.maximum(
                self.G_max_decayed_immunity_factors[disease.G].values[g_and_p_match],
                self.P_max_decayed_immunity_factors[disease.P].values[g_and_p_match]
            )

            # Apply protection with appropriate decay factor
            # People without immunity have full susceptibility (rel_sus = 1.0)

            # The full rel_sus calculation combines strain match efficacy, decayed immunity factor, and infection history scaling.
            # * strain_match_immunity_efficacy is the base protection level based on match type (0.0 to 1.0)
            # * final_decay_factor reduces this protection over time since last infection (0.0 to 1.0). In the case of a partial match, it uses the max decay from either G or P.
            # * infection_history_susceptibility_factor scales susceptibility based on total prior infections. It does not decay over time.


            disease.rel_sus[:] = (1- strain_match_immunity_efficacy * final_decayed_immunity_factor)

    def record_infection(self, disease, new_infected_uids):
        self.num_current_infections[new_infected_uids] += 1.0
    
    def record_recovery(self, disease, recovered_uids):
        """
        Update bitmasks when people recover from infections
        
        This method should be called by Rotavirus instances when infections resolve.
        
        Args:
            disease: Rotavirus disease instance
            recovered_uids: Array of UIDs who recovered from this disease
        """
        if len(recovered_uids) == 0:
            return
            
        if not isinstance(disease, Rotavirus):
            return  # Only handle Rotavirus diseases
            
        # Get bit positions for this disease's G,P genotypes and combination
        G_bit = 1 << self.G_to_bit[disease.G]  
        P_bit = 1 << self.P_to_bit[disease.P]
        GP_bit = 1 << self.GP_to_bit[(disease.G, disease.P)]
        
        # Update bitmasks using IntArr bitwise ops
        current_G = self.exposed_G_bitmask[recovered_uids]
        current_P = self.exposed_P_bitmask[recovered_uids]
        current_GP = self.exposed_GP_bitmask[recovered_uids]
        
        self.exposed_G_bitmask[recovered_uids] = current_G | G_bit
        self.exposed_P_bitmask[recovered_uids] = current_P | P_bit
        self.exposed_GP_bitmask[recovered_uids] = current_GP | GP_bit
        
        # Mark as having immunity
        self.has_immunity[recovered_uids] = True
        
        # Increment total infection count for each recovered agent
        self.num_current_infections[recovered_uids] -= 1.0
        self.num_recovered_infections[recovered_uids] += 1.0
        
        # Track oldest infection time (only set if first infection)
        first_infections = np.isnan(self.oldest_infection[recovered_uids])
        self.oldest_infection[recovered_uids[first_infections]] = self.sim.ti
        
    
    # @staticmethod
    # def match_strain(strain1, strain2):
    #     """
    #     Determine genetic match type between two strains
    #
    #     Args:
    #         strain1: Tuple (G,P) or Rotavirus instance
    #         strain2: Tuple (G,P) or Rotavirus instance
    #
    #     Returns:
    #         PathogenMatch: HOMOTYPIC, PARTIAL_HETERO, or COMPLETE_HETERO
    #     """
    #     # Extract G,P tuples
    #     if isinstance(strain1, Rotavirus):
    #         gp1 = strain1.strain
    #     else:
    #         gp1 = strain1[:2]  # Assume tuple format
    #
    #     if isinstance(strain2, Rotavirus):
    #         gp2 = strain2.strain
    #     else:
    #         gp2 = strain2[:2]  # Assume tuple format
    #
    #     # Compare G,P genotypes
    #     if gp1 == gp2:
    #         return PathogenMatch.HOMOTYPIC
    #     elif gp1[0] == gp2[0] or gp1[1] == gp2[1]:  # Shared G or P
    #         return PathogenMatch.PARTIAL_HETERO
    #     else:
    #         return PathogenMatch.COMPLETE_HETERO