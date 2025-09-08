"""
High-performance immunity connector for Rotavirus v2 architecture
Uses bitmask vectorization for cross-strain immunity calculations
"""
import numpy as np
import starsim as ss
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
        """Initialize immunity connector with default parameters"""
        super().__init__()
        
        # Define immunity parameters
        self.define_pars(
            # TODO: Rename to *_immunity_efficacy - these are efficacy values, not rates
            homotypic_immunity_rate = 0.5,        # Protection from same G,P strain
            partial_heterotypic_immunity_rate = 0.5,  # Protection from shared G or P
            complete_heterotypic_immunity_rate = 0.5, # Protection from different G,P
            waning_rate = ss.perday(1/273),           # Rate of immunity waning (omega parameter, ~273 days)
            infection_history_rel_sus_mapping = {0: 1.0, 1: 0.61, 2: 0.48, 3: 0.33},  # Susceptibility scaling based on total infection history
            cotransmission_prob = ss.bernoulli(p=0.02),  # Probability of transmitting all strains instead of dominant strain selection (2%)
        )
        
        # Update with user parameters
        self.update_pars(pars=pars, **kwargs)
        
        # Define immunity state arrays
        self.define_states(
            ss.FloatArr('exposed_GP_bitmask', default=0.0),  # Bitmask of exposed (G,P) pairs
            ss.FloatArr('exposed_G_bitmask', default=0.0),   # Bitmask of exposed G types
            ss.FloatArr('exposed_P_bitmask', default=0.0),   # Bitmask of exposed P types  
            ss.FloatArr('oldest_infection', default=np.nan), # Time of first infection (for waning)
            ss.BoolArr('has_immunity', default=False),       # Whether agent has any immunity
            ss.FloatArr('num_infections', default=0.0),   # Total number of prior infections (for scaling susceptibility)
        )
        
        # Will be populated during init_post
        self.rota_diseases = []
        self.G_to_bit = {}
        self.P_to_bit = {}
        self.GP_to_bit = {}
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}

    def init_results(self):
        """Initialize results storage for immunity-related outputs"""
        super().init_results()

        self.define_results(ss.Result('n_waned', dtype=int, scale=True, summarize_by='sum', label='Number of people losing immunity each timestep'))
        return

    def init_post(self):
        """Auto-detect Rotavirus diseases and create bitmask mappings"""
        super().init_post()
        
        # Auto-detect all Rotavirus disease instances
        self.rota_diseases = [d for d in self.sim.diseases.values() 
                            if isinstance(d, Rotavirus)]
        
        if len(self.rota_diseases) == 0:
            print("Warning: No Rotavirus diseases found in simulation")
            return
            
        print(f"RotaImmunityConnector: Found {len(self.rota_diseases)} Rotavirus strains")
        
        # Create bit mappings for unique G,P types and combinations in simulation
        unique_G = sorted(set(d.G for d in self.rota_diseases))
        unique_P = sorted(set(d.P for d in self.rota_diseases))
        unique_GP = sorted(set((d.G, d.P) for d in self.rota_diseases))
        
        # Ensure we don't exceed bitwise limits (32/64 bits)
        max_bits = 32  # Conservative limit
        if len(unique_G) > max_bits or len(unique_P) > max_bits or len(unique_GP) > max_bits:
            raise ValueError(f"Too many unique genotypes: {len(unique_G)} G types, {len(unique_P)} P types, {len(unique_GP)} GP pairs. Max {max_bits} each.")
        
        # Create mappings: genotype -> bit position
        self.G_to_bit = {g: i for i, g in enumerate(unique_G)}
        self.P_to_bit = {p: i for i, p in enumerate(unique_P)}
        self.GP_to_bit = {gp: i for i, gp in enumerate(unique_GP)}
        
        print(f"  - G genotypes: {unique_G} -> bits {list(self.G_to_bit.values())}")
        print(f"  - P genotypes: {unique_P} -> bits {list(self.P_to_bit.values())}")
        print(f"  - GP pairs: {unique_GP} -> bits {list(self.GP_to_bit.values())}")
        
        # Pre-compute disease-specific bitmasks for fast lookup
        self.disease_G_masks = {}
        self.disease_P_masks = {}
        self.disease_GP_masks = {}
        for disease in self.rota_diseases:
            self.disease_G_masks[disease.name] = 1 << self.G_to_bit[disease.G]
            self.disease_P_masks[disease.name] = 1 << self.P_to_bit[disease.P]
            self.disease_GP_masks[disease.name] = 1 << self.GP_to_bit[(disease.G, disease.P)]
            
        print(f"  - Pre-computed bitmasks for {len(self.rota_diseases)} diseases")
        
        
    def step(self):
        """Main connector step: apply waning and update cross-immunity"""
        if len(self.rota_diseases) == 0:
            return
            
        # 1. Apply vectorized immunity waning
        self._apply_waning()
        
        # 2. Update cross-immunity protection for all diseases
        self._update_cross_immunity()
        
        # 3. Apply fitness-weighted transmission for coinfected agents
        self._apply_fitness_weighted_transmission()
        
    def _apply_waning(self):
        """Vectorized immunity waning - clear entire immunity portfolios based on oldest infection"""
        if not self.has_immunity.any():
            return
            
        # Number of people to lose immunity this timestep (Poisson process)
        n_waning = np.random.poisson(self.pars.waning_rate * self.t.dt * self.has_immunity.sum())
        
        if n_waning > 0:
            # Get UIDs of people who have immunity
            immune_uids = self.has_immunity.uids
            
            # Find oldest infections (longest time since first infection)
            oldest_times = self.oldest_infection[immune_uids]
            
            # Select n_waning people with oldest infections
            waning_indices = np.argpartition(oldest_times, min(n_waning, len(oldest_times) - 1))[:n_waning]
            waning_uids = immune_uids[waning_indices]
            
            # Clear all immunity bitmasks for these people
            self.exposed_GP_bitmask[waning_uids] = 0.0
            self.exposed_G_bitmask[waning_uids] = 0.0
            self.exposed_P_bitmask[waning_uids] = 0.0
            self.has_immunity[waning_uids] = False
            self.oldest_infection[waning_uids] = np.nan
            
            if len(waning_uids) > 0:
                print(f"  Immunity waning: {len(waning_uids)} people lost all immunity")
            self.results.n_waned[self.ti] += len(waning_uids)

            for disease in self.rota_diseases:
                disease.susceptible[waning_uids] = True
                
    def _update_cross_immunity(self):
        """Fully vectorized cross-immunity using bitwise operations - NO UID LOOPS"""
        # Only process people who have immunity (performance optimization)
        if not self.has_immunity.any():
            return
            
        immune_mask = self.has_immunity
        
        for disease in self.rota_diseases:
            disease_G_mask = self.disease_G_masks[disease.name]
            disease_P_mask = self.disease_P_masks[disease.name]
            disease_GP_mask = self.disease_GP_masks[disease.name]
            
            # Convert FloatArr to int for bitwise operations
            G_bits = self.exposed_G_bitmask.astype(int)
            P_bits = self.exposed_P_bitmask.astype(int)
            GP_bits = self.exposed_GP_bitmask.astype(int)
            
            # Vectorized matching using bitwise operations
            has_exact_match = (GP_bits & disease_GP_mask) != 0
            has_G_match = (G_bits & disease_G_mask) != 0
            has_P_match = (P_bits & disease_P_mask) != 0
            
            # Determine immunity type for entire population
            has_homotypic = has_exact_match
            has_partial = (has_G_match | has_P_match) & ~has_homotypic
            
            # Vectorized protection assignment using numpy.where
            strain_protection = np.where(
                has_homotypic, self.pars.homotypic_immunity_rate,
                np.where(has_partial, self.pars.partial_heterotypic_immunity_rate,
                        self.pars.complete_heterotypic_immunity_rate)
            )
            
            # Get infection history susceptibility scaling factor
            infection_history_sus_factor = np.ones(len(self.sim.people), dtype=float)
            for n_inf, scalar in self.pars.infection_history_rel_sus_mapping.items():
                if n_inf < 3:
                    mask = (self.num_infections == n_inf)
                else:  # 3+ infections use same scalar as 3
                    mask = (self.num_infections >= 3)
                infection_history_sus_factor[mask] = scalar
            
            # Apply protection only to people with immunity
            # People without immunity have full susceptibility (rel_sus = 1.0)
            disease.rel_sus[:] = (1.0 - (strain_protection * infection_history_sus_factor * immune_mask))
    
    def _apply_strain_selection(self):
        """
        Apply strain selection for coinfected transmission using vectorized operations

        Most of the time we want to transmit only a single strain of the virus, even if the host
        is coinfected with multiple strains. We accomplish this by adjusting the relative transmissibility
        of the various strains for each host. There is also a possibility that multiple strains will transmit.

        """
        # Build infection matrix and fitness values
        infection_matrix = np.zeros((len(self.sim.people), len(self.rota_diseases)), dtype=bool)
        fitness_values = np.zeros(len(self.rota_diseases), dtype=float)
        
        for i, disease in enumerate(self.rota_diseases):
            infection_matrix[:, i] = disease.infected[:]
            fitness_values[i] = disease.pars.beta.mean()  # Use beta as fitness metric
        
        # Find coinfected agents (>1 strain)
        infections_per_agent = infection_matrix.sum(axis=1)
        coinfected_uids = np.where(infections_per_agent > 1)[0]
        
        if len(coinfected_uids) == 0:
            return  # No coinfected agents
        
        # Filter out cotransmission agents
        cotransmission_mask = self.pars.cotransmission_prob.rvs(coinfected_uids)
        strain_selection_uids = coinfected_uids[~cotransmission_mask]
        
        if len(strain_selection_uids) == 0:
            return  # All coinfected agents are cotransmission
        
        # Vectorized strain selection
        coinfected_infections = infection_matrix[strain_selection_uids, :]  # Shape: (n_agents, n_strains)
        
        # For each agent, find maximum fitness among their infected strains
        agent_fitness_matrix = coinfected_infections * fitness_values[np.newaxis, :]  # Broadcast fitness
        agent_fitness_matrix[~coinfected_infections] = -np.inf  # Mask uninfected strains
        max_fitness_per_agent = np.max(agent_fitness_matrix, axis=1, keepdims=True)
        
        # Create mask for dominant strains (fitness == max_fitness for each agent)
        dominant_mask = (agent_fitness_matrix == max_fitness_per_agent) & coinfected_infections
        
        # Randomly select one strain among tied dominant strains per agent
        selected_strains = np.zeros_like(dominant_mask, dtype=bool)
        for i, uid in enumerate(strain_selection_uids):
            dominant_indices = np.where(dominant_mask[i, :])[0]
            if len(dominant_indices) > 0:
                selected_idx = np.random.choice(dominant_indices)
                selected_strains[i, selected_idx] = True
        
        # Set rel_trans: 0 for non-selected strains, 1 for selected strains
        for strain_idx, disease in enumerate(self.rota_diseases):
            # Reset rel_trans to 1 for all strain_selection_uids first
            disease.rel_trans[ss.uids(strain_selection_uids)] = 1.0
            
            # Set to 0 for infected but non-selected strains
            infected_but_not_selected = coinfected_infections[:, strain_idx] & ~selected_strains[:, strain_idx]
            disease.rel_trans[ss.uids(strain_selection_uids[infected_but_not_selected])] = 0.0
    
    def _apply_fitness_weighted_transmission(self):
        """
        Apply fitness-weighted transmission for coinfected agents using vectorized operations
        
        For coinfected agents, set rel_trans for all strains such that their sum equals 1,
        with weights proportional to fitness (beta values). This allows multiple strains
        to transmit but with probabilities based on their relative fitness.
        """
        # Build infection matrix and fitness values
        infection_matrix = np.zeros((len(self.sim.people), len(self.rota_diseases)), dtype=bool)
        fitness_values = np.zeros(len(self.rota_diseases), dtype=float)
        
        for i, disease in enumerate(self.rota_diseases):
            infection_matrix[:, i] = disease.infected[:]
            fitness_values[i] = disease.pars.beta.mean()  # Use beta as fitness metric
        
        # Find coinfected agents (>1 strain)
        infections_per_agent = infection_matrix.sum(axis=1)
        coinfected_uids = np.where(infections_per_agent > 1)[0]
        
        if len(coinfected_uids) == 0:
            return  # No coinfected agents
        
        # Filter out cotransmission agents
        cotransmission_mask = self.pars.cotransmission_prob.rvs(coinfected_uids)
        fitness_weighted_uids = coinfected_uids[~cotransmission_mask]
        
        if len(fitness_weighted_uids) == 0:
            return  # All coinfected agents are cotransmission
        
        # Vectorized fitness-weighted transmission
        coinfected_infections = infection_matrix[fitness_weighted_uids, :]  # Shape: (n_agents, n_strains)
        
        # Calculate fitness weights for each agent's infected strains
        agent_fitness_matrix = coinfected_infections * fitness_values[np.newaxis, :]  # Broadcast fitness
        agent_fitness_matrix[~coinfected_infections] = 0.0  # Zero out uninfected strains
        
        # Calculate sum of fitness for each coinfected agent
        fitness_sums = agent_fitness_matrix.sum(axis=1, keepdims=True)  # Shape: (n_agents, 1)
        fitness_sums[fitness_sums == 0] = 1.0  # Avoid division by zero
        
        # Normalize to sum to 1 for each agent
        normalized_weights = agent_fitness_matrix / fitness_sums  # Shape: (n_agents, n_strains)
        
        # Set rel_trans for each strain based on normalized weights
        for strain_idx, disease in enumerate(self.rota_diseases):
            # Get normalized weights for this strain
            strain_weights = normalized_weights[:, strain_idx]
            
            # Only update agents that are infected with this strain
            infected_with_strain = coinfected_infections[:, strain_idx]
            uids_to_update = fitness_weighted_uids[infected_with_strain]
            weights_to_apply = strain_weights[infected_with_strain]
            
            if len(uids_to_update) > 0:
                disease.rel_trans[ss.uids(uids_to_update)] = weights_to_apply
    
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
        
        # Update bitmasks with type conversion (FloatArr bitwise ops)
        current_G = self.exposed_G_bitmask[recovered_uids].astype(int)
        current_P = self.exposed_P_bitmask[recovered_uids].astype(int)
        current_GP = self.exposed_GP_bitmask[recovered_uids].astype(int)
        
        self.exposed_G_bitmask[recovered_uids] = (current_G | G_bit).astype(float)
        self.exposed_P_bitmask[recovered_uids] = (current_P | P_bit).astype(float)
        self.exposed_GP_bitmask[recovered_uids] = (current_GP | GP_bit).astype(float)
        
        # Mark as having immunity
        self.has_immunity[recovered_uids] = True
        
        # Increment total infection count for each recovered agent
        self.num_infections[recovered_uids] += 1.0
        
        # Track oldest infection time (only set if first infection)
        first_infections = np.isnan(self.oldest_infection[recovered_uids])
        self.oldest_infection[recovered_uids[first_infections]] = self.sim.ti
        
        print(f"  Immunity update: {len(recovered_uids)} people recovered from {disease.name}")
    
    @staticmethod
    def match_strain(strain1, strain2):
        """
        Determine genetic match type between two strains
        
        Args:
            strain1: Tuple (G,P) or Rotavirus instance  
            strain2: Tuple (G,P) or Rotavirus instance
            
        Returns:
            PathogenMatch: HOMOTYPIC, PARTIAL_HETERO, or COMPLETE_HETERO
        """
        # Extract G,P tuples
        if isinstance(strain1, Rotavirus):
            gp1 = strain1.strain
        else:
            gp1 = strain1[:2]  # Assume tuple format
            
        if isinstance(strain2, Rotavirus):
            gp2 = strain2.strain  
        else:
            gp2 = strain2[:2]  # Assume tuple format
            
        # Compare G,P genotypes
        if gp1 == gp2:
            return PathogenMatch.HOMOTYPIC
        elif gp1[0] == gp2[0] or gp1[1] == gp2[1]:  # Shared G or P
            return PathogenMatch.PARTIAL_HETERO
        else:
            return PathogenMatch.COMPLETE_HETERO