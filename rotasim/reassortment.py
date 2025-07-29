"""
RotaReassortmentConnector - Handles genetic reassortment in multi-strain rotavirus simulations

Implements the v1 reassortment logic using v2 architecture:
- Per-host probability draws instead of population-level Poisson
- Activates pre-populated dormant diseases instead of dynamic creation
- Uses vectorized operations for performance
"""
import itertools
import numpy as np
import starsim as ss


class RotaReassortmentConnector(ss.Connector):
    """
    Connector for rotavirus genetic reassortment between co-infected strains
    
    This connector replicates the v1 reassortment logic where co-infected hosts
    can generate reassortant strains by mixing G,P antigenic segments from 
    their infecting strains. All possible reassortants are pre-populated as
    dormant diseases and activated when reassortment occurs.
    
    The v1 algorithm:
    1. Find co-infected hosts (≥2 rotavirus strains)  
    2. Draw reassortment events using population-level Poisson
    3. For selected hosts, generate G,P combinations excluding parents
    4. Infect with all valid reassortants
    
    The v2 algorithm:
    1. Find co-infected hosts (≥2 rotavirus diseases active)
    2. Per-host Bernoulli draws based on reassortment rate  
    3. For hosts with reassortment, generate G,P combinations excluding parents
    4. Activate dormant diseases using set_prognoses
    """
    
    def __init__(self, reassortment_rate=0.1, **kwargs):
        """
        Initialize reassortment connector
        
        Args:
            reassortment_rate: Daily probability of reassortment per co-infected host (default: 0.1)
            **kwargs: Additional arguments passed to ss.Connector
        """
        super().__init__(**kwargs)
        
        # Define parameters
        self.define_pars(
            reassortment_rate = ss.rate_prob(reassortment_rate),  # Convert to daily probability
        )
        
        # Will be populated during initialization
        self._rotavirus_diseases = []  # List of Rotavirus disease instances
        self._gp_to_disease = {}       # Mapping: (G,P) → disease instance
        self._disease_to_gp = {}       # Mapping: disease instance → (G,P)
        
    def init_pre(self, sim):
        """Initialize before simulation starts - detect Rotavirus diseases"""
        super().init_pre(sim)
        
        # Find all Rotavirus disease instances
        self._rotavirus_diseases = []
        self._gp_to_disease = {}
        self._disease_to_gp = {}
        
        for disease in sim.diseases.values():
            # Check if this is a Rotavirus disease by looking for G,P attributes
            if hasattr(disease, 'G') and hasattr(disease, 'P'):
                gp_tuple = (disease.G, disease.P)
                self._rotavirus_diseases.append(disease)
                self._gp_to_disease[gp_tuple] = disease
                self._disease_to_gp[disease] = gp_tuple
        
        n_diseases = len(self._rotavirus_diseases)
        if n_diseases == 0:
            raise ValueError("RotaReassortmentConnector requires at least one Rotavirus disease")
        
        print(f"RotaReassortmentConnector: Found {n_diseases} Rotavirus diseases")
        if n_diseases >= 10:
            print(f"  First 5: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases[:5]]}")
            print(f"  Last 5: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases[-5:]]}")
        else:
            print(f"  All strains: {[f'G{d.G}P{d.P}' for d in self._rotavirus_diseases]}")
        
        print(f"  Reassortment rate: {self.pars.reassortment_rate} per day per co-infected host")
        
    def step(self):
        """
        Perform reassortment step - called each timestep
        
        Algorithm:
        1. Find co-infected hosts (agents with ≥2 active Rotavirus infections)
        2. For each co-infected host, draw Bernoulli to determine if reassortment occurs
        3. For hosts with reassortment, generate all valid G,P combinations from parent strains
        4. Exclude parent combinations (already present)
        5. Activate dormant diseases for valid reassortant combinations
        """
        
        # Step 1: Find co-infected hosts using vectorized operations
        co_infected_uids = self._get_coinfected_hosts()
        
        if len(co_infected_uids) == 0:
            return  # No co-infections, no reassortment possible
        
        # Step 2: Per-host Bernoulli draws for reassortment events
        reassortment_events = ss.bernoulli(self.pars.reassortment_rate, len(co_infected_uids))
        reassorting_uids = co_infected_uids[reassortment_events]
        
        if len(reassorting_uids) == 0:
            return  # No reassortment events this timestep
        
        n_events = len(reassorting_uids)
        n_coinfected = len(co_infected_uids)
        print(f"Reassortment: {n_events}/{n_coinfected} co-infected hosts reassorting")
        
        # Step 3-5: For each reassorting host, generate and activate reassortants
        total_new_infections = 0
        for uid in reassorting_uids:
            new_infections = self._reassort_host(uid)
            total_new_infections += new_infections
        
        if total_new_infections > 0:
            print(f"  → {total_new_infections} new reassortant infections created")
    
    def _get_coinfected_hosts(self):
        """
        Find hosts with ≥2 active Rotavirus infections using vectorized operations
        
        Returns:
            np.array: UIDs of co-infected hosts
        """
        # Count active rotavirus infections per host
        infection_counts = np.zeros(len(self.sim.people), dtype=int)
        
        for disease in self._rotavirus_diseases:
            # Add 1 for each host infected with this disease
            infected_uids = disease.infected.uids  # Get UIDs of infected agents
            infection_counts[infected_uids] += 1
        
        # Find hosts with ≥2 infections
        coinfected_mask = infection_counts >= 2
        coinfected_uids = np.where(coinfected_mask)[0]
        
        return coinfected_uids
    
    def _reassort_host(self, uid):
        """
        Perform reassortment for a single co-infected host
        
        Args:
            uid: Host agent ID
            
        Returns:
            int: Number of new reassortant infections created
        """
        # Find which diseases are currently infecting this host
        active_diseases = []
        for disease in self._rotavirus_diseases:
            if disease.infected[uid]:  # Check if this disease is active in this host
                active_diseases.append(disease)
        
        if len(active_diseases) < 2:
            return 0  # Should not happen due to filtering, but safety check
        
        # Get G,P genotypes from active parent strains
        parent_gps = [self._disease_to_gp[disease] for disease in active_diseases]
        
        # Generate all possible G,P reassortant combinations
        G_variants = {gp[0] for gp in parent_gps}  # Unique G genotypes
        P_variants = {gp[1] for gp in parent_gps}  # Unique P genotypes
        
        # Cartesian product of G × P variants
        all_combinations = list(itertools.product(G_variants, P_variants))
        
        # Exclude parent combinations (already present in this host)
        reassortant_combinations = [gp for gp in all_combinations if gp not in parent_gps]
        
        if len(reassortant_combinations) == 0:
            return 0  # No new combinations possible
        
        # Activate dormant diseases for valid reassortants
        new_infections = 0
        for G, P in reassortant_combinations:
            reassortant_disease = self._gp_to_disease.get((G, P))
            if reassortant_disease is not None:
                # Check if already infected with this reassortant
                if not reassortant_disease.infected[uid]:
                    # Activate infection using Starsim method
                    reassortant_disease.set_prognoses(uid, 'infected', value=True)
                    new_infections += 1
                    
        return new_infections


# Make importable from package root
__all__ = ['RotaReassortmentConnector']