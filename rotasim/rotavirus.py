"""
Rotavirus disease class for v2 architecture
Individual strain-specific disease instances following traditional Starsim patterns
"""
import starsim as ss


class PathogenMatch:
    """Define whether pathogens are completely heterotypic, partially heterotypic, or homotypic"""
    COMPLETE_HETERO = 1
    PARTIAL_HETERO = 2
    HOMOTYPIC = 3


class Rotavirus(ss.Infection):
    """
    Individual rotavirus strain as a separate disease instance
    
    Each Rotavirus instance represents a specific G,P combination that behaves
    as an independent disease in the simulation, with cross-strain interactions
    handled by connector classes.
    """
    
    def __init__(self, G, P, pars=None, **kwargs):
        """
        Initialize Rotavirus strain
        
        Args:
            G (int): G genotype (antigenic segment)
            P (int): P genotype (antigenic segment) 
            pars (dict, optional): Parameters dict
            **kwargs: Standard ss.Infection parameters (init_prev, beta, name, etc.)
        """
        # Store G,P genotypes as attributes
        self.G = G
        self.P = P
        
        # Auto-generate name if not provided
        if 'name' not in kwargs:
            kwargs['name'] = f"G{G}P{P}"
            
        super().__init__()
        
        self.define_pars(
            init_prev = ss.bernoulli(p=0.01),     # Initial prevalence
            beta = ss.rate_prob(0.1),             # Transmission rate (will be modified by fitness)
            dur_inf = ss.lognorm_ex(mean=7),      # Duration of infection (~7 days)
        )
        
        self.update_pars(pars=pars, **kwargs)
        
    @property
    def strain(self):
        """Return strain tuple for compatibility with existing code"""
        return (self.G, self.P)