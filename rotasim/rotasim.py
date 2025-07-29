"""
Rotasim convenience class for multi-strain simulations
Provides an easy-to-use interface for researchers while maintaining flexibility
"""
import starsim as ss
from .utils import create_strain_diseases, validate_initial_strains, list_fitness_scenarios
from .immunity import RotaImmunityConnector
from .reassortment import RotaReassortmentConnector


class Rotasim(ss.Sim):
    """
    Convenience class for multi-strain Rotavirus simulations
    
    This class automatically generates all possible strain reassortants from initial strains
    and applies fitness multipliers. Standard ss.Sim arguments like connectors and analyzers
    can be passed as kwargs.
    
    Example usage:
        # Simple case
        sim = Rotasim(initial_strains=[(1,8), (2,4)])
        sim.run()
        
        # Custom fitness scenario
        sim = Rotasim(initial_strains=[(1,8), (2,4)], 
                      fitness_scenario='high_diversity',
                      base_beta=0.15)
        
        # Custom connectors and analyzers
        sim = Rotasim(initial_strains=[(1,8), (2,4)], 
                      connectors=[RotaImmunityConnector()],
                      analyzers=[MyAnalyzer()])
    """
    
    def __init__(self, initial_strains, fitness_scenario='baseline', base_beta=0.1, **kwargs):
        """
        Initialize Rotasim simulation
        
        Args:
            initial_strains: List of (G,P) tuples representing starting strains, e.g. [(1,8), (2,4)]
            fitness_scenario: String name of built-in scenario or dict of custom fitness multipliers
            base_beta: Base transmission rate before fitness adjustment (default: 0.1)
            **kwargs: Additional arguments passed to ss.Sim (n_agents, start, stop, connectors, analyzers, etc.)
            
        Raises:
            ValueError: If initial_strains format is invalid or fitness_scenario is unknown
        """
        # Validate inputs
        validate_initial_strains(initial_strains)
        
        # Store configuration for reference
        self._initial_strains = initial_strains
        self._fitness_scenario = fitness_scenario  
        self._base_beta = base_beta
        
        print("Rotasim: Setting up multi-strain simulation")
        print(f"  Initial strains: {initial_strains}")
        print(f"  Fitness scenario: {fitness_scenario if isinstance(fitness_scenario, str) else 'custom'}")
        print(f"  Base beta: {base_beta}")
        
        # Generate all strain diseases using utility function
        diseases = create_strain_diseases(initial_strains, fitness_scenario, base_beta)
        
        # Add default connectors if none provided
        if 'connectors' not in kwargs:
            default_connectors = [
                RotaImmunityConnector(),
                RotaReassortmentConnector()
            ]
            
            kwargs['connectors'] = default_connectors
            connector_names = [type(c).__name__ for c in default_connectors]
            print(f"  Connectors: Adding default connectors: {', '.join(connector_names)}")
        else:
            print(f"  Connectors: Using {len(kwargs['connectors'])} custom connectors")
        
        # Set reasonable defaults for rotavirus simulations if not provided  
        rotasim_defaults = {
            'n_agents': 10_000,
            'start': 2020,  # Start year 
            'stop': 2030,   # 10 years later
            'unit': 'day',
            'dt': 1,  # Daily timesteps
            'networks': 'random',
        }
        
        # Apply defaults only if not explicitly provided
        for key, default_value in rotasim_defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value
        
        print(f"  Simulation: {kwargs['n_agents']} agents, {kwargs['start']}-{kwargs['stop']}, dt={kwargs['dt']}")
        print(f"  Total diseases: {len(diseases)} ({len(initial_strains)} active + {len(diseases)-len(initial_strains)} dormant)")
        
        # Initialize parent Sim class
        super().__init__(diseases=diseases, **kwargs)
        
    @property
    def initial_strains(self):
        """Get the initial strains used to create this simulation"""
        return self._initial_strains
        
    @property
    def fitness_scenario(self):
        """Get the fitness scenario used"""
        return self._fitness_scenario
        
    @property
    def base_beta(self):
        """Get the base beta parameter used"""
        return self._base_beta
        
    def get_strain_summary(self):
        """
        Get a summary of all strains in the simulation
        
        Returns:
            Dict with strain information
        """
        # Check if simulation has been initialized
        if not hasattr(self, 'diseases'):
            # Use utility function to get expected strain information
            from .utils import generate_gp_reassortments
            expected_combinations = generate_gp_reassortments(self._initial_strains)
            
            summary = {
                'total_diseases': len(expected_combinations),
                'initial_strains': self._initial_strains,
                'active_strains': [],
                'dormant_strains': [],
            }
            
            # Generate expected strain info
            for G, P in expected_combinations:
                strain_info = {
                    'name': f'G{G}P{P}',
                    'G': G,
                    'P': P,
                    'strain': (G, P),
                }
                
                if (G, P) in self._initial_strains:
                    summary['active_strains'].append(strain_info)
                else:
                    summary['dormant_strains'].append(strain_info)
            
            return summary
        
        # If initialized, use actual diseases
        summary = {
            'total_diseases': len(self.diseases),
            'initial_strains': self._initial_strains,
            'active_strains': [],
            'dormant_strains': [],
        }
        
        for disease in self.diseases.values():
            if hasattr(disease, 'G') and hasattr(disease, 'P'):  # Is Rotavirus
                strain_info = {
                    'name': disease.name,
                    'G': disease.G,
                    'P': disease.P,
                    'strain': disease.strain,
                }
                
                # Check if this was an initial strain
                if (disease.G, disease.P) in self._initial_strains:
                    summary['active_strains'].append(strain_info)
                else:
                    summary['dormant_strains'].append(strain_info)
        
        return summary
        
    def print_strain_summary(self):
        """Print a summary of all strains in the simulation"""
        summary = self.get_strain_summary()
        
        print(f"\n=== Rotasim Strain Summary ===")
        print(f"Total diseases: {summary['total_diseases']}")
        print(f"Initial strains: {len(summary['active_strains'])}")
        print(f"Dormant reassortants: {len(summary['dormant_strains'])}")
        
        print(f"\nActive strains:")
        for strain in summary['active_strains']:
            print(f"  {strain['name']}: G{strain['G']}P{strain['P']}")
            
        if summary['dormant_strains']:
            print(f"\nDormant reassortants (first 10):")
            for strain in summary['dormant_strains'][:10]:
                print(f"  {strain['name']}: G{strain['G']}P{strain['P']}")
            if len(summary['dormant_strains']) > 10:
                print(f"  ... and {len(summary['dormant_strains']) - 10} more")
        
    @staticmethod
    def list_fitness_scenarios():
        """List available built-in fitness scenarios"""
        return list_fitness_scenarios()
        
    def __repr__(self):
        """String representation of Rotasim instance"""
        return (f"Rotasim(initial_strains={self._initial_strains}, "
                f"fitness_scenario={self._fitness_scenario}, "
                f"n_agents={self.pars.n_agents})")


# Make Rotasim importable from the package root
__all__ = ['Rotasim']