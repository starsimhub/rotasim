"""
Sim convenience class for multi-strain simulations
For starsim v2, the class must be named "Sim", not "Rotasim"
Provides an easy-to-use interface for researchers while maintaining flexibility
"""
# Third-party imports
import starsim as ss

# Local imports
from .immunity import RotaImmunityConnector
from .reassortment import RotaReassortmentConnector
from .utils import validate_scenario, apply_scenario_overrides


class Sim(ss.Sim):
    """
    Unified Rotavirus simulation class using scenario-based configuration
    
    This class uses predefined scenarios that contain strain definitions, fitness values, 
    and prevalence settings all in one place. Easy override parameters allow customization.
    Automatically generates all possible strain reassortants.
    
    Example usage:
        # Simple predefined scenario
        sim = Sim(scenario='realistic_competition')
        sim.run()
        
        # Override prevalence for all strains
        sim = Sim(scenario='baseline', override_prevalence=0.02)
        
        # Override specific strain fitness
        sim = Sim(scenario='baseline', override_fitness={(1,8): 0.95, (2,4): 0.8})
        
        # Add new strain to existing scenario
        sim = Sim(scenario='baseline', override_strains={(9,6): {'fitness': 0.7, 'prevalence': 0.003}})
        
        # Custom scenario
        sim = Sim(scenario={
            'strains': {
                (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
                (2, 4): {'fitness': 0.8, 'prevalence': 0.010}
            },
            'default_fitness': 0.5
        })
        
        # Custom connectors and analyzers still supported
        sim = Sim(scenario='baseline', connectors=[RotaImmunityConnector()], analyzers=[MyAnalyzer()])
    """
    
    def __init__(self, scenario='baseline', base_beta=0.1, override_fitness=None, override_prevalence=None, 
                 override_strains=None, use_preferred_partners=False, **kwargs):
        """
        Initialize Rotasim simulation using unified scenario system
        
        Args:
            scenario: String name of built-in scenario or dict containing custom scenario data
            base_beta: Base transmission rate before fitness adjustment (default: 0.1)
            override_fitness: Override fitness values - float (all strains) or dict {(G,P): fitness}
            override_prevalence: Override prevalence values - float (all strains) or dict {(G,P): prevalence}
            override_strains: Add/modify strains - dict {(G,P): {'fitness': X, 'prevalence': Y}}
            use_preferred_partners: Whether to filter reassortments by preferred partners (default: False)
            **kwargs: Additional arguments passed to ss.Sim (n_agents, start, stop, connectors, analyzers, etc.)
            
        Raises:
            ValueError: If scenario is invalid or override parameters are malformed
            
        Examples:
            # Simple usage
            sim = Sim(scenario='realistic_competition')
            
            # Override all prevalence values
            sim = Sim(scenario='baseline', override_prevalence=0.02)
            
            # Override specific strain fitness
            sim = Sim(scenario='baseline', override_fitness={(1,8): 0.95, (2,4): 0.8})
            
            # Add new strain
            sim = Sim(scenario='baseline', override_strains={(9,6): {'fitness': 0.7, 'prevalence': 0.003}})
        """
        # Validate and process scenario
        validated_scenario = validate_scenario(scenario)
        
        # Apply any overrides
        final_scenario = apply_scenario_overrides(
            validated_scenario,
            override_fitness=override_fitness,
            override_prevalence=override_prevalence, 
            override_strains=override_strains
        )
        
        # Store configuration for reference
        self._scenario = scenario
        self._final_scenario = final_scenario
        self._base_beta = base_beta
        self._use_preferred_partners = use_preferred_partners
        
        print("Rotasim: Setting up multi-strain simulation")
        scenario_name = scenario if isinstance(scenario, str) else 'custom'
        print(f"  Scenario: {scenario_name}")
        print(f"  Base beta: {base_beta}")
        print(f"  Strains: {len(final_scenario['strains'])}")
        
        diseases = self._create_strain_diseases(final_scenario, base_beta, use_preferred_partners, verbose=kwargs.get('verbose', False))
        
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
            'dt': ss.days(1),  # Daily timesteps
        }
        
        # Apply defaults only if not explicitly provided
        for key, default_value in rotasim_defaults.items():
            if key not in kwargs:
                kwargs[key] = default_value
        
        print(f"  Time units: day, dt={kwargs.get('dt', ss.days(1))}")
        active_strains = len(final_scenario['strains'])
        print(f"  Total diseases: {len(diseases)} ({active_strains} active + {len(diseases)-active_strains} dormant)")

        if 'networks' not in kwargs:
            kwargs['networks'] = 'random'
            print("  Networks: Using default random network")
        
        # Initialize parent Sim class
        super().__init__(diseases=diseases, **kwargs)
        
    @property
    def scenario(self):
        """Get the scenario used to create this simulation"""
        return self._scenario
        
    @property
    def initial_strains(self):
        """Get the initial strains from the final scenario"""
        return list(self._final_scenario['strains'].keys())
        
    @property
    def base_beta(self):
        """Get the base beta parameter used"""
        return self._base_beta
        
    @property
    def final_scenario(self):
        """Get the final processed scenario with all overrides applied"""
        return self._final_scenario
        
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
            initial_strains = list(self._final_scenario['strains'].keys())
            expected_combinations = generate_gp_reassortments(initial_strains, use_preferred_partners=self._use_preferred_partners)
            
            summary = {
                'total_diseases': len(expected_combinations),
                'initial_strains': initial_strains,
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
                
                if (G, P) in self._final_scenario['strains']:
                    summary['active_strains'].append(strain_info)
                else:
                    summary['dormant_strains'].append(strain_info)
            
            return summary
        
        # If initialized, use actual diseases
        summary = {
            'total_diseases': len(self.diseases),
            'initial_strains': list(self._final_scenario['strains'].keys()) if hasattr(self, '_final_scenario') else [],
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
                
                # Check if this was an initial strain (has prevalence > 0)
                initial_strains = list(self._final_scenario['strains'].keys()) if hasattr(self, '_final_scenario') else []
                if (disease.G, disease.P) in initial_strains:
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
        
    def _create_strain_diseases(self, scenario, base_beta=0.1, use_preferred_partners=False, verbose=False):
        """
        Create all Rotavirus disease instances from unified scenario data
        
        This method generates all possible reassortant strains from the scenario's initial strains,
        applies fitness and prevalence values, and creates Rotavirus disease instances ready for simulation.
        
        Args:
            scenario: Dict containing scenario data with 'strains' and 'default_fitness'
            base_beta: Base transmission rate before fitness adjustment
            use_preferred_partners: Whether to filter reassortments by preferred partners
            verbose: Verbose level for printing setup information
            
        Returns:
            List of Rotavirus disease instances for all possible reassortants
        """
        from .utils import generate_gp_reassortments
        from .rotavirus import Rotavirus
        
        if not scenario['strains']:
            raise ValueError("Scenario must contain at least one strain")

        # Extract initial strains from scenario
        initial_strains = list(scenario['strains'].keys())
            
        # Generate all possible G,P combinations
        gp_combinations = generate_gp_reassortments(initial_strains, use_preferred_partners, verbose)
        
        # Strain creation details (debug verbose)
        if verbose > 1:
            print(f"Creating {len(gp_combinations)} strain diseases from {len(initial_strains)} initial strains")
            print(f"  Initial strains: {initial_strains}")
            print(f"  All combinations: {gp_combinations}")
            print(f"  Default fitness: {scenario.get('default_fitness', 1.0)}")
        
        diseases = []
        active_count = 0
        dormant_count = 0
        
        for G, P in gp_combinations:
            strain_key = (G, P)
            
            # Get strain data from scenario (if it exists) or use defaults
            if strain_key in scenario['strains']:
                strain_data = scenario['strains'][strain_key]
                strain_fitness = strain_data['fitness']
                strain_prevalence = strain_data['prevalence']
            else:
                # Dormant reassortant - use default fitness and zero prevalence
                strain_fitness = scenario.get('default_fitness', 1.0)
                strain_prevalence = 0.0
            
            # Apply fitness multiplier to base beta
            adjusted_beta = base_beta * strain_fitness
            
            # Create disease instance with proper Starsim parameter format
            disease = Rotavirus(G=G, P=P, 
                              init_prev=ss.bernoulli(p=strain_prevalence), 
                              beta=ss.perday(adjusted_beta),
                              dur_inf = ss.lognorm_ex(mean=4),)
            diseases.append(disease)
            
            if strain_prevalence > 0:
                active_count += 1
                # Individual strain details (debug verbose)
                if verbose > 1:
                    print(f"    {disease.name}: beta={adjusted_beta:.3f} (x{strain_fitness:.2f}), prevalence={strain_prevalence} [ACTIVE]")
            else:
                dormant_count += 1
                
        # Summary (basic verbose)
        if verbose:
            print(f"  Created {active_count} active strains and {dormant_count} dormant reassortants")
        
        return diseases


    def get_connector_by_type(self, connector_type, warn_if_multiple=True):
        """
        Find a connector by type, with warning if not exactly one found
        
        Args:
            connector_type: Class type or string name of the connector to find
            warn_if_multiple: Whether to warn if multiple connectors found (default: True)
            
        Returns:
            Connector instance or None if not found
        """
        # Handle string type names by checking class names
        if isinstance(connector_type, str):
            matching_connectors = [c for c in self.connectors.values() 
                                  if c.__class__.__name__ == connector_type]
            type_name = connector_type
        else:
            matching_connectors = [c for c in self.connectors.values() 
                                  if isinstance(c, connector_type)]
            type_name = connector_type.__name__
        
        if len(matching_connectors) == 0:
            if self.pars.verbose:
                print(f"Warning: No {type_name} found in simulation")
            return None
        elif len(matching_connectors) > 1 and warn_if_multiple:
            if self.pars.verbose:
                print(f"Warning: Multiple {type_name}s found ({len(matching_connectors)}), using first one")
        
        return matching_connectors[0]

    def __repr__(self):
        """String representation of Rotasim instance"""
        scenario_name = self._scenario if isinstance(self._scenario, str) else 'custom'
        return (f"Sim(scenario={scenario_name}, "
                f"strains={len(self._final_scenario['strains'])}, "
                f"n_agents={self.pars.n_agents})")


# Make Rotasim importable from the package root
__all__ = ['Sim']