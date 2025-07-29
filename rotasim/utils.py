"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""
import itertools
from .rotavirus import Rotavirus


# Built-in fitness scenarios based on v1 fitness hypotheses
FITNESS_SCENARIOS = {
    'baseline': {
        (1, 8): 1.0,
        (2, 4): 0.8,
    },
    'high_diversity': {
        (1, 8): 0.98,
        (2, 4): 0.4,
        (3, 8): 0.7,
        (4, 8): 0.6,
        (9, 8): 0.7,
        (12, 8): 0.75,
        (9, 6): 0.58,
        (11, 8): 0.2,
    },
    'low_diversity': {
        (1, 8): 1.0,
        (2, 4): 0.85,
        (3, 8): 0.85,
        (4, 8): 0.88,
        (9, 8): 0.95,
        (12, 8): 0.93,
        (9, 6): 0.85,
        (12, 6): 0.90,
        (9, 4): 0.90,
        (1, 6): 0.6,
        (2, 8): 0.6,
        (2, 6): 0.6,
    },
}


def generate_gp_reassortments(initial_strains):
    """
    Generate all possible G,P combinations from initial strains
    
    Args:
        initial_strains: List of (G,P) tuples, e.g. [(1,8), (2,4)]
        
    Returns:
        List of (G,P) tuples representing all possible reassortants
        
    Example:
        >>> generate_gp_reassortments([(1,8), (2,4)])
        [(1, 8), (1, 4), (2, 8), (2, 4)]
    """
    if not initial_strains:
        raise ValueError("initial_strains cannot be empty")
        
    # Extract unique G and P genotypes
    unique_G = sorted(set(g for g, p in initial_strains))
    unique_P = sorted(set(p for g, p in initial_strains))
    
    # Generate all possible combinations
    all_combinations = list(itertools.product(unique_G, unique_P))
    
    return all_combinations


def get_fitness_multiplier(G, P, scenario):
    """
    Get fitness multiplier for a G,P combination
    
    Args:
        G: G genotype
        P: P genotype  
        scenario: Dict of fitness multipliers or string name of built-in scenario
        
    Returns:
        Float fitness multiplier (defaults to 1.0 if not found)
        
    Example:
        >>> get_fitness_multiplier(1, 8, 'baseline')
        1.0
        >>> get_fitness_multiplier(2, 4, 'baseline') 
        0.8
        >>> get_fitness_multiplier(3, 6, 'baseline')  # Not in dict
        1.0
    """
    # Handle string scenario names
    if isinstance(scenario, str):
        if scenario not in FITNESS_SCENARIOS:
            raise ValueError(f"Unknown fitness scenario '{scenario}'. Available: {list(FITNESS_SCENARIOS.keys())}")
        scenario = FITNESS_SCENARIOS[scenario]
    
    # Return fitness multiplier, defaulting to 1.0
    return scenario.get((G, P), 1.0)


def create_strain_diseases(initial_strains, fitness_scenario='baseline', base_beta=0.1):
    """
    Create all Rotavirus disease instances for multi-strain simulation
    
    This is the main user-facing function that generates all possible reassortant
    strains from the initial strains, applies fitness multipliers, and creates
    Rotavirus disease instances ready for simulation.
    
    Args:
        initial_strains: List of (G,P) tuples, e.g. [(1,8), (2,4)]
        fitness_scenario: Dict of fitness multipliers or string name ('baseline', 'high_diversity', 'low_diversity')
        base_beta: Base transmission rate before fitness adjustment
        
    Returns:
        List of Rotavirus disease instances for all possible reassortants
        
    Example:
        >>> diseases = create_strain_diseases([(1,8), (2,4)], 'baseline', 0.1)
        >>> len(diseases)  # 4 combinations: (1,8), (1,4), (2,8), (2,4)
        4
        >>> diseases[0].name
        'G1P8'
        >>> diseases[0].pars.beta  # base_beta * fitness_multiplier
        0.1  # 0.1 * 1.0 for G1P8 in baseline scenario
    """
    if not initial_strains:
        raise ValueError("initial_strains cannot be empty")
        
    # Generate all possible G,P combinations
    gp_combinations = generate_gp_reassortments(initial_strains)
    
    print(f"Creating {len(gp_combinations)} strain diseases from {len(initial_strains)} initial strains")
    print(f"  Initial strains: {initial_strains}")
    print(f"  All combinations: {gp_combinations}")
    print(f"  Fitness scenario: {fitness_scenario if isinstance(fitness_scenario, str) else 'custom'}")
    
    diseases = []
    active_count = 0
    dormant_count = 0
    
    for G, P in gp_combinations:
        # Initial strains have prevalence, reassortants start dormant
        init_prev = 0.01 if (G, P) in initial_strains else 0.0
        
        # Apply fitness multiplier to base beta
        fitness_mult = get_fitness_multiplier(G, P, fitness_scenario)
        adjusted_beta = base_beta * fitness_mult
        
        # Create disease instance
        disease = Rotavirus(G=G, P=P, init_prev=init_prev, beta=adjusted_beta)
        diseases.append(disease)
        
        if init_prev > 0:
            active_count += 1
            print(f"    {disease.name}: beta={adjusted_beta:.3f} (x{fitness_mult:.2f}), init_prev={init_prev} [ACTIVE]")
        else:
            dormant_count += 1
            
    print(f"  Created {active_count} active strains and {dormant_count} dormant reassortants")
    
    return diseases


def list_fitness_scenarios():
    """
    List available built-in fitness scenarios
    
    Returns:
        Dict mapping scenario names to descriptions
    """
    return {
        'baseline': 'Simple baseline scenario with G1P8 dominant',
        'high_diversity': 'High diversity scenario with many competing strains',
        'low_diversity': 'Low diversity scenario with few dominant strains',
    }


def validate_initial_strains(initial_strains):
    """
    Validate initial_strains format and content
    
    Args:
        initial_strains: List of (G,P) tuples
        
    Raises:
        ValueError: If format is invalid
        
    Returns:
        True if valid
    """
    if not initial_strains:
        raise ValueError("initial_strains cannot be empty")
        
    if not isinstance(initial_strains, (list, tuple)):
        raise ValueError("initial_strains must be a list or tuple")
        
    for i, strain in enumerate(initial_strains):
        if not isinstance(strain, (list, tuple)) or len(strain) != 2:
            raise ValueError(f"Strain {i} must be a (G,P) tuple, got {strain}")
            
        G, P = strain
        if not isinstance(G, int) or not isinstance(P, int):
            raise ValueError(f"G and P must be integers, got G={G} (type {type(G)}), P={P} (type {type(P)})")
            
        if G <= 0 or P <= 0:
            raise ValueError(f"G and P must be positive integers, got G={G}, P={P}")
    
    return True