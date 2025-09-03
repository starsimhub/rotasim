"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""
import itertools
import starsim as ss
from .rotavirus import Rotavirus

INITIAL_STRAIN_SCENARIOS = {
    'default': [(1, 8), (2, 4), (3, 8)],  # Default initial strains
    'high_diversity': [(1, 8), (2, 4), (3, 8), (4, 8), (9, 8), (12, 8), (9, 6), (12, 6), (9, 4), (1, 6), (2, 8), (2, 6)],
    'low_diversity': [(1, 8), (2, 4), (3, 8), (4, 8)],
}

# Built-in fitness scenarios based on v1 fitness hypotheses
# TODO update scenarios based on fitness hypotheses in old version, rename to match
FITNESS_HYPOTHESES = {
    'baseline': {
        'default': 1.0,  # Default fitness multiplier if not specified
        (1, 8): 1.0,
        (2, 4): 0.8,
    },
    '2': {
        'default': 0.9,  # Default fitness multiplier if not specified
        (1, 1): 0.93,
        (2, 2): 0.93,
        (3, 3): 0.93,
        (4, 4): 0.93,
    },
    '15': {
        'default': 0.4,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.7,
        (3, 8): 0.93,
        (4, 8): 0.93,
        (9, 8): 0.95,
        (12, 8): 0.94,
        (9, 6): 0.3,
        (11, 8): 0.35,
    },
    '17': {
        'default': 0.7,  # Default fitness multiplier if not specified
        (1, 8): 1.0,
        (2, 4): 0.85,
        (3, 8): 0.85,
        (4, 8): 0.88,
        (9, 8): 0.95,
        (12, 8): 0.93,
        (9, 6): 0.83,
        (12, 6): 0.90,
        (9, 4): 0.90,
        (1, 6): 0.8,
        (2, 8): 0.8,
        (2, 6): 0.8,
    },
}

# INITIAL_PREVALENCE_SCENARIOS = {
#     'equal': {
#         (1, 8): n_init_seg,
#         (2, 4): n_init_seg,
#         (9, 8): n_init_seg,
#         (4, 8): n_init_seg,
#         (3, 8): n_init_seg,
#         (12, 8): n_init_seg,
#         (12, 6): n_init_seg,
#         (9, 4): n_init_seg,
#         (9, 6): n_init_seg,
#         (1, 6): n_init_seg,
#         (2, 8): n_init_seg,
#         (2, 6): n_init_seg,
#         (11, 8): n_init_seg,
#         (11, 6): n_init_seg,
#         (1, 4): n_init_seg,
#         (12, 4): n_init_seg,
#     },
#     'low_diversity': {
#         (1, 8): n_init_seg,
#         (3, 8): n_init_seg,
#         (2, 4): n_init_seg,
#         (4, 8): n_init_seg,
#     }
# }


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
        if scenario not in FITNESS_HYPOTHESES:
            raise ValueError(f"Unknown fitness scenario '{scenario}'. Available: {list(FITNESS_HYPOTHESES.keys())}")
        scenario = FITNESS_HYPOTHESES[scenario]
    
    # Return fitness multiplier, defaulting to 1.0
    default = scenario.get('default', 1.0)
    return scenario.get((G, P), default)


def _parse_init_prev_parameter(init_prev, initial_strains):
    """
    Parse init_prev parameter into a standardized dict format
    
    Args:
        init_prev: Float or dict specifying initial prevalence
        initial_strains: List of (G,P) tuples for validation
        
    Returns:
        Dict mapping (G,P) tuples to prevalence values
        
    Raises:
        ValueError: If format is invalid or values are out of range
    """

    def _parse_dict(d):
        # Dict format: {(G,P): prevalence}
        result = {}
        for strain, prev in d.items():
            if not isinstance(strain, (tuple, list)) or len(strain) != 2:
                raise ValueError(f"Dict keys must be (G,P) tuples, got {strain}")
            if not isinstance(prev, (int, float)) or prev < 0 or prev > 1:
                raise ValueError(f"Prevalence values must be between 0 and 1, got {prev} for strain {strain}")
            result[tuple(strain)] = float(prev)
        return result

    if isinstance(init_prev, (int, float)):
        # Float format: same prevalence for all initial strains
        if init_prev < 0 or init_prev > 1:
            raise ValueError(f"init_prev must be between 0 and 1, got {init_prev}")
        return {strain: float(init_prev) for strain in initial_strains}

    elif isinstance(init_prev, dict):
        return _parse_dict(init_prev)  # Validate format

    elif isinstance(init_prev, str):
        # String format: use predefined prevalence scenarios
        if init_prev not in PREVALENCE_SCENARIOS:
            raise ValueError(f"Unknown prevalence scenario '{init_prev}'. Available: {list(PREVALENCE_SCENARIOS.keys())}")
        return _parse_dict(PREVALENCE_SCENARIOS[init_prev])
    else:
        raise ValueError(f"init_prev must be float or dict, got {type(init_prev)}")


def create_strain_diseases(initial_strains, fitness_scenario='baseline', base_beta=0.1, init_prev=0.01):
    """
    Create all Rotavirus disease instances for multi-strain simulation
    
    This is the main user-facing function that generates all possible reassortant
    strains from the initial strains, applies fitness multipliers, and creates
    Rotavirus disease instances ready for simulation.
    
    Args:
        initial_strains: List of (G,P) tuples, e.g. [(1,8), (2,4)]
        fitness_scenario: Dict of fitness multipliers or string name ('baseline', 'high_diversity', 'low_diversity')
        base_beta: Base transmission rate before fitness adjustment
        init_prev: Initial prevalence for active strains. Can be:
                   - Float: Same prevalence for all initial strains (default: 0.01)
                   - Dict: {(G,P): prevalence} for strain-specific values
        
    Returns:
        List of Rotavirus disease instances for all possible reassortants
        
    Examples:
        >>> # Uniform initial prevalence
        >>> diseases = create_strain_diseases([(1,8), (2,4)], init_prev=0.02)
        
        >>> # Strain-specific prevalence
        >>> diseases = create_strain_diseases([(1,8), (2,4)], 
        ...                                  init_prev={(1,8): 0.02, (2,4): 0.005})
    """
    if not initial_strains:
        raise ValueError("initial_strains cannot be empty")

    if isinstance(initial_strains, str):
        if initial_strains not in INITIAL_STRAIN_SCENARIOS:
            raise ValueError(f"Unknown initial_strains scenario '{initial_strains}'. Available: {list(INITIAL_STRAIN_SCENARIOS.keys())}")
        initial_strains = INITIAL_STRAIN_SCENARIOS[initial_strains]
        
    # Parse init_prev parameter into a dict format
    init_prev_dict = _parse_init_prev_parameter(init_prev, initial_strains)
        
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
        # Get initial prevalence for this strain (0.0 for dormant reassortants)
        strain_init_prev = init_prev_dict.get((G, P), 0.0)
        
        # Apply fitness multiplier to base beta
        fitness_mult = get_fitness_multiplier(G, P, fitness_scenario)
        adjusted_beta = base_beta * fitness_mult
        
        # Create disease instance with proper Starsim parameter format
        disease = Rotavirus(G=G, P=P, 
                          init_prev=ss.bernoulli(p=strain_init_prev), 
                          beta=ss.rate_prob(adjusted_beta))
        diseases.append(disease)
        
        if strain_init_prev > 0:
            active_count += 1
            print(f"    {disease.name}: beta={adjusted_beta:.3f} (x{fitness_mult:.2f}), init_prev={strain_init_prev} [ACTIVE]")
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
        
    if not isinstance(initial_strains, (list, tuple, str)):
        raise ValueError("initial_strains must be a list, tuple, or scenario name string")

    if isinstance(initial_strains, str):
        if initial_strains not in INITIAL_STRAIN_SCENARIOS:
            raise ValueError(f"Unknown initial_strains scenario '{initial_strains}'. Available: {list(INITIAL_STRAIN_SCENARIOS.keys())}")
        initial_strains = INITIAL_STRAIN_SCENARIOS[initial_strains]
        
    for i, strain in enumerate(initial_strains):
        if not isinstance(strain, (list, tuple)) or len(strain) != 2:
            raise ValueError(f"Strain {i} must be a (G,P) tuple, got {strain}")
            
        G, P = strain
        if not isinstance(G, int) or not isinstance(P, int):
            raise ValueError(f"G and P must be integers, got G={G} (type {type(G)}), P={P} (type {type(P)})")
            
        if G <= 0 or P <= 0:
            raise ValueError(f"G and P must be positive integers, got G={G}, P={P}")
    
    return True