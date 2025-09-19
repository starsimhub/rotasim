"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""
import itertools

INITIAL_STRAIN_SCENARIOS = {
    'default': [(1, 8), (2, 4), (3, 8)],  # Default initial strains
    'high_diversity': [(1, 8), (2, 4), (3, 8), (4, 8), (9, 8), (12, 8), (9, 6), (12, 6), (9, 4), (1, 6), (2, 8), (2, 6)],
    'low_diversity': [(1, 8), (2, 4), (3, 8), (4, 8)],
}

# Built-in fitness scenarios based on v1 fitness hypotheses
FITNESS_HYPOTHESES = {
    'default': {
        'default': 1.0,  # Default fitness multiplier if not specified
    },
    '1': {
        'default': 1,  # Default fitness multiplier if not specified
    },
    '2': {
        'default': 0.9,  # Default fitness multiplier if not specified
        (1, 1): 0.93,
        (2, 2): 0.93,
        (3, 3): 0.93,
        (4, 4): 0.93,
    },
    '3': {
        'default': 0.87,
        (1, 1): 0.93,
        (2, 2): 0.93,
        (3, 3): 0.90,
        (4, 4): 0.90,
    },
    '4': {
        'default': 1,
        (1, 1): 1,
        (2, 2): 0.2,
    },
    '5': {
        'default': 0.2,  # Default fitness multiplier if not specified
        (1, 1): 1,
        (2, 1): 0.5,
        (1, 3): 0.5,
    },
    '6': {
        'default': 0.05,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.2,
        (3, 8): 0.4,
        (4, 8): 0.5,
    },
    '7': {
        'default': 0.05,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.3,
        (3, 8): 0.7,
        (4, 8): 0.6,
    },
    '8': {
        'default': 0.05,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.4,
        (3, 8): 0.9,
        (4, 8): 0.8,
    },
    '9': {
        'default': 0.2,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.6,
        (3, 8): 0.9,
        (4, 8): 0.9,
    },
    '10': {
        'default': 0.4,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.6,
        (3, 8): 0.9,
        (4, 8): 0.9,
    },
    '11': {
        'default': 0.5,  # Default fitness multiplier if not specified
        (1, 8): 0.98,
        (2, 4): 0.7,
        (3, 8): 0.8,
        (4, 8): 0.8,
    },
    '12': {
        'default': 0.5,  # Default fitness multiplier if not specified
        (1, 8): 0.98,
        (2, 4): 0.7,
        (3, 8): 0.9,
        (4, 8): 0.9,
    },
    '13': {
        'default': 0.7,  # Default fitness multiplier if not specified
        (1, 8): 0.98,
        (2, 4): 0.8,
        (3, 8): 0.9,
        (4, 8): 0.9,
    },
    '14': {
        'default': 0.05,  # Default fitness multiplier if not specified
        (1, 8): 0.98,
        (2, 4): 0.4,
        (3, 8): 0.7,
        (12, 8): 0.75,
        (9, 6): 0.58,
        (11, 8): 0.2,
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
    '16': {
        'default': 0.4,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.7,
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
    # below fitness hypo. 18 was used in the analysis for the high baseline diversity setting in the report
    '18': {
        'default': 0.65,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.92,
        (3, 8): 0.79,
        (4, 8): 0.81,
        (9, 8): 0.95,
        (12, 8): 0.89,
        (9, 6): 0.80,
        (12, 6): 0.86,
        (9, 4): 0.83,
        (1, 6): 0.75,
        (2, 8): 0.75,
        (2, 6): 0.75,
    },
    # below fitness hypo 19 was used for the low baseline diversity setting analysis in the report
    '19': {
        'default': 0.4,  # Default fitness multiplier if not specified
        (1, 8): 1,
        (2, 4): 0.5,
        (3, 8): 0.55,
        (4, 8): 0.55,
        (9, 8): 0.6,
    },
}

# Preferred partners for reassortment. Dict key is G, value is list of preferred P partners
PREFERRED_PARTNERS = {
    1: [6, 8],
    2: [4, 6, 8],
    3: [6, 8],
    4: [8],
    9: [4, 6, 8],
    12: [6, 8],
}

def generate_gp_reassortments(initial_strains, use_preferred_partners=False, verbose=False):
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

    all_reassortments = []
    # Optionally filter P genotypes to preferred partners
    if use_preferred_partners:
        for g in unique_G:
            if g not in PREFERRED_PARTNERS:
                raise ValueError(f"No preferred partners defined for G genotype {g}")
            for p in unique_P:
                if p not in PREFERRED_PARTNERS[g]:
                    if verbose:
                        print(f"Warning: P genotype {p} is not a preferred partner for G genotype {g}")
                all_reassortments.append((g, p))

    else:
        # Generate all possible combinations
        all_reassortments = list(itertools.product(unique_G, unique_P))

    return all_reassortments


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
        >>> get_fitness_multiplier(1, 1, '4')
        1.0
        >>> get_fitness_multiplier(2, 2, '4')
        0.8
        >>> get_fitness_multiplier(3, 6, '4')  # Not in dict
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
