"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""
import itertools

# Unified scenario system - contains strains, fitness, and prevalence all in one place
SCENARIOS = {
    'simple': {
        'description': 'Simple two-strain scenario - G1P8 and G2P4 with equal fitness and prevalence',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.01},
            (2, 4): {'fitness': 1.0, 'prevalence': 0.01}
        },
        'default_fitness': 1.0
    },
    
    'baseline': {
        'description': 'Baseline scenario - common global strains with equal fitness',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
            (2, 4): {'fitness': 1.0, 'prevalence': 0.008}, 
            (3, 8): {'fitness': 1.0, 'prevalence': 0.007}
        },
        'default_fitness': 1.0
    },
    
    'realistic_competition': {
        'description': 'G1P8 dominant with realistic strain competition',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
            (2, 4): {'fitness': 0.2, 'prevalence': 0.008},
            (3, 8): {'fitness': 0.4, 'prevalence': 0.007},
            (4, 8): {'fitness': 0.5, 'prevalence': 0.005}
        },
        'default_fitness': 0.05
    },
    
    'balanced_competition': {
        'description': 'G1P8 dominant with moderate balanced competition',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
            (2, 4): {'fitness': 0.6, 'prevalence': 0.008},
            (3, 8): {'fitness': 0.9, 'prevalence': 0.007},
            (4, 8): {'fitness': 0.9, 'prevalence': 0.005}
        },
        'default_fitness': 0.2
    },
    
    'high_diversity': {
        'description': 'High diversity with 12 strains and varied fitness',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.012},
            (2, 4): {'fitness': 0.7, 'prevalence': 0.007},
            (3, 8): {'fitness': 0.85, 'prevalence': 0.005},
            (4, 8): {'fitness': 0.88, 'prevalence': 0.004},
            (9, 8): {'fitness': 0.95, 'prevalence': 0.003},
            (12, 8): {'fitness': 0.93, 'prevalence': 0.003},
            (9, 6): {'fitness': 0.85, 'prevalence': 0.002},
            (12, 6): {'fitness': 0.90, 'prevalence': 0.002},
            (9, 4): {'fitness': 0.90, 'prevalence': 0.002},
            (1, 6): {'fitness': 0.6, 'prevalence': 0.002},
            (2, 8): {'fitness': 0.6, 'prevalence': 0.002},
            (2, 6): {'fitness': 0.6, 'prevalence': 0.002}
        },
        'default_fitness': 0.4
    },
    
    'low_diversity': {
        'description': 'Low diversity with 4 main competitive strains',
        'strains': {
            (1, 8): {'fitness': 0.98, 'prevalence': 0.020},
            (2, 4): {'fitness': 0.7, 'prevalence': 0.012},
            (3, 8): {'fitness': 0.8, 'prevalence': 0.008},
            (4, 8): {'fitness': 0.8, 'prevalence': 0.005}
        },
        'default_fitness': 0.5
    },
    
    'emergence_scenario': {
        'description': 'Scenario for studying strain emergence with weak background',
        'strains': {
            (1, 8): {'fitness': 1.0, 'prevalence': 0.015},
            (2, 4): {'fitness': 0.4, 'prevalence': 0.005},
            (3, 8): {'fitness': 0.7, 'prevalence': 0.003}
        },
        'default_fitness': 0.05  # Very low fitness for new emerging strains
    }
}

# Legacy support - keep these for any remaining internal code that uses them
INITIAL_STRAIN_SCENARIOS = {
    'default': {
        'description': 'Default initial strains - common global strains',
        'strains': {
            (1, 8): 0.015,  # G1P8 most prevalent
            (2, 4): 0.008,  # G2P4 moderate
            (3, 8): 0.007   # G3P8 lower
        }
    },
    'high_diversity': {
        'description': 'High diversity scenario with 12 different G,P combinations',
        'strains': {
            (1, 8): 0.012,   # Primary strains with higher prevalence
            (2, 4): 0.007,
            (3, 8): 0.005,
            (4, 8): 0.004,
            (9, 8): 0.003,   # Secondary strains with moderate prevalence
            (12, 8): 0.003,
            (9, 6): 0.002,   # Minor strains with low prevalence
            (12, 6): 0.002,
            (9, 4): 0.002,
            (1, 6): 0.002,
            (2, 8): 0.002,
            (2, 6): 0.002
        }
    },
    'low_diversity': {
        'description': 'Low diversity scenario with 4 main strains',
        'strains': {
            (1, 8): 0.020,  # Higher individual prevalence
            (2, 4): 0.012,
            (3, 8): 0.008,
            (4, 8): 0.005
        }
    },
}

# Built-in fitness scenarios based on v1 fitness hypotheses
FITNESS_HYPOTHESES = {
    'default': {
        'description': 'Default scenario - all strains equal fitness',
        'fitness': {
            'default': 1.0,  # Default fitness multiplier if not specified
        }
    },
    '1': {
        'description': 'Fitness hypothesis 1 - default fitness 1.0',
        'fitness': {
            'default': 1,  # Default fitness multiplier if not specified
        }
    },
    '2': {
        'description': 'Fitness hypothesis 2 - default fitness 0.9, some strains at 0.93',
        'fitness': {
            'default': 0.9,  # Default fitness multiplier if not specified
            (1, 1): 0.93,
            (2, 2): 0.93,
            (3, 3): 0.93,
            (4, 4): 0.93,
        }
    },
    '3': {
        'description': 'Fitness hypothesis 3 - default fitness 0.87, varied strain fitness',
        'fitness': {
            'default': 0.87,
            (1, 1): 0.93,
            (2, 2): 0.93,
            (3, 3): 0.90,
            (4, 4): 0.90,
        }
    },
    '4': {
        'description': 'Fitness hypothesis 4 - G1P1 dominant, G2P2 weak',
        'fitness': {
            'default': 1,
            (1, 1): 1,
            (2, 2): 0.2,
        }
    },
    '5': {
        'description': 'Fitness hypothesis 5 - G1P1 dominant, partial hetero protection',
        'fitness': {
            'default': 0.2,  # Default fitness multiplier if not specified
            (1, 1): 1,
            (2, 1): 0.5,
            (1, 3): 0.5,
        }
    },
    '6': {
        'description': 'Fitness hypothesis 6 - G1P8 dominant, realistic strain competition',
        'fitness': {
            'default': 0.05,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.2,
            (3, 8): 0.4,
            (4, 8): 0.5,
        }
    },
    '7': {
        'description': 'Fitness hypothesis 7 - G1P8 dominant, moderate competition',
        'fitness': {
            'default': 0.05,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.3,
            (3, 8): 0.7,
            (4, 8): 0.6,
        }
    },
    '8': {
        'description': 'Fitness hypothesis 8 - G1P8 dominant, strong competition',
        'fitness': {
            'default': 0.05,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.4,
            (3, 8): 0.9,
            (4, 8): 0.8,
        }
    },
    '9': {
        'description': 'Fitness hypothesis 9 - G1P8 dominant, balanced competition',
        'fitness': {
            'default': 0.2,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.6,
            (3, 8): 0.9,
            (4, 8): 0.9,
        }
    },
    '10': {
        'description': 'Fitness hypothesis 10 - G1P8 dominant, higher background fitness',
        'fitness': {
            'default': 0.4,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.6,
            (3, 8): 0.9,
            (4, 8): 0.9,
        }
    },
    '11': {
        'description': 'Fitness hypothesis 11 - G1P8 slightly reduced, balanced strains',
        'fitness': {
            'default': 0.5,  # Default fitness multiplier if not specified
            (1, 8): 0.98,
            (2, 4): 0.7,
            (3, 8): 0.8,
            (4, 8): 0.8,
        }
    },
    '12': {
        'description': 'Fitness hypothesis 12 - G1P8 slightly reduced, strong G3P8',
        'fitness': {
            'default': 0.5,  # Default fitness multiplier if not specified
            (1, 8): 0.98,
            (2, 4): 0.7,
            (3, 8): 0.9,
            (4, 8): 0.9,
        }
    },
    '13': {
        'description': 'Fitness hypothesis 13 - Higher background fitness, competitive strains',
        'fitness': {
            'default': 0.7,  # Default fitness multiplier if not specified
            (1, 8): 0.98,
            (2, 4): 0.8,
            (3, 8): 0.9,
            (4, 8): 0.9,
        }
    },
    '14': {
        'description': 'Fitness hypothesis 14 - Complex multi-strain competition',
        'fitness': {
            'default': 0.05,  # Default fitness multiplier if not specified
            (1, 8): 0.98,
            (2, 4): 0.4,
            (3, 8): 0.7,
            (12, 8): 0.75,
            (9, 6): 0.58,
            (11, 8): 0.2,
        }
    },
    '15': {
        'description': 'Fitness hypothesis 15 - High diversity with strong G9P8, G12P8',
        'fitness': {
            'default': 0.4,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.7,
            (3, 8): 0.93,
            (4, 8): 0.93,
            (9, 8): 0.95,
            (12, 8): 0.94,
            (9, 6): 0.3,
            (11, 8): 0.35,
        }
    },
    '16': {
        'description': 'Fitness hypothesis 16 - Very high diversity, balanced competition',
        'fitness': {
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
        }
    },
    '17': {
        'description': 'Fitness hypothesis 17 - High diversity with stronger background',
        'fitness': {
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
        }
    },
    '18': {
        'description': 'Fitness hypothesis 18 - High baseline diversity analysis scenario',
        'fitness': {
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
        }
    },
    '19': {
        'description': 'Fitness hypothesis 19 - Low baseline diversity analysis scenario',
        'fitness': {
            'default': 0.4,  # Default fitness multiplier if not specified
            (1, 8): 1,
            (2, 4): 0.5,
            (3, 8): 0.55,
            (4, 8): 0.55,
            (9, 8): 0.6,
        }
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
        scenario = FITNESS_HYPOTHESES[scenario]['fitness']
    
    # Return fitness multiplier, defaulting to 1.0
    default = scenario.get('default', 1.0)
    return scenario.get((G, P), default)


def _parse_init_prev_parameter(init_prev, initial_strains, scenario_data=None):
    """
    Parse init_prev parameter into a standardized dict format with hierarchical precedence
    
    Precedence logic:
    1. If init_prev is not None and is a float, use init_prev for all initial strains
    2. If init_prev is a dictionary and current strain is in keys, use that value
    3. Otherwise, use prevalence from initial strain scenario (if available)
    4. Default to 0.0 for dormant reassortants
    
    Args:
        init_prev: Float or dict specifying initial prevalence, or None
        initial_strains: List of (G,P) tuples for validation
        scenario_data: Dict containing scenario information with 'strains' key, or None
        
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

    # Handle None case - use scenario data or default
    if init_prev is None:
        if scenario_data and 'strains' in scenario_data:
            return dict(scenario_data['strains'])  # Use scenario prevalence values
        else:
            # Default fallback - use 0.01 for initial strains, 0.0 for others
            return {strain: 0.01 for strain in initial_strains}

    # Handle float case - override all initial strains with same value
    if isinstance(init_prev, (int, float)):
        if init_prev < 0 or init_prev > 1:
            raise ValueError(f"init_prev must be between 0 and 1, got {init_prev}")
        return {strain: float(init_prev) for strain in initial_strains}

    # Handle dict case - use hierarchical precedence for each strain
    elif isinstance(init_prev, dict):
        validated_dict = _parse_dict(init_prev)  # Validate format first
        result = {}
        
        # Apply precedence logic for each initial strain
        for strain in initial_strains:
            if strain in validated_dict:
                # Use strain-specific value from init_prev dict
                result[strain] = validated_dict[strain]
            elif scenario_data and 'strains' in scenario_data and strain in scenario_data['strains']:
                # Use prevalence from initial strain scenario
                result[strain] = scenario_data['strains'][strain]
            else:
                # Default fallback
                result[strain] = 0.01
                
        return result

    elif isinstance(init_prev, str):
        # String format: use predefined prevalence scenarios
        if init_prev not in PREVALENCE_SCENARIOS:
            raise ValueError(f"Unknown prevalence scenario '{init_prev}'. Available: {list(PREVALENCE_SCENARIOS.keys())}")
        return _parse_dict(PREVALENCE_SCENARIOS[init_prev])
    else:
        raise ValueError(f"init_prev must be float, dict, or None, got {type(init_prev)}")




def list_scenarios():
    """
    List available built-in simulation scenarios
    
    Returns:
        Dict mapping scenario names to descriptions
    """
    return {name: data['description'] for name, data in SCENARIOS.items()}


def get_scenario(scenario_name):
    """
    Get a scenario by name
    
    Args:
        scenario_name: String name of the scenario
        
    Returns:
        Dict containing scenario data
        
    Raises:
        ValueError: If scenario not found
    """
    if scenario_name not in SCENARIOS:
        available = list(SCENARIOS.keys())
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available: {available}")
    return SCENARIOS[scenario_name].copy()  # Return a copy to avoid modification


def validate_scenario(scenario):
    """
    Validate scenario format and content
    
    Args:
        scenario: Dict containing scenario data or string name
        
    Returns:
        Dict containing validated scenario data
        
    Raises:
        ValueError: If format is invalid
    """
    if isinstance(scenario, str):
        return get_scenario(scenario)
    
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario must be string name or dict, got {type(scenario)}")
    
    if 'strains' not in scenario:
        raise ValueError("Scenario dict must contain 'strains' key")
    
    if not isinstance(scenario['strains'], dict):
        raise ValueError("Scenario 'strains' must be a dict")
    
    # Validate each strain
    for strain, data in scenario['strains'].items():
        if not isinstance(strain, (tuple, list)) or len(strain) != 2:
            raise ValueError(f"Strain key must be (G,P) tuple, got {strain}")
        
        G, P = strain
        if not isinstance(G, int) or not isinstance(P, int) or G <= 0 or P <= 0:
            raise ValueError(f"G,P must be positive integers, got G={G}, P={P}")
        
        if not isinstance(data, dict):
            raise ValueError(f"Strain data must be dict, got {type(data)} for {strain}")
        
        if 'fitness' not in data or 'prevalence' not in data:
            raise ValueError(f"Strain data must contain 'fitness' and 'prevalence', got {data.keys()} for {strain}")
        
        fitness = data['fitness']
        prevalence = data['prevalence']
        
        if not isinstance(fitness, (int, float)) or fitness <= 0:
            raise ValueError(f"Fitness must be positive number, got {fitness} for {strain}")
        
        if not isinstance(prevalence, (int, float)) or prevalence < 0 or prevalence > 1:
            raise ValueError(f"Prevalence must be between 0 and 1, got {prevalence} for {strain}")
    
    # Set default_fitness if not provided
    if 'default_fitness' not in scenario:
        scenario = scenario.copy()
        scenario['default_fitness'] = 1.0
    
    return scenario


def _apply_field_override(strains_dict, override_value, field_name):
    """
    Generalized function to apply overrides to a specific field in strain data
    
    Args:
        strains_dict: Dict of strain data to modify
        override_value: Number (apply to all) or dict (strain-specific) 
        field_name: Name of field to override ('fitness' or 'prevalence')
    """
    if isinstance(override_value, (int, float)):
        # Apply to all strains
        for strain in strains_dict:
            strains_dict[strain][field_name] = float(override_value)
    elif isinstance(override_value, dict):
        # Apply strain-specific values
        for strain, value in override_value.items():
            if tuple(strain) in strains_dict:
                strains_dict[tuple(strain)][field_name] = float(value)
    else:
        raise ValueError(f"override_{field_name} must be number or dict, got {type(override_value)}")


def apply_scenario_overrides(scenario, override_fitness=None, override_prevalence=None, override_strains=None):
    """
    Apply override parameters to a scenario
    
    Args:
        scenario: Base scenario dict
        override_fitness: Dict of {(G,P): fitness} or float for all strains
        override_prevalence: Dict of {(G,P): prevalence} or float for all strains  
        override_strains: Dict of {(G,P): {'fitness': X, 'prevalence': Y}} to add/modify strains
        
    Returns:
        Dict containing modified scenario
    """
    result = scenario.copy()
    result['strains'] = {k: v.copy() for k, v in scenario['strains'].items()}
    
    # Apply strain overrides first (add new strains or modify existing)
    if override_strains:
        for strain, data in override_strains.items():
            if not isinstance(strain, (tuple, list)) or len(strain) != 2:
                raise ValueError(f"Override strain key must be (G,P) tuple, got {strain}")
            if not isinstance(data, dict) or 'fitness' not in data or 'prevalence' not in data:
                raise ValueError(f"Override strain data must contain 'fitness' and 'prevalence', got {data}")
            result['strains'][tuple(strain)] = data.copy()
    
    # Apply field overrides using generalized function
    if override_fitness is not None:
        _apply_field_override(result['strains'], override_fitness, 'fitness')
    
    if override_prevalence is not None:
        _apply_field_override(result['strains'], override_prevalence, 'prevalence')
    
    return result


# Legacy functions for backwards compatibility with internal code
def list_initial_strain_scenarios():
    """
    List available built-in initial strain scenarios
    
    Returns:
        Dict mapping scenario names to descriptions
    """
    return {name: data['description'] for name, data in INITIAL_STRAIN_SCENARIOS.items()}


def list_fitness_scenarios():
    """
    List available built-in fitness scenarios
    
    Returns:
        Dict mapping scenario names to descriptions
    """
    return {name: data['description'] for name, data in FITNESS_HYPOTHESES.items()}


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
        initial_strains = list(INITIAL_STRAIN_SCENARIOS[initial_strains]['strains'].keys())
        
    for i, strain in enumerate(initial_strains):
        if not isinstance(strain, (list, tuple)) or len(strain) != 2:
            raise ValueError(f"Strain {i} must be a (G,P) tuple, got {strain}")
            
        G, P = strain
        if not isinstance(G, int) or not isinstance(P, int):
            raise ValueError(f"G and P must be integers, got G={G} (type {type(G)}), P={P} (type {type(P)})")
            
        if G <= 0 or P <= 0:
            raise ValueError(f"G and P must be positive integers, got G={G}, P={P}")
    
    return True
