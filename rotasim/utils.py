"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""

# Third-party imports
import sciris as sc

__all__ = ["SCENARIOS", "PREFERRED_PARTNERS", "generate_gp_reassortments", "list_scenarios", "get_scenario", "validate_scenario", "apply_scenario_overrides", "strain_pars"]

# Share of initial infections in each age group: (minimum age, maximum age, share)
DEFAULT_INIT_AGE_DIST = [
    (0, 1, 0.138),
    (1, 2, 0.277),
    (2, 5, 0.469),
    (5, 200, 0.116),
]

# What each strain has unless its scenario says otherwise
SCENARIO_DEFAULTS = dict(fitness=1.0, init_prevalence=0.0, init_age_dist=DEFAULT_INIT_AGE_DIST)

# Built-in scenarios: the strains at the start, with any other strains (the reassortants) at default_fitness
SCENARIOS = {
    "simple": {
        "description": "Simple two-strain scenario - G1P8 and G2P4 with equal fitness and prevalence",
        "strains": {
            (1, 8): {
                "init_prevalence": 0.03,
            },
            (2, 4): {
                "init_prevalence": 0.01
            },
        },
    },
    "single": {
        "description": "Simple strain scenario for debugging - G1P8",
        "strains": {
            (1, 8): {
                "fitness": 1.0,
                "init_prevalence": 0.04,
            },
        },
    },
    "baseline": {
        "description": "Baseline scenario - common global strains with equal fitness",
        "strains": {
            (1, 8): {"fitness": 1.0, "init_prevalence": 0.015},
            (2, 4): {"fitness": 1.0, "init_prevalence": 0.008},
            (3, 8): {"fitness": 1.0, "init_prevalence": 0.007},
        },
        "default_fitness": 1.0,
    },
    "realistic_competition": {
        "description": "G1P8 dominant with realistic strain competition",
        "strains": {
            (1, 8): {"fitness": 1.0, "init_prevalence": 0.015},
            (2, 4): {"fitness": 0.2, "init_prevalence": 0.008},
            (3, 8): {"fitness": 0.4, "init_prevalence": 0.007},
            (4, 8): {"fitness": 0.5, "init_prevalence": 0.005},
        },
        "default_fitness": 0.05,
    },
    "balanced_competition": {
        "description": "G1P8 dominant with moderate balanced competition",
        "strains": {
            (1, 8): {"fitness": 1.0, "init_prevalence": 0.015},
            (2, 4): {"fitness": 0.6, "init_prevalence": 0.008},
            (3, 8): {"fitness": 0.9, "init_prevalence": 0.007},
            (4, 8): {"fitness": 0.9, "init_prevalence": 0.005},
        },
        "default_fitness": 0.2,
    },
    "high_diversity": {
        "description": "High diversity with 12 strains and varied fitness",
        "strains": {
            (1, 8): {"fitness": 1.0, "init_prevalence": 0.012},
            (2, 4): {"fitness": 0.7, "init_prevalence": 0.007},
            (3, 8): {"fitness": 0.85, "init_prevalence": 0.005},
            (4, 8): {"fitness": 0.88, "init_prevalence": 0.004},
            (9, 8): {"fitness": 0.95, "init_prevalence": 0.003},
            (12, 8): {"fitness": 0.93, "init_prevalence": 0.003},
            (9, 6): {"fitness": 0.85, "init_prevalence": 0.002},
            (12, 6): {"fitness": 0.90, "init_prevalence": 0.002},
            (9, 4): {"fitness": 0.90, "init_prevalence": 0.002},
            (1, 6): {"fitness": 0.6, "init_prevalence": 0.002},
            (2, 8): {"fitness": 0.6, "init_prevalence": 0.002},
            (2, 6): {"fitness": 0.6, "init_prevalence": 0.002},
        },
        "default_fitness": 0.4,
    },
    "low_diversity": {
        "description": "Low diversity with 4 main competitive strains",
        "strains": {
            (1, 8): {"fitness": 0.98, "init_prevalence": 0.020},
            (2, 4): {"fitness": 0.7, "init_prevalence": 0.012},
            (3, 8): {"fitness": 0.8, "init_prevalence": 0.008},
            (4, 8): {"fitness": 0.8, "init_prevalence": 0.005},
        },
        "default_fitness": 0.5,
    },
    "emergence_scenario": {
        "description": "Scenario for studying strain emergence with weak background",
        "strains": {
            (1, 8): {"fitness": 1.0, "init_prevalence": 0.015},
            (2, 4): {"fitness": 0.4, "init_prevalence": 0.005},
            (3, 8): {"fitness": 0.7, "init_prevalence": 0.003},
        },
        "default_fitness": 0.05,  # Very low fitness for new emerging strains
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


def generate_gp_reassortments(initial_strains, use_preferred_partners=False):
    """
    Generate all G,P combinations of the initial strains

    Args:
        initial_strains: List of (G,P) tuples, e.g. [(1,8), (2,4)]
        use_preferred_partners: Whether to keep only the combinations in PREFERRED_PARTNERS (the initial strains are always kept)

    Returns:
        List of (G,P) tuples

    Example:
        >>> generate_gp_reassortments([(1,8), (2,4)])
        [(1, 4), (1, 8), (2, 4), (2, 8)]
    """
    if not len(initial_strains):
        raise ValueError("initial_strains cannot be empty")

    unique_G = sorted(set(g for g, p in initial_strains))
    unique_P = sorted(set(p for g, p in initial_strains))
    combinations = [(g, p) for g in unique_G for p in unique_P]
    if use_preferred_partners:
        combinations = [(g, p) for g, p in combinations if (g, p) in initial_strains or p in PREFERRED_PARTNERS.get(g, [])]
    return combinations


def list_scenarios():
    """List the built-in scenarios, as a dict of names and descriptions"""
    return {name: data["description"] for name, data in SCENARIOS.items()}


def get_scenario(scenario_name):
    """Get a built-in scenario by name"""
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_name]


def check_strain(strain, data):
    """Check one entry of a scenario's strains"""
    is_pair = isinstance(strain, tuple) and len(strain) == 2 and all(isinstance(x, int) and x > 0 for x in strain)
    if not is_pair:
        raise ValueError(f"Strain key must be a (G,P) tuple of positive integers, got {strain}")
    if not isinstance(data, dict):
        raise ValueError(f"Strain data must be dict, got {type(data)} for strain {strain}")


def validate_scenario(scenario):
    """
    Check a scenario, and return it as a dict

    Args:
        scenario: The name of a built-in scenario, or a dict with "strains": {(G,P): {"fitness": x, "init_prevalence": y}}, and optionally "default_fitness"
    """
    if isinstance(scenario, str):
        return get_scenario(scenario)
    if not isinstance(scenario, dict):
        raise ValueError(f"Scenario must be string or dict, got {type(scenario)}")
    if not scenario.get("strains"):
        raise ValueError("Scenario must contain at least one strain, in 'strains'")
    for strain, data in scenario["strains"].items():
        check_strain(strain, data)
    return scenario


def apply_scenario_overrides(scenario, override_fitness=None, override_prevalence=None, override_strains=None):
    """
    Apply overrides to a copy of a scenario

    Args:
        scenario: Base scenario dict
        override_fitness: Fitness of all strains (number), or by strain ({(G,P): fitness})
        override_prevalence: Initial prevalence of all strains, or by strain ({(G,P): prevalence}), replacing the scenario's: a number (a share of all ages), a function, or an ss.Dist
        override_strains: Strains to add or replace ({(G,P): {"fitness": x, "init_prevalence": y}})

    Returns:
        The modified scenario
    """
    result = sc.dcp(scenario)
    strains = result["strains"]
    for strain, data in (override_strains or {}).items():
        check_strain(strain, data)
        strains[strain] = dict(data)
    for key, override in [("fitness", override_fitness), ("init_prev", override_prevalence)]:
        if override is not None:
            values = override if isinstance(override, dict) else dict.fromkeys(strains, override)
            for strain, value in values.items():
                strains.setdefault(strain, {})[key] = sc.dcp(value) # Each strain needs its own copy of a distribution
    return result


def strain_pars(scenario, strain):
    """The parameters of a strain in a scenario: the defaults, then the scenario's default fitness, then its own"""
    fitness = {"fitness": scenario["default_fitness"]} if "default_fitness" in scenario else {}
    return sc.mergedicts(SCENARIO_DEFAULTS, fitness, scenario["strains"].get(strain))
