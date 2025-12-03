"""
Utility functions for multi-strain Rotavirus simulations
Provides convenient functions for generating strain combinations and fitness scenarios
"""

# Standard library imports
import itertools

import numpy as np
import starsim as ss
import sciris as sc


def _init_prevalence_by_age(self, sim, uids):
    """
    Callback function to set age-specific initial prevalence probabilities based on defined age distribution. Intended
    to be passed to a ss.Bernoulli dist as the p parameter.

    Args:
        self: Rotavirus instance
        sim: Simulation instance
        uids: Array of unique IDs for the population
    """
    p = np.zeros(len(uids))
    total_pop = len(uids)
    n_infections_expected = int(total_pop * self.pars.initial_population_prevalence)
    for age_min, age_max, percent in self.pars.init_age_dist:
        age_group_members = ((sim.people.age >= age_min) & (sim.people.age < age_max)).uids
        age_group_pop = len(age_group_members)
        n_infections_in_age_group = percent * n_infections_expected
        p[age_group_members] = n_infections_in_age_group / age_group_pop
        print(f" ({age_min}-{age_max}): percent of infections: {percent} / total infections: {n_infections_in_age_group} / age group size: {age_group_pop} / prob per agent in age group: {n_infections_in_age_group / age_group_pop}")
    return p

DEFAULT_INITIAL_INFECTION_AGE_DIST = [
    (0, 1, 0.138),  # <1 year: 13.8% of infections
    (1, 2, 0.277),  # 1-2 years: 27.7% of infections
    (2, 5, 0.469),  # 2-5 years: 46.9% of infections
    (5, 200, 0.116),  # ≥5 years: 11.6% of infections
]

SCENARIO_DEFAULTS = {
    "fitness": 0.0,
    "initial_population_prevalence": 0.0,
    "init_prevalence_callback": _init_prevalence_by_age,
    "init_age_dist": DEFAULT_INITIAL_INFECTION_AGE_DIST,
    "dur_inf": ss.lognorm_ex(mean=13),
    "waning_delay": ss.days(0),
    "waning_rate_dist": ss.normal(loc=91, scale=14, unit="days"),
}

# Unified scenario system - contains strains, fitness, and prevalence all in one place
# IMPORTANT: If a strain does not specify fitness or prevalence, it will use the scenario default which are both 0. This
# means the strain will not spread unless explicitly declared in the scenario or overridden later. This is most likely to
# be a source of confusion for strains that emerge through reassortment and are not explicitly declared in the scenario.
SCENARIOS = {
    "simple": {
        "description": "Simple two-strain scenario - G1P8 and G2P4 with equal fitness and prevalence",
        "strains": {
            (1, 8): {
                "initial_population_prevalence": 0.03,
                "fitness": 1.0,
            },
            (2, 4): {
                "initial_population_prevalence": 0.01,
                "fitness": 1.0,
            },
        },
        "defaults": SCENARIO_DEFAULTS,
    },
    "single": {
        "description": "Simple strain scenario for debugging - G1P8",
        "strains": {
            (1, 8): {
                "fitness": 1.0,
                "initial_population_prevalence": 0.04,
            },
        },
        "defaults": SCENARIO_DEFAULTS,
    },
    "baseline": {
        "description": "Baseline scenario - common global strains with equal fitness",
        "strains": {
            (1, 8): {"fitness": 1.0, "initial_population_prevalence": 0.015},
            (2, 4): {"fitness": 1.0, "initial_population_prevalence": 0.008},
            (3, 8): {"fitness": 1.0, "initial_population_prevalence": 0.007},
        },
        "defaults": SCENARIO_DEFAULTS,
    },
    "realistic_competition": {
        "description": "G1P8 dominant with realistic strain competition",
        "strains": {
            (1, 8): {"fitness": 1.0, "initial_population_prevalence": 0.015},
            (2, 4): {"fitness": 0.2, "initial_population_prevalence": 0.008},
            (3, 8): {"fitness": 0.4, "initial_population_prevalence": 0.007},
            (4, 8): {"fitness": 0.5, "initial_population_prevalence": 0.005},
        },
        "defaults": SCENARIO_DEFAULTS,
    },
    "balanced_competition": {
        "description": "G1P8 dominant with moderate balanced competition",
        "strains": {
            (1, 8): {"fitness": 1.0, "initial_population_prevalence": 0.015},
            (2, 4): {"fitness": 0.6, "initial_population_prevalence": 0.008},
            (3, 8): {"fitness": 0.9, "initial_population_prevalence": 0.007},
            (4, 8): {"fitness": 0.9, "initial_population_prevalence": 0.005},
        },
        "defaults": SCENARIO_DEFAULTS,
    },
    "high_diversity": {
        "description": "High diversity with 12 strains and varied fitness",
        "strains": {
            (1, 8): {"fitness": 1.0, "initial_population_prevalence": 0.012},
            (2, 4): {"fitness": 0.7, "initial_population_prevalence": 0.007},
            (3, 8): {"fitness": 0.85, "initial_population_prevalence": 0.005},
            (4, 8): {"fitness": 0.88, "initial_population_prevalence": 0.004},
            (9, 8): {"fitness": 0.95, "initial_population_prevalence": 0.003},
            (12, 8): {"fitness": 0.93, "initial_population_prevalence": 0.003},
            (9, 6): {"fitness": 0.85, "initial_population_prevalence": 0.002},
            (12, 6): {"fitness": 0.90, "initial_population_prevalence": 0.002},
            (9, 4): {"fitness": 0.90, "initial_population_prevalence": 0.002},
            (1, 6): {"fitness": 0.6, "initial_population_prevalence": 0.002},
            (2, 8): {"fitness": 0.6, "initial_population_prevalence": 0.002},
            (2, 6): {"fitness": 0.6, "initial_population_prevalence": 0.002},
        },
        "defaults": SCENARIO_DEFAULTS,
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
                if verbose:
                    print(f"Warning: No preferred partners defined for G genotype {g}. Skipping.")
                continue
                # raise ValueError(f"No preferred partners defined for G genotype {g}")
            for p in unique_P:
                if p not in PREFERRED_PARTNERS[g]:
                    if verbose:
                        print(f"Warning: P genotype {p} is not a preferred partner for G genotype {g}. Skipping.")
                    continue
                all_reassortments.append((g, p))

    else:
        # Generate all possible combinations
        all_reassortments = list(itertools.product(unique_G, unique_P))

    return all_reassortments


def list_scenarios():
    """
    List available built-in simulation scenarios

    Returns:
        Dict mapping scenario names to descriptions
    """
    return {name: data["description"] for name, data in SCENARIOS.items()}


def get_scenario(scenario_name):
    """
    Get a scenario by name

    Args:
        scenario_name: Name of the scenario to retrieve

    Returns:
        Dict containing scenario data
    """
    if scenario_name not in SCENARIOS:
        raise ValueError(f"Unknown scenario '{scenario_name}'. Available scenarios: {list(SCENARIOS.keys())}")
    return SCENARIOS[scenario_name]


def validate_scenario(scenario):
    """
    Validate scenario format and content

    Args:
        scenario: Either a string (scenario name) or dict (custom scenario)

    Returns:
        Dict containing validated scenario data
    """
    if isinstance(scenario, str):
        # Built-in scenario
        return get_scenario(scenario)
    elif isinstance(scenario, dict):
        # Custom scenario - validate structure
        if "strains" not in scenario:
            raise ValueError("Custom scenario must contain 'strains' key")

        if not isinstance(scenario["strains"], dict):
            raise ValueError("Scenario 'strains' must be a dictionary")

        if len(scenario["strains"]) == 0:
            raise ValueError("Scenario must contain at least one strain")

        # Validate each strain entry
        for strain, data in scenario["strains"].items():
            if not isinstance(strain, tuple) or len(strain) != 2:
                raise ValueError(f"Strain key must be a (G,P) tuple, got {strain}")

            G, P = strain
            if not isinstance(G, int) or not isinstance(P, int):
                raise ValueError(f"G and P must be integers, got G={G}, P={P}")

            if G <= 0 or P <= 0:
                raise ValueError(f"G and P must be positive, got G={G}, P={P}")

            if not isinstance(data, dict):
                raise ValueError(f"Strain data must be dict, got {type(data)} for strain {strain}")

            if "fitness" not in data or "prevalence" not in data:
                raise ValueError(f"Strain data must contain 'fitness' and 'prevalence' keys for strain {strain}")

        # Set default fitness if not provided
        if "default_fitness" not in scenario:
            scenario["default_fitness"] = 1.0

        return scenario
    else:
        raise ValueError(f"Scenario must be string or dict, got {type(scenario)}")


def apply_scenario_overrides(scenario, override_fitness=None, override_prevalence=None, override_strains=None):
    """
    Apply override parameters to a scenario

    Args:
        scenario: Base scenario dict
        override_fitness: Override fitness values - float (all strains) or dict {(G,P): fitness}
        override_prevalence: Override prevalence values - float (all strains), callable (all strains), ss.Dist (all strains), or dict {(G,P): prevalence} (per-strain)
        override_strains: Add/modify strains - dict {(G,P): {'fitness': X, 'prevalence': Y}}

    Returns:
        Dict containing modified scenario
    """
    import copy

    result = copy.deepcopy(scenario)

    # Apply strain overrides first (adds/modifies strains)
    if override_strains is not None:
        if not isinstance(override_strains, dict):
            raise ValueError("override_strains must be a dict")
        for strain, data in override_strains.items():
            if not isinstance(strain, tuple) or len(strain) != 2:
                raise ValueError(f"Strain key must be (G,P) tuple, got {strain}")
            if not isinstance(data, dict):
                raise ValueError(f"Strain data must be dict, got {type(data)}")
            if "fitness" not in data or "init_prev" not in data:
                raise ValueError(f"Strain data must contain 'fitness' and 'init_prev' for {strain}")
            result["strains"][strain] = data.copy()

    # Apply fitness overrides
    if override_fitness is not None:
        if isinstance(override_fitness, (int, float)):
            # Apply to all strains
            for strain in result["strains"]:
                result["strains"][strain]["fitness"] = float(override_fitness)
        elif isinstance(override_fitness, dict):
            # Apply to specific strains
            for strain, fitness in override_fitness.items():
                if strain in result["strains"]:
                    result["strains"][strain]["fitness"] = float(fitness)
        else:
            raise ValueError("override_fitness must be number or dict")

    # Apply prevalence overrides
    if override_prevalence is not None:

        # if prevalence is an int, convert it to float
        if isinstance(override_prevalence, int):
            override_prevalence = float(override_prevalence)

        # if prevalence is a float or callable, assign it to all strains directly
        if isinstance(override_prevalence, float) or callable(override_prevalence):
            # Apply to all strains
            for strain in result["strains"]:
                result["strains"][strain]["init_prev"] = override_prevalence
        elif isinstance(override_prevalence, ss.Dist):
            for strain in result["strains"]:
                result["strains"][strain]["init_prev"] = sc.dcp(override_prevalence) # each strain needs its own copy of the dist

        # if prevalence is a dict, it can contain ints, floats, dists, or callables
        elif isinstance(override_prevalence, dict):
            # Apply to specific strains
            for strain, prevalence in override_prevalence.items():
                if strain in result["strains"]:
                    if isinstance(prevalence, float) or callable(prevalence):
                        result["strains"][strain]["init_prev"] = prevalence
                    elif isinstance(prevalence, int):
                        result["strains"][strain]["init_prev"] = float(prevalence)
                    elif isinstance(prevalence, ss.Dist):
                        result["strains"][strain]["init_prev"] = prevalence
                    else:
                        raise ValueError(f"Prevalence in dict must be float, int, or callable, got {type(prevalence)}")
        else:
            raise ValueError("override_prevalence must be number, callable, or dict of numbers or callables")

    return result
