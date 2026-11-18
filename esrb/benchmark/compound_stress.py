"""Paired reset panels with simultaneous perturbations on existing scene axes."""
import random

from yaml import safe_load

from esrb.benchmark.fixed_chain import PANEL_SIZE
from esrb.detection.gate_cases import gate_case

FRACTION_RANGE = (0.60, 0.85)
PROMPT_STYLES = ("detailed", "brief")


def load_profiles(path, scenarios):
    """Load physical reset directions and reject unsupported scene axes."""
    profiles = safe_load(path.read_text())
    if not isinstance(profiles, dict) or profiles.keys() != scenarios.keys():
        raise ValueError("compound stress profiles must cover the configured scenarios")
    for name, profile in profiles.items():
        bounds = scenarios[name][0].reset_bounds
        if not isinstance(profile, dict) or len(profile) < 2 or profile.keys() - bounds.keys():
            raise ValueError(f"invalid compound stress reset fields for {name}")
        if any(direction not in ("high", "low", "edge") for direction in profile.values()):
            raise ValueError(f"invalid compound stress reset direction for {name}")
    return profiles


def compound_cases(task, profile, seed):
    """Generate ten replayable cases with the same seeds in both prompt arms."""
    rng = random.Random(seed)
    seeds = rng.sample(range(118500000, 119500000), PANEL_SIZE)
    cases = []
    for index, case_seed in enumerate(seeds):
        point = dict(task.reset_reference)
        for axis, (field, direction) in enumerate(profile.items()):
            low, high = task.reset_bounds[field]
            use_low = direction == "low" or (direction == "edge" and (index + axis) % 2 == 0)
            endpoint = low if use_low else high
            point[field] += rng.uniform(*FRACTION_RANGE) * (endpoint - point[field])
        cases.append(gate_case(task, case_seed, "fixed_chain", point))
    return cases
