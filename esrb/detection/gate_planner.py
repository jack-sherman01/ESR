"""Schedule historical replay and new rollout slots within one gate budget."""

from random import Random

from esrb.detection.change_impact import ChangeImpactGraph, ComponentType
from esrb.detection.counterexample import CounterexampleLibrary


def plan_gate(
    graph: ChangeImpactGraph,
    library: CounterexampleLibrary,
    *,
    changed: set[ComponentType],
    scenario: str,
    generation: int,
    budget: int,
    fresh_min: int,
    fresh_seed: int,
    supported_tests: set[str],
    replay_all: bool = False,
    replay_seed: int = 0,
) -> dict:
    """Reserve fresh slots and select at most one historical replay per reset.

    Sample relevant resets uniformly without replacement when slots are limited.
    In all-replay mode, require every historical reset in this scenario.
    """
    if not 1 <= fresh_min <= budget:
        raise ValueError("fresh_min must be between 1 and the gate budget")

    impacted = graph.impacted_components(changed)
    required_tests = set(graph.required_tests(changed))
    if replay_all or not changed:
        eligible = library.all()
    else:
        eligible = library.relevant_to({component.value for component in impacted})
    witnesses = {}
    for cx in eligible:
        source = cx.source
        if source is not None and source.scenario == scenario and cx.generation_found < generation:
            witnesses.setdefault(source.reset_key, []).append(cx)

    replay_capacity = budget - fresh_min
    replay_count = len(witnesses) if replay_all else min(replay_capacity, len(witnesses))
    required_budget = replay_count + fresh_min
    blocked = required_budget > budget
    replay_keys = list(witnesses)
    sampling_seed = None
    if 0 < replay_count < len(witnesses):
        sampling_seed = f"{replay_seed}:{scenario}:{generation}"
        replay_keys = Random(sampling_seed).sample(sorted(witnesses), replay_count)

    replays = []
    for reset_key in replay_keys[:replay_count]:
        entries = witnesses[reset_key]
        source = entries[0].source
        assert source is not None
        replays.append({
            "phase": "replay", "seed": source.seed, "reset_key": reset_key,
            "counterexample_ids": [cx.id for cx in entries],
        })
    cases = []
    if not blocked:
        cases.extend(replays)
        cases.extend(
            {"phase": "fresh", "seed": fresh_seed + slot}
            for slot in range(budget - len(cases))
        )

    return {
        "changed_components": sorted(component.value for component in changed),
        "impacted_components": sorted(component.value for component in impacted),
        "required_tests": sorted(required_tests),
        "scheduled_tests": sorted(required_tests & supported_tests) if not blocked else [],
        "unmapped_required_tests": sorted(required_tests - supported_tests),
        "supported_tests": sorted(supported_tests),
        "budget": budget, "fresh_min": fresh_min, "required_budget": required_budget,
        "replay_mode": "all" if replay_all else "capped",
        "replay_sampling_seed": sampling_seed,
        "replay_capacity": replay_capacity,
        "eligible_replay_count": len(witnesses),
        "skipped_replay_count": len(witnesses) - replay_count,
        "mandatory_replays": replays if replay_all else [],
        "blocked_reason": "gate_budget_shortfall" if blocked else None,
        "cases": cases,
    }
