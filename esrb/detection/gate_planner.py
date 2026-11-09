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
    boundary_catalog=(),
    boundary_limit: int = 0,
    reduction_budget: int = 0,
) -> dict:
    """Reserve fresh slots and select at most one historical replay per reset.

    Sample relevant resets uniformly without replacement when slots are limited.
    In all-replay mode, require every historical reset in this scenario.
    """
    if not 1 <= fresh_min <= budget:
        raise ValueError("fresh_min must be between 1 and the gate budget")

    impacted = graph.impacted_components(changed)
    required_tests = set(graph.required_tests(changed))
    scheduled_tests = required_tests & supported_tests
    boundaries = sorted(
        (case for case in boundary_catalog if scheduled_tests.intersection(case["tests"])),
        key=lambda case: -len(scheduled_tests.intersection(case["tests"])),
    )
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
    if not replay_all:
        replay_capacity = max(0, replay_capacity - min(len(boundaries), boundary_limit) - reduction_budget)
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
        case = {
            "phase": "replay", "seed": source.seed, "reset_key": reset_key,
            "counterexample_ids": [cx.id for cx in entries],
        }
        if source.disturbance is not None:
            case["disturbance"] = source.disturbance
        replays.append(case)
    cases = []
    reserved_reduction = 0
    if not blocked:
        spare = budget - required_budget
        boundaries = [
            case for case in boundaries
            if all(case["disturbance"] != replay.get("disturbance") for replay in replays)
        ][:min(spare, boundary_limit)]
        reserved_reduction = min(reduction_budget, spare - len(boundaries))
        if reserved_reduction < 2:
            reserved_reduction = 0
        cases.extend(replays)
        cases.extend(
            {"phase": "boundary", "seed": fresh_seed + slot, "boundary_id": case["id"],
             "test_ids": sorted(scheduled_tests.intersection(case["tests"])),
             "disturbance": case["disturbance"]}
            for slot, case in enumerate(boundaries)
        )
        cases.extend(
            {"phase": "fresh", "seed": fresh_seed + len(boundaries) + slot}
            for slot in range(budget - len(cases) - reserved_reduction)
        )

    return {
        "changed_components": sorted(component.value for component in changed),
        "impacted_components": sorted(component.value for component in impacted),
        "required_tests": sorted(required_tests),
        "scheduled_tests": sorted(scheduled_tests) if not blocked else [],
        "unmapped_required_tests": sorted(required_tests - supported_tests),
        "supported_tests": sorted(supported_tests),
        "budget": budget, "fresh_min": fresh_min, "required_budget": required_budget,
        "replay_mode": "all" if replay_all else "capped",
        "replay_sampling_seed": sampling_seed,
        "replay_capacity": replay_capacity,
        "eligible_replay_count": len(witnesses),
        "skipped_replay_count": len(witnesses) - replay_count,
        "mandatory_replays": replays if replay_all else [],
        "reduction_reserved": reserved_reduction,
        "blocked_reason": "gate_budget_shortfall" if blocked else None,
        "cases": cases,
    }
