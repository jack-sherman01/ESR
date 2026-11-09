import sys

import pytest

from esrb.benchmark import HazardType
from esrb.detection.change_impact import ChangeImpactGraph, ComponentType
from esrb.detection.counterexample import Counterexample, CounterexampleLibrary, CounterexampleSource
from esrb.detection.gate_planner import plan_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from experiments.run_benchmark import (
    _build_evaluator_drift, _build_scripted_peg, _build_sensor_drift, _build_tool_misuse,
    main, run_loop,
)


def test_impact_selects_only_relevant_history_from_earlier_generations():
    library = CounterexampleLibrary()
    for seed, component, scenario, gen in [
        (4, ComponentType.ROBOT_INTERFACE, "tool", 0),
        (5, ComponentType.EVALUATOR, "tool", 0),
        (6, ComponentType.ROBOT_INTERFACE, "other", 0),
        (7, ComponentType.ROBOT_INTERFACE, "tool", 1),
    ]:
        library.add(Counterexample.from_episode(
            {"force": 30}, HazardType.TOOL_PERMISSION_MISUSE, gen, ["force"],
            component.value,
            source=CounterexampleSource(scenario, str(seed), seed, "fresh", "policy"),
        ))

    options = {
        "changed": {ComponentType.TOOLS}, "scenario": "tool", "generation": 1,
        "fresh_min": 1, "fresh_seed": 99, "supported_tests": {"force_safety"},
    }
    graph = ChangeImpactGraph()
    plan = plan_gate(graph, library, budget=2, **options)

    assert [(case["phase"], case["seed"]) for case in plan["cases"]] == [
        ("replay", 4), ("fresh", 99),
    ]
    assert plan["impacted_components"] == ["robot_interface", "tools"]
    assert "force_safety" in plan["required_tests"]
    assert "permission_audit" in plan["unmapped_required_tests"]

    all_plan = plan_gate(graph, library, budget=3, replay_all=True, **options)
    assert [(case["phase"], case["seed"]) for case in all_plan["cases"]] == [
        ("replay", 4), ("replay", 5), ("fresh", 99),
    ]


def test_runner_replays_rejected_candidate_violation_under_same_budget(monkeypatch):
    instance, agent = _build_tool_misuse(2, 3, 0.5, 42)
    original_reset = instance.task.reset
    resets = []

    def record_reset(seed):
        resets.append(seed)
        return original_reset(seed=seed)

    monkeypatch.setattr(instance.task, "reset", record_reset)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0.5, 3, gate_planner="impact",
    )

    first, second = [row["gate_plan"] for row in rows]
    assert all(case["phase"] == "fresh" for case in first["executed"])
    replay, *fresh = second["executed"]
    assert replay["phase"] == "replay"
    assert replay["seed"] == first["executed"][0]["seed"]
    assert resets[9:] == [replay["seed"], *(case["seed"] for case in fresh)]
    assert second["used_rollouts"] == 3
    assert len(replay["counterexample_ids"]) == 3
    sources = {cx.id: cx for cx in loop.cx_library.all()}
    assert all(sources[cx_id].generation_found == 0 for cx_id in replay["counterexample_ids"])
    assert sources[replay["new_counterexample_id"]].source.phase == "replay"
    assert len(sources) == 6
    assert replay["safe"] is False
    assert all(row["accepted"] is False for row in rows)
    assert second["test_results"]["permission_audit"] == {
        "status": "fail", "unauthorized_permissions": ["unrestricted_force_control"],
    }
    assert second["test_results"]["tool_timing_check"]["status"] == "pass"
    assert second["test_results"]["force_safety"]["status"] == "fail"
    assert second["test_results"]["dynamics_safety"]["status"] == "unavailable"


def test_scheduled_checks_can_reject_without_changing_the_physical_selector(monkeypatch):
    instance, agent = _build_tool_misuse(1, 2, 0.5, 42)
    original_step = instance.task.step

    def corrupt_timing(action):
        obs, reward, done, episode = original_step(action)
        if done:
            episode.pop("command_steps")
        return obs, reward, done, episode

    monkeypatch.setattr(instance.task, "step", corrupt_timing)
    _, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=1.0), 0.5, 2, gate_planner="impact",
    )
    gate = rows[0]["gate_plan"]
    assert gate["test_results"]["tool_timing_check"]["status"] == "missing_evidence"
    assert set(gate["blocking_tests"]) == {"tool_timing_check", "permission_audit"}
    assert rows[0]["accepted"] is False
    assert gate["used_rollouts"] == 2


def test_evaluator_change_dispatches_integrity_and_cross_view_checks():
    instance, agent = _build_evaluator_drift(1, 2, 0.5, 42)
    _, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0.5, 2, gate_planner="impact",
    )
    gate = rows[0]["gate_plan"]
    assert gate["test_results"]["evaluator_integrity"]["status"] == "fail"
    assert gate["test_results"]["cross_view_agreement"]["status"] == "fail"
    assert gate["test_results"]["regression_suite"]["status"] == "not_run"
    assert gate["test_results"]["permission_audit"]["status"] == "unavailable"


def test_capped_replay_sampling_is_reproducible_and_uses_distinct_resets():
    library = CounterexampleLibrary()
    for seed in range(10):
        library.add(Counterexample.from_episode(
            {"force": 30}, HazardType.TOOL_PERMISSION_MISUSE, 0, ["force"],
            ComponentType.ROBOT_INTERFACE.value,
            source=CounterexampleSource("tool", str(seed), seed, "fresh", "policy"),
        ))
    options = {
        "changed": {ComponentType.TOOLS}, "scenario": "tool", "generation": 1,
        "budget": 4, "fresh_min": 1, "fresh_seed": 99, "supported_tests": {"force_safety"},
    }
    graph = ChangeImpactGraph()
    first = plan_gate(graph, library, replay_seed=42, **options)
    assert first == plan_gate(graph, library, replay_seed=42, **options)
    second = plan_gate(graph, library, replay_seed=43, **options)
    assert {case["reset_key"] for case in first["cases"][:-1]} != {
        case["reset_key"] for case in second["cases"][:-1]
    }
    assert len({case["reset_key"] for case in first["cases"][:-1]}) == 3
    assert first["cases"][-1] == {"phase": "fresh", "seed": 99}
    assert first["replay_sampling_seed"] == "42:tool:1"


def test_runner_blocks_shortfall_and_keeps_fixed_path_unchanged():
    selector = RiskSensitiveSelector()
    instance, agent = _build_tool_misuse(2, 1, 0.5, 42)
    loop, rows = run_loop(instance, agent, selector, 0.5, 1, gate_planner="impact", replay_all=True)
    blocked = rows[1]
    assert blocked["gate_plan"]["blocked_reason"] == "gate_budget_shortfall"
    assert blocked["gate_plan"]["required_budget"] == 2
    assert blocked["gate_plan"]["used_rollouts"] == 0
    assert blocked["candidate_safe"] is None
    assert blocked["accepted"] is False
    assert loop.agent.tool_permissions == {"grasp"}

    instance, agent = _build_tool_misuse(2, 1, 0.5, 42)
    loop, rows = run_loop(instance, agent, selector, 0.5, 1)
    assert all("gate_plan" not in row for row in rows)
    assert all(row["candidate_rollout_safe"] == [False] for row in rows)
    assert len(loop.cx_library) == 0


@pytest.mark.parametrize("replay_all", [False, True])
def test_seeded_task_respects_budget_with_distinct_historical_resets(replay_all):
    instance, agent = _build_scripted_peg(2, 2, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0.5, 2,
        gate_planner="impact", training_seed_stop=2100, replay_all=replay_all,
    )

    first, second = [row["gate_plan"] for row in rows]
    assert first["test_results"]["dynamics_safety"]["status"] == "fail"
    assert [case["seed"] for case in first["executed"]] == [2100, 2101]
    assert first["executed"][0]["reset_key"] != first["executed"][1]["reset_key"]
    assert second["eligible_replay_count"] == 2
    if replay_all:
        assert second["blocked_reason"] == "gate_budget_shortfall"
        assert [case["seed"] for case in second["mandatory_replays"]] == [2100, 2101]
        assert len(loop.cx_library) == 2
    else:
        assert second["blocked_reason"] is None
        assert second["skipped_replay_count"] == 1
        replay, fresh = second["executed"]
        assert replay["phase"] == "replay" and replay["seed"] in {2100, 2101}
        assert fresh["phase"] == "fresh" and fresh["seed"] == 2102
        assert second["replay_sampling_seed"] == "42:scripted_peg_speed_injection:1"
        assert second["test_results"]["regression_suite"]["failed_seeds"] == [replay["seed"]]


def test_sensor_drift_replay_restores_candidate_fault_and_calibrated_parent():
    instance, agent = _build_sensor_drift(2, 2, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0.5, 2, gate_planner="impact",
    )
    first, second = [row["gate_plan"] for row in rows]
    replay, fresh = second["executed"]
    assert second["changed_components"] == ["data"]
    assert replay["phase"] == "replay" and fresh["phase"] == "fresh"
    assert replay["seed"] == first["executed"][0]["seed"]
    assert replay["signal_values"]["peak_force_n"] == 30.0
    assert all(row["baseline_sigs"]["peak_force_n"] == 10.0 for row in rows)
    assert second["candidate_task_config"] == {"true_force": 30.0, "sensor_bias": -20.0}
    assert second["test_results"]["distribution_check"]["status"] == "fail"
    assert all(cx.source.task_config == second["candidate_task_config"] for cx in loop.cx_library.all())

@pytest.mark.parametrize(("options", "message"), [
    (["--rollouts", "0"], "--rollouts must be positive"),
    (["--gate-planner", "impact", "--fresh-min", "6"], "--fresh-min must be between"),
    (["--replay-all"], "--replay-all requires --gate-planner impact"),
    (["--gate-planner", "impact", "--replay-all", "--fresh-min", "5"],
     "requires --rollouts > --fresh-min"),
])
def test_invalid_budget_fails_before_run_directory(monkeypatch, tmp_path, capsys, options, message):
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--out-dir", str(out), *options])
    with pytest.raises(SystemExit) as error:
        main()
    assert error.value.code == 2
    assert message in capsys.readouterr().err
    assert not out.exists()
