from copy import deepcopy
from dataclasses import asdict

import numpy as np
import pytest

from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from esrb.tasks.contact_rich import PegInsertionTask
from experiments.run_benchmark import _build_scripted_peg, run_loop


def test_explicit_reset_restores_seeded_or_custom_xy():
    task = PegInsertionTask()
    seeded = task.reset(42)
    resolved = task.reset(99, xy_offset_m=task.offset_for_seed(42))
    np.testing.assert_array_equal(resolved.state, seeded.state)
    first = task.reset(1, xy_offset_m=(0.012, -0.004))
    task.step(np.array([0.001, 0, 0]))
    second = task.reset(2, xy_offset_m=(0.012, -0.004))
    np.testing.assert_array_equal(first.state, second.state)
    with pytest.raises(ValueError, match="initial XY offset"):
        task.reset(xy_offset_m=(0.031, 0))


def test_peg_gate_runs_boundary_replay_reduction_and_audits_raw_evidence():
    instance, agent = _build_scripted_peg(2, 2, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0.5, 8, gate_planner="impact",
        fresh_min=2, peg_boundary_cases=2, peg_reduction_budget=3,
    )
    first, second = [row["gate_plan"] for row in rows]
    assert all(plan["used_rollouts"] == 8 for plan in (first, second))
    assert any(case["phase"] == "replay" for case in second["executed"])
    assert any(case.get("test_ids") == ["dynamics_safety"] for case in first["executed"])
    replays = {case["reset_key"] for case in second["executed"] if case["phase"] == "replay"}
    assert not replays.intersection(
        case["reset_key"] for case in second["executed"] if case["phase"] == "boundary"
    )
    assert first["reduction"]["status"] == "verified"
    assert first["reduction"]["original"] != first["reduction"]["reduced"]
    report = {"scenario": "scripted_peg", "threshold": 0.25, "generations": rows}
    archive = {
        "scenario": instance.name, "reset_mode": "structured", "task": instance.task.name,
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    audit = audit_gate(report, archive)
    assert audit["status"] == "PASS"
    assert sum(audit["rollouts_by_phase"].values()) == 16
    corrupted = deepcopy(report)
    corrupted["generations"][0]["gate_plan"]["executed"][0]["raw_result"]["oracle"]["safe"] = True
    with pytest.raises(ValueError, match="invalid raw evidence"):
        audit_gate(corrupted, archive)


def test_safe_peg_panel_returns_unused_reduction_budget_to_fresh():
    instance, agent = _build_scripted_peg(1, 2, 0, 42)
    _, rows = run_loop(
        instance, agent, RiskSensitiveSelector(), 0, 5, gate_planner="impact",
        fresh_min=2, peg_boundary_cases=1, peg_reduction_budget=2,
    )
    plan = rows[0]["gate_plan"]
    assert not plan["required_tests"]
    assert plan["reduction"] == {"status": "no_off_reference_unsafe_case", "evaluations": 0}
    assert [case["phase"] for case in plan["executed"]] == ["fresh"] * 5
