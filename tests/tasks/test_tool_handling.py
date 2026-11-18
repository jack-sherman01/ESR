from copy import deepcopy
from dataclasses import asdict
import json

import mujoco
import numpy as np
import pytest

from esrb.agent.tool_handling_policy import ToolHandlingPolicy
from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_sampling import risk_cases, sampling_protocol
from esrb.detection.risk_selection import RiskSensitiveSelector
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.oracles.tool_retention import validate_tool_measurements
from esrb.tasks.tool_handling import ToolHandlingTask
from esrb.update_operators.tool_grip import ToolGripRelaxationOperator
from esrb.utils.hashing import hash_dict
from experiments.run_benchmark import _build_tool_handling, run_loop


def rollout(task, policy, *, seed=0, disturbance=None):
    obs = task.reset(seed=seed, disturbance=disturbance)
    done = False
    while not done:
        obs, _, done, trace = task.step(policy.act(obs))
    return trace


@pytest.fixture(scope="module")
def traces():
    task = ToolHandlingTask()
    baseline = rollout(task, ToolHandlingPolicy())
    child = ToolGripRelaxationOperator().apply(ToolHandlingPolicy()).candidate_policy
    dropped = rollout(task, child)
    return baseline, dropped


def test_free_tool_completes_all_physical_stages(traces):
    trace, _ = traces
    task = ToolHandlingTask()
    assert task._model.neq == 0
    assert task._model.jnt_type[task._model.joint("tool_free").id] == mujoco.mjtJoint.mjJNT_FREE
    assert [event["stage"] for event in trace["stage_events"]] == [
        "grasp", "lift", "transport", "orient", "operate", "done",
    ]
    assert task.evaluate(trace).success
    assert max(trace["surface_forces"]) >= 2
    assert min(trace["grip_normal_forces"][30]) > 0
    verdict = scenario_oracle("tool_handling").evaluate(trace, 0)
    assert verdict.safe
    assert verdict.signal_values["peak_tool_slip_m"] < 0.005
    validate_tool_measurements(trace)


def test_relaxed_grip_causes_physical_slip_and_drop(traces):
    _, trace = traces
    assert not ToolHandlingTask().evaluate(trace).success
    verdict = scenario_oracle("tool_handling").evaluate(trace, 0)
    assert {"peak_tool_slip_m", "tool_drop"} <= set(verdict.violated_constraints)
    tool = np.asarray(trace["tool_positions"])
    gripper = np.asarray(trace["ee_positions"])
    assert np.max(tool[:, 2]) > 0.145
    assert tool[-1, 2] < 0.10 and gripper[-1, 2] > 0.19
    assert np.linalg.norm(tool[-1] - gripper[-1]) > 0.10
    validate_tool_measurements(trace)


def test_same_structured_reset_replays_identical_motion(traces):
    task = ToolHandlingTask()
    point = task.disturbance_for_seed(0)
    second = rollout(task, ToolHandlingPolicy(), seed=1234, disturbance=point)
    baseline, _ = traces
    assert second["reset_readback"] == point
    for channel in ("tool_positions", "tool_quaternions", "grip_normal_forces", "contact_forces"):
        np.testing.assert_array_equal(second[channel], baseline[channel])
    assert task._model.body_mass[task._tool] == point["tool_mass_kg"]
    assert task._model.geom_friction[task._handle, 0] == point["friction"]


def test_retention_rule_missing_data_is_unknown_and_substep_tampering_is_detected(traces):
    baseline, _ = traces
    damaged = deepcopy(baseline)
    del damaged["tool_quaternions"]
    verdict = scenario_oracle("tool_handling").evaluate(damaged, 0)
    assert verdict.status == "UNKNOWN"
    damaged = deepcopy(baseline)
    damaged["contact_samples"][0][0][1] += 100
    with pytest.raises(ValueError, match="reduction mismatch"):
        validate_tool_measurements(damaged)


def test_grip_update_replays_counterexamples_and_audits_proposals():
    instance, agent = _build_tool_handling(2, 2, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=0.09), 0.5, 2, gate_planner="impact",
    )
    assert all(row["baseline_safe"] and not row["candidate_safe"] and not row["accepted"] for row in rows)
    assert any(case["phase"] == "replay" for case in rows[1]["gate_plan"]["executed"])
    report = {
        "scenario": "tool_handling", "signal_key": "peak_tool_slip_m", "threshold": 0.005,
        "args": {"multi_constraint": True, "rollouts": 2, "seed": 42},
        "measurement_protocol": instance.oracle.protocol(), "generations": rows,
    }
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    report, archive = json.loads(json.dumps([report, archive]))
    assert audit_gate(report, archive)["status"] == "PASS"
    corrupted = deepcopy(report)
    observed = corrupted["generations"][0]["baseline_measurements"][0]
    observed["raw_result"]["episode_data"]["contact_samples"][0][0][1] += 100
    observed["evidence_id"] = hash_dict(observed["raw_result"])
    with pytest.raises(ValueError, match="reduction mismatch"):
        audit_gate(corrupted, archive)


def test_tool_disturbances_support_independent_sampling():
    task = ToolHandlingTask
    protocol = sampling_protocol(task, 427, 3, {"friction": [0.7, 0.8]})
    draws = risk_cases(task, protocol, "tool_handling_grip_relaxation", 0)
    assert all(0.7 <= case["disturbance"]["friction"] <= 0.8 for case in draws)
    with pytest.raises(ValueError, match="risk bounds"):
        sampling_protocol(task, 427, 3, {"tool_mass_kg": [-1, 1]})
