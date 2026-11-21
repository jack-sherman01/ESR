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
from esrb.oracles.panda import validate_panda_measurements
from esrb.tasks.tool_handling import ToolHandlingTask
from esrb.update_operators.tool_grip import ToolGripRelaxationOperator
from esrb.utils.hashing import hash_dict
from experiments.builders import _build_tool_handling
from esrb.runtime.benchmark import run_loop


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
    assert task._model.neq == 1  # Native equality couples the fingers only.
    assert task._model.eq_type[0] == mujoco.mjtEq.mjEQ_JOINT
    assert task._model.eq_obj1id[0] == task._model.joint('finger_joint1').id
    assert task._model.eq_obj2id[0] == task._model.joint('finger_joint2').id
    assert task.action_space['shape'] == (8,)
    assert task._model.nu == 8 and task._model.nq == 16
    assert all(task._model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE for j in task._arm_joints)
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


def test_driver_tip_contacts_are_measured_with_full_resource_identity(traces):
    trace, _ = traces
    contacts = [contact for step in trace["contacts"] for contact in step]
    assert any({contact["geom1"], contact["geom2"]} == {"tool_bit", "work_surface"}
               for contact in contacts)
    assert trace["model"] == "panda_tool_workcell/3"
    assert trace["model_resource_digest"] == hash_dict(trace["model_resources"])
    assert "driver_handle.obj" in trace["model_resources"]
    assert "panda/assets/hand_0.obj" in trace["model_resources"]
    assert "panda/assets/link1.stl" in trace["model_resources"]
    validate_panda_measurements(trace)
    task = ToolHandlingTask()
    try:
        assert {task._model.geom(index).name for index in task._tool_geoms} == {
            "tool_handle", "tool_shaft", "tool_bit",
        }
        for index in range(task._model.ngeom):
            if task._model.geom_group[index] == 2:
                assert task._model.geom_contype[index] == task._model.geom_conaffinity[index] == 0
    finally:
        task.close()


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


def test_joint_actions_and_forward_kinematics_are_auditable(traces):
    trace, _ = traces
    q = np.asarray(trace['arm_joint_positions'])
    assert q.shape == (len(trace['times_s']), 7)
    assert np.count_nonzero(np.ptp(q, axis=0) > .02) >= 5
    assert np.shape(trace['commands']) == (len(q), 8)
    assert not any(trace['robot_collisions'])
    damaged = deepcopy(trace)
    damaged['arm_joint_positions'][10][0] += .1
    damaged['robot_physics'][10][-1]['qpos'][0] += .1
    with pytest.raises(ValueError, match='forward kinematics mismatch'):
        validate_panda_measurements(damaged)
    missing = deepcopy(trace)
    del missing['robot_physics']
    assert scenario_oracle('tool_handling').evaluate(missing, 0).status == 'UNKNOWN'
    task = ToolHandlingTask()
    try:
        task.reset(0)
        with pytest.raises(ValueError, match='8 finite actuator targets'):
            task.step(np.zeros(5))
    finally:
        task.close()


def test_arm_fixture_contact_is_reported_as_robot_collision():
    task = ToolHandlingTask()
    try:
        task.reset(0)
        # Deliberately initialize a reachable arm pose with fingers penetrating the fixture.
        task._pick = np.array([.10, 0., .045])
        task._reset_robot()
        mujoco.mj_forward(task._model, task._data)
        _, _, _, trace = task.step(np.r_[task._data.qpos[:7], 0.])
        assert trace['robot_collisions'] == [1]
        assert 'robot_collision' in scenario_oracle('tool_handling').evaluate(trace, 0).violated_constraints
    finally:
        task.close()


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
    corrupted = deepcopy(report)
    observed = corrupted["generations"][0]["baseline_measurements"][0]
    observed["raw_result"]["episode_data"]["model_resources"]["driver_handle.obj"] = "0" * 64
    observed["evidence_id"] = hash_dict(observed["raw_result"])
    with pytest.raises(ValueError, match="resource identity mismatch"):
        audit_gate(corrupted, archive)
    corrupted = deepcopy(report)
    observed = corrupted["generations"][0]["baseline_measurements"][0]
    observed["raw_result"]["episode_data"]["model"] = "unregistered/999"
    observed["evidence_id"] = hash_dict(observed["raw_result"])
    with pytest.raises(ValueError, match="unsupported tool measurement model"):
        audit_gate(corrupted, archive)


def test_tool_disturbances_support_independent_sampling():
    task = ToolHandlingTask
    protocol = sampling_protocol(task, 427, 3, {"friction": [0.7, 0.8]})
    draws = risk_cases(task, protocol, "tool_handling_grip_relaxation", 0)
    assert all(0.7 <= case["disturbance"]["friction"] <= 0.8 for case in draws)
    with pytest.raises(ValueError, match="risk bounds"):
        sampling_protocol(task, 427, 3, {"tool_mass_kg": [-1, 1]})
