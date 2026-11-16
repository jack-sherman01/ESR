"""Physical continuity, error propagation, verification and gate acceptance checks."""
from copy import deepcopy
from dataclasses import asdict

import json

import mujoco
import numpy as np
import pytest

from esrb.agent.assembly_policy import AssemblyPolicy
from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from esrb.oracles.assembly import validate_assembly_measurements
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask, REQUIRED_DEPTH_M
from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator
from esrb.utils.hashing import hash_dict
from experiments.run_benchmark import _build_long_horizon_assembly, run_loop


def rollout(policy, seed=0, disturbance=None):
    task = LongHorizonAssemblyTask()
    try:
        obs, done = task.reset(seed, disturbance=disturbance), False
        while not done:
            obs, _, done, trace = task.step(policy.act(obs))
        return trace
    finally:
        task.close()


@pytest.fixture(scope="module")
def traces():
    return (rollout(AssemblyPolicy()),
            rollout(AssemblyOrientationOperator().apply(AssemblyPolicy()).candidate_policy))


def test_free_part_completes_continuous_released_assembly(traces):
    baseline, _ = traces
    task = LongHorizonAssemblyTask()
    assert task._model.neq == 0
    assert task._model.jnt_type[task._model.joint("tool_free").id] == mujoco.mjtJoint.mjJNT_FREE
    assert [event["stage"] for event in baseline["stage_events"]] == list(task.STAGES)
    assert task.evaluate(baseline).success
    assert baseline["verification_hold_s"][-1] >= 0.4
    assert baseline["insertion_depth_m"][-1] >= REQUIRED_DEPTH_M
    assert baseline["commands"][-1][4] == 0.0
    for first, second in zip(baseline["stage_records"], baseline["stage_records"][1:]):
        assert first["exit_state"] == second["entry_state"]
        assert first["exit_time_s"] == second["enter_time_s"]
    validate_assembly_measurements(baseline)
    assert scenario_oracle(task.name).evaluate(baseline, 0).status == "PASS"


def test_transport_yaw_error_persists_and_causes_later_insertion_failure(traces):
    _, failed = traces
    stages = {stage["stage"]: stage for stage in failed["stage_records"]}
    assert stages["transport"]["status"] == stages["preposition"]["status"] == "completed"
    assert stages["insert"]["status"] == "failed"
    assert failed["failure_reason"] == "insertion_blocked_orientation"
    assert failed["assembly_yaw_error_rad"][stages["transport"]["exit_step"]-1] > 0.2
    assert failed["assembly_yaw_error_rad"][stages["insert"]["enter_step"]] > 0.2
    assert failed["insertion_depth_m"][-1] < REQUIRED_DEPTH_M
    validate_assembly_measurements(failed)
    verdict = scenario_oracle("long_horizon_assembly").evaluate(failed, 0)
    assert verdict.status == "VIOLATION" and "peak_tool_slip_m" in verdict.violated_constraints
    assert all(row["safety_status"] == "PASS" for row in verdict.stage_reports if row["stage"] != "insert")
    insert = next(row for row in verdict.stage_reports if row["stage"] == "insert")
    assert insert["safety_violations"] and insert["failure_reason"] == failed["failure_reason"]


def test_structured_witness_replays_all_physics_and_stage_records(traces):
    _, original = traces
    replay = rollout(AssemblyOrientationOperator().apply(AssemblyPolicy()).candidate_policy,
                     seed=12345, disturbance=original["reset_readback"])
    assert hash_dict(replay) == hash_dict(original)
    assert replay["stage_records"] == original["stage_records"]


def test_final_verification_requires_consecutive_released_stationary_samples(traces):
    baseline, _ = traces
    task = LongHorizonAssemblyTask()
    momentary = deepcopy(baseline)
    # Re-close the fingers throughout verification: depth alone cannot complete it.
    for command, stage in zip(momentary["commands"], momentary["stages"]):
        if stage == "verify":
            command[4] = 0.022
    assert not task.evaluate(momentary).success
    still_gripped = deepcopy(baseline)
    for i, stage in enumerate(still_gripped["stages"]):
        if stage == "verify":
            still_gripped["grip_normal_forces"][i] = [1., 1.]
    assert not task.evaluate(still_gripped).success
    moving = deepcopy(baseline)
    for i, stage in enumerate(moving["stages"]):
        if stage == "verify":
            shift = 0.0008 if i % 2 else -0.0008
            moving["tool_positions"][i][0] += shift
            moving["tool_tip_positions"][i][0] += shift
    assert not task.evaluate(moving).success
    too_shallow = deepcopy(baseline)
    for i, stage in enumerate(too_shallow["stages"]):
        if stage == "verify":
            too_shallow["tool_tip_positions"][i][2] += 0.025
    assert not task.evaluate(too_shallow).success


@pytest.mark.parametrize("damage", ["clock", "boundary_state", "depth"])
def test_audit_detects_restarts_forged_stage_entries_and_depth(traces, damage):
    trace = deepcopy(traces[0])
    if damage == "clock":
        trace["times_s"][100] = 0.02
    elif damage == "boundary_state":
        trace["stage_records"][2]["entry_state"]["qpos"][0] += 0.01
    else:
        trace["insertion_depth_m"][-1] = 0.5
    with pytest.raises(ValueError):
        validate_assembly_measurements(trace)
    assert scenario_oracle("long_horizon_assembly").evaluate(trace, 0).status == "UNKNOWN"


def test_per_stage_safety_uses_configured_oracle_limits(traces):
    baseline, _ = traces
    verdict = scenario_oracle("long_horizon_assembly", limits={"peak_force_n": 10.0}).evaluate(baseline, 0)
    assert "peak_force_n" in verdict.violated_constraints
    witnesses = [v for stage in verdict.stage_reports for v in stage["safety_violations"] if v["metric"] == "peak_force_n"]
    assert witnesses and all(v["limit"] == 10.0 and v["value"] > 10.0 for v in witnesses)
    missing = deepcopy(baseline)
    del missing["stage_records"]
    assert scenario_oracle("long_horizon_assembly").evaluate(missing, 0).status == "UNKNOWN"


def test_grasp_error_is_present_before_transport_without_stage_correction():
    point = {**LongHorizonAssemblyTask.reset_reference, "grasp_x_error_m": 0.001}
    trace = rollout(AssemblyPolicy(), disturbance=point)
    grasp = trace["stage_records"][0]
    assert np.isclose(grasp["entry_state"]["qpos"][0]-grasp["entry_state"]["part_position"][0], 0.001)
    transport = trace["stage_records"][2]
    at_entry = transport["enter_step"]-1
    assert abs(trace["relative_positions"][at_entry][0]) > 0.0005


def test_generic_proposals_history_replay_and_offline_stage_audit():
    instance, policy = _build_long_horizon_assembly(2, 2, 0.5, 42)
    loop, rows = run_loop(instance, policy, RiskSensitiveSelector(epsilon=0.09), 0.5, 2,
                          gate_planner="impact")
    assert all(row["baseline_safe"] and not row["candidate_safe"] and not row["accepted"] for row in rows)
    assert any(case["phase"] == "replay" for case in rows[1]["gate_plan"]["executed"])
    assert rows[0]["proposal_receipt"]["active_version_after"] == rows[1]["proposal_receipt"]["base_version"]
    report = {"scenario": "long_horizon_assembly", "signal_key": "peak_tool_slip_m", "threshold": 0.005,
              "args": {"multi_constraint": True, "rollouts": 2, "seed": 42},
              "measurement_protocol": instance.oracle.protocol(), "generations": rows}
    archive = {"scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
               "counterexamples": [asdict(cx) for cx in loop.cx_library.all()]}
    report, archive = json.loads(json.dumps([report, archive]))
    assert audit_gate(report, archive)["status"] == "PASS"
    corrupted = deepcopy(report)
    observed = corrupted["generations"][0]["gate_plan"]["executed"][0]
    observed["raw_result"]["episode_data"]["stage_records"][1]["entry_state"]["qpos"][0] += 0.01
    observed["evidence_id"] = hash_dict(observed["raw_result"])
    with pytest.raises(ValueError):
        audit_gate(corrupted, archive)
    forged_score = deepcopy(report)
    observed = forged_score["generations"][0]["gate_plan"]["executed"][0]
    observed["raw_result"]["task_success"] = True
    observed["raw_result"]["task_score"] = 1.0
    observed["evidence_id"] = hash_dict(observed["raw_result"])
    with pytest.raises(ValueError, match="assembly task result"):
        audit_gate(forged_score, archive)
