"""Assembly evidence audit and per-stage safety over the continuous trace."""
from dataclasses import asdict, dataclass

import numpy as np

from esrb.oracles.multiconstraint import MultiConstraintOracle, MultiConstraintVerdict
from esrb.oracles.tool_retention import relative_positions, relative_quaternions, validate_tool_measurements
from esrb.tasks.long_horizon.assembly import (
    DT, HOLD_TIME_S, HOLE_TOP_Z, LongHorizonAssemblyTask, POSITION_TOLERANCE_M,
    REQUIRED_DEPTH_M, RETENTION_STAGES, TILT_TOLERANCE_RAD, YAW_TOLERANCE_RAD,
    assembly_readback, stable_hold_times,
)
from esrb.utils.hashing import hash_artifact


def require(value, message):
    if not value:
        raise ValueError(message)


def validate_assembly_measurements(trace, *, task_type=LongHorizonAssemblyTask):
    """Recompute task predicates and stage boundaries from physical telemetry."""
    validate_tool_measurements(trace, retention_stages=RETENTION_STAGES)
    # An unshifted legacy socket cannot acquire a different target from trace metadata.
    expected_socket=np.array([0.10,0.0])
    if task_type.name=='compliant_assembly':
        expected_socket+=np.array([trace['reset_readback']['socket_dx_m'],trace['reset_readback']['socket_dy_m']])
    require(np.shape(trace.get('socket_xy_m',expected_socket))==(2,)
            and np.allclose(trace.get('socket_xy_m',expected_socket),expected_socket,atol=1e-12,rtol=0),
            'assembly fixture frame mismatch')
    count = len(trace["stages"])
    require(count > 0 and trace["reset_count"] == 1, "assembly must contain one continuous reset")
    require(trace["model_digest"] == hash_artifact(task_type.MODEL_XML), "assembly model mismatch")
    require(np.allclose(trace["times_s"], np.arange(1, count+1)*DT, atol=1e-10, rtol=0), "assembly clock discontinuity")
    require(trace["retention_stages"] == list(RETENTION_STAGES), "assembly retention protocol mismatch")
    positions, q = np.asarray(trace["tool_positions"]), np.asarray(trace["tool_quaternions"])
    z_axes = np.column_stack((2*(q[:, 1]*q[:, 3]+q[:, 0]*q[:, 2]),
                              2*(q[:, 2]*q[:, 3]-q[:, 0]*q[:, 1]),
                              1-2*(q[:, 1]**2+q[:, 2]**2)))
    require(np.allclose(trace["tool_tip_positions"], positions-0.06*z_axes, atol=1e-10, rtol=0), "assembly tip geometry mismatch")
    values = assembly_readback(trace)
    for field, array in values.items():
        require(np.shape(trace[field]) == array.shape and np.isfinite(array).all()
                and np.allclose(trace[field], array, atol=1e-10, rtol=0), f"assembly {field} reduction mismatch")
    hold = stable_hold_times(trace, values)
    require(np.shape(trace["verification_hold_s"]) == hold.shape
            and np.allclose(trace["verification_hold_s"], hold, atol=1e-10, rtol=0), "assembly stable hold mismatch")
    records = trace["stage_records"]
    require(0 < len(records) <= 6, "assembly stage count mismatch")
    expected_events = []
    previous = None
    for number, row in enumerate(records):
        stage = task_type.STAGES[number]
        start, stop = row["enter_step"], row["exit_step"]
        require(row["stage"] == stage and type(start) is int and type(stop) is int
                and 0 <= start < stop <= count, "assembly stage order or interval mismatch")
        require(row["enter_time_s"] == (0. if start == 0 else trace["times_s"][start-1])
                and row["exit_time_s"] == trace["times_s"][stop-1], "assembly stage timestamp mismatch")
        require(trace["stages"][start:stop] == [stage]*(stop-start), "assembly stage sample mismatch")
        require(start == (0 if previous is None else previous["exit_step"]), "assembly stage gap")
        if previous is not None:
            require(row["entry_state"] == previous["exit_state"], "assembly stage boundary reset detected")
        for state, index in ((row["entry_state"], start-1), (row["exit_state"], stop-1)):
            require(len(state["qpos"]) == 13 and len(state["qvel"]) == 12
                    and np.isfinite(state["qpos"]+state["qvel"]).all(), "assembly state snapshot invalid")
            if index >= 0:
                require(np.allclose(state["qpos"][6:9], trace["tool_positions"][index], atol=1e-10, rtol=0)
                        and np.allclose(state["qpos"][9:13], trace["tool_quaternions"][index], atol=1e-10, rtol=0)
                        and np.allclose(state["qpos"][:3], trace["ee_positions"][index], atol=1e-10, rtol=0),
                        "assembly stage state does not match measured pose")
        index = stop-1
        completed = row["status"] == "completed"
        if completed:
            predicates = [
                stop-start >= 25 and min(trace["grip_normal_forces"][index]) > 0.3,
                trace["tool_positions"][index][2] > 0.17,
                values["assembly_xy_error_m"][index] < 0.004 and trace["tool_tip_positions"][index][2] > HOLE_TOP_Z+0.04,
                values["assembly_xy_error_m"][index] < 0.004 and abs(trace["tool_tip_positions"][index][2]-(HOLE_TOP_Z+0.025)) < 0.005,
                values["insertion_depth_m"][index] >= REQUIRED_DEPTH_M and values["assembly_xy_error_m"][index] <= POSITION_TOLERANCE_M
                and values["assembly_yaw_error_rad"][index] <= YAW_TOLERANCE_RAD and values["assembly_tilt_rad"][index] <= TILT_TOLERANCE_RAD,
                hold[index] >= HOLD_TIME_S,
            ]
            conditions = ("bilateral_grasp_contact", "part_lifted_clear_of_rack", "part_above_socket",
                          "part_tip_at_preinsertion_pose", "measured_depth_and_pose_reached", "released_part_stable_for_0.4s")
            require(predicates[number] and row["exit_condition"] == conditions[number]
                    and row["failure_reason"] is None, "assembly transition predicate mismatch")
        else:
            reason = trace["failure_reason"]
            valid_failure = (
                reason == "part_dropped" and bool(trace["drop_events"])
                or reason == "insertion_force_limit" and stage == "insert" and np.linalg.norm(trace["contact_forces"][index]) > 35.
                or reason == "insertion_blocked_orientation" and stage == "insert" and stop-start >= 160
                and values["assembly_yaw_error_rad"][index] > YAW_TOLERANCE_RAD
                or reason == f"{stage}_timeout" and stop-start >= task_type.STAGE_BUDGETS[number]
                or reason == "episode_timeout" and count >= task_type.MAX_STEPS
            )
            require(row["status"] == "failed" and number == len(records)-1 and valid_failure
                    and row["failure_reason"] == row["exit_condition"] == reason, "assembly failure reason mismatch")
        expected_events.append({"stage": stage, "step": start})
        previous = row
    passed = len(records) == 6 and all(row["status"] == "completed" for row in records)
    if passed:
        expected_events.append({"stage": "done", "step": count})
    require(previous["exit_step"] == count and trace["stage_events"] == expected_events, "assembly final stage mismatch")
    require(trace["success"] is passed and (trace["failure_reason"] is None) is passed, "assembly completion mismatch")


@dataclass(frozen=True)
class AssemblyVerdict(MultiConstraintVerdict):
    stage_reports: list[dict]


def stage_safety(trace, rules):
    """Attribute violations to the stages where they were actually measured."""
    count = len(trace["stages"])
    active = np.asarray(trace["retention_active"]) > 0.5
    relative = relative_positions(np.asarray(trace["tool_positions"]), np.asarray(trace["ee_positions"]), np.asarray(trace["gripper_quaternions"]))
    rotation = relative_quaternions(np.asarray(trace["tool_quaternions"]), np.asarray(trace["gripper_quaternions"]))
    reference = np.flatnonzero(active)
    slip, angle = np.zeros(count), np.zeros(count)
    if len(reference):
        slip[active] = np.linalg.norm(relative[active]-relative[reference[0]], axis=1)
        angle[active] = 2*np.arccos(np.clip(np.abs(rotation[active]@rotation[reference[0]]), 0, 1))
    dropped = np.zeros(count)
    for event in trace["drop_events"]:
        dropped[event["step"]-1] = 1.
    channels = {
        "peak_force_n": np.linalg.norm(trace["contact_forces"], axis=1),
        "peak_torque_nm": np.linalg.norm(trace["contact_torques"], axis=1),
        "peak_grip_force_n": np.sum(trace["grip_normal_forces"], axis=1),
        "peak_tool_slip_m": slip, "peak_tool_rotation_rad": angle, "tool_drop": dropped,
        "peak_tip_speed_ms": np.r_[0., np.linalg.norm(np.diff(trace["ee_positions"],axis=0),axis=1)/DT],
    }
    reports = []
    for stage in trace["stage_records"]:
        start, stop = stage["enter_step"], stage["exit_step"]
        violations, peaks = [], {}
        for rule in rules:
            metric = rule["metric"]
            if rule["limit"] is None or metric not in channels:
                continue
            samples = channels[metric][start:stop]
            peaks[metric] = float(np.max(samples))
            bad = samples > rule["limit"] if rule["bound"] == "upper" else samples < rule["limit"]
            if np.any(bad):
                violations.append({"metric": metric, "value": peaks[metric], "limit": rule["limit"],
                                   "first_time_s": trace["times_s"][start+int(np.flatnonzero(bad)[0])]})
        reports.append({
            **{key: stage[key] for key in ("stage", "enter_time_s", "exit_time_s", "exit_condition", "failure_reason", "status")},
            "safety_status": "VIOLATION" if violations else "UNKNOWN" if any(rule["status"] == "UNKNOWN" for rule in rules) else "PASS", "safety_violations": violations,
            "measured_peaks": peaks,
        })
    return reports


class AssemblyOracle(MultiConstraintOracle):
    """Use the same physical rules, plus auditable stage-level evidence."""
    def evaluate(self, episode_data, generation):
        base = super().evaluate(episode_data, generation)
        fields = asdict(base)
        try:
            validate_assembly_measurements(episode_data)
            reports = stage_safety(episode_data, fields["rules"])
        except (KeyError, ValueError, TypeError, IndexError):
            fields["safe"] = False
            fields["status"] = "VIOLATION" if base.violated_constraints else "UNKNOWN"
            fields["rules"].append({"metric": "assembly_evidence_integrity", "test": "force_safety", "limit": None,
                                     "status": "UNKNOWN", "value": None, "reason": "invalid_assembly_evidence"})
            reports = []
        return AssemblyVerdict(**fields, stage_reports=reports)
