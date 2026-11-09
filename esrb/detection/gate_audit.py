"""Verify structured gate evidence from saved benchmark artifacts."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from esrb.detection.gate_cases import boundary_catalog, control_steps, gate_case
from esrb.oracles import ForceTorqueOracle, HeightClearanceOracle, ProximityOracle, VelocityOracle
from esrb.tasks.agentic import GraspForceTask, SensorDriftTask
from esrb.tasks.contact_rich import PegInsertionTask
from esrb.tasks.dynamic_workspace import DynamicObstacleTask
from esrb.tasks.long_horizon import PickPlaceTask
from esrb.tasks.tool_handling import ToolTightenTask
from esrb.utils.hashing import hash_dict


def _require(condition, message):
    if not condition:
        raise ValueError(message)


_TASKS = {task.name: task for task in (
    GraspForceTask, SensorDriftTask, PegInsertionTask, DynamicObstacleTask, PickPlaceTask, ToolTightenTask,
)}
_ORACLES = {
    "peak_tip_speed_ms": VelocityOracle,
    "min_obstacle_dist_m": ProximityOracle,
    "peak_transit_height_m": HeightClearanceOracle,
    "peak_force_n": ForceTorqueOracle,
}


def audit_gate(report, archive):
    legacy_peg = archive["reset_mode"] == "explicit_xy_offset" and report["scenario"] == "scripted_peg"
    _require(archive["reset_mode"] == "structured" or legacy_peg,
             "expected structured gate artifacts")
    task = PegInsertionTask if legacy_peg else _TASKS[archive["task"]]
    signal = report.get("signal_key", "peak_tip_speed_ms")
    oracle = _ORACLES[signal](report["threshold"])
    primary_test = "force_safety" if isinstance(oracle, ForceTorqueOracle) else "dynamics_safety"
    boundaries = {case["id"]: case for case in boundary_catalog(task, primary_test)}
    bank = {cx["id"]: cx for cx in archive["counterexamples"]}
    _require(len(bank) == len(archive["counterexamples"]), "duplicate counterexample ID")
    phases = Counter()
    total_steps = 0
    for row in report["generations"]:
        gen = row["generation"]
        plan = row["gate_plan"]
        executed = plan["executed"]
        count = len(executed)
        _require(count == plan["used_rollouts"] == len(plan["cases"]), f"generation {gen}: rollout count mismatch")
        if plan["blocked_reason"]:
            _require(count == 0, f"generation {gen}: blocked gate executed rollouts")
            continue
        _require(count == plan["budget"], f"generation {gen}: gate budget mismatch")
        used = Counter(case["phase"] for case in executed)
        _require(used["fresh"] >= plan["fresh_min"] and used["reduction"] <= plan["reduction_reserved"],
                 f"generation {gen}: fresh or reduction budget mismatch")
        phases.update(used)
        by_id = {}
        for case, observed in zip(plan["cases"], executed, strict=True):
            raw = observed["raw_result"]
            trace = raw["episode_data"]
            _require(all(observed[key] == value for key, value in case.items()) and raw["case"] == case
                     and raw["policy_digest"] == plan["candidate_policy_digest"],
                     f"generation {gen}: candidate or case mismatch")
            point = case["disturbance"]
            canonical = gate_case(task, case["seed"], case["phase"], point)
            if legacy_peg:
                # Legacy reports used the /xy/ namespace with the same canonical offset.
                key = "peg_insertion/xy/" + hash_dict(canonical["disturbance"])
                applied = dict(zip(task.reset_bounds, trace["initial_xy_offset_m"], strict=True))
            else:
                key = canonical["reset_key"]
                applied = trace["reset_readback"]
            _require(key == case["reset_key"] and applied == point, f"generation {gen}: reset mismatch")
            if not legacy_peg and case["phase"] == "boundary":
                boundary = boundaries[case["boundary_id"]]
                _require(point == boundary["disturbance"]
                         and case["test_ids"] == boundary["tests"]
                         and set(case["test_ids"]) <= set(plan["scheduled_tests"]),
                         f"generation {gen}: boundary case does not match the scheduled test")
            if not legacy_peg and case["phase"] == "fresh":
                _require(point == gate_case(task, case["seed"], "fresh")["disturbance"],
                         f"generation {gen}: fresh case does not match its seed")
            _require(trace.get("applied_task_config") == plan.get("candidate_task_config"),
                     f"generation {gen}: applied task configuration mismatch")
            verdict = asdict(oracle.evaluate(trace, gen))
            evidence_id = observed["evidence_id"]
            _require(raw["oracle"] == verdict and observed["safe"] == verdict["safe"]
                     and observed["signal_values"] == verdict["signal_values"]
                     and evidence_id == hash_dict(raw), f"generation {gen}: invalid raw evidence")
            _require(observed["control_steps"] == control_steps(trace),
                     f"generation {gen}: invalid control-step count")
            by_id[evidence_id] = observed
            new_id = observed["new_counterexample_id"]
            if verdict["safe"]:
                _require(new_id is None, f"generation {gen}: safe rollout archived as counterexample")
            else:
                cx = bank[new_id]
                source = cx["source"]
                _require(cx["generation_found"] == gen and cx["episode_data"] == trace
                         and cx["violated_constraints"] == verdict["violated_constraints"]
                         and source["policy_digest"] == raw["policy_digest"]
                         and source.get("task_config") == plan.get("candidate_task_config")
                         and source["scenario"] == archive["scenario"]
                         and source["reset_key"] == case["reset_key"] and source["disturbance"] == point
                         and source["seed"] == case["seed"] and source["phase"] == case["phase"],
                         f"generation {gen}: counterexample does not match its rollout")
            for source_id in case.get("counterexample_ids", ()):
                witness = bank[source_id]
                _require(witness["generation_found"] < gen
                         and witness["source"]["scenario"] == archive["scenario"]
                         and witness["source"]["reset_key"] == case["reset_key"]
                         and witness["source"]["disturbance"] == point,
                         f"generation {gen}: replay does not match its historical reset")
        steps = sum(case["control_steps"] for case in executed)
        _require(steps == plan["used_control_steps"], f"generation {gen}: total control steps mismatch")
        total_steps += steps
        reduction = plan["reduction"]
        _require(used["reduction"] == (reduction["evaluations"] if reduction else 0),
                 f"generation {gen}: reduction count mismatch")
        if used["reduction"]:
            origin = by_id[reduction["origin_evidence_id"]]
            _require(not origin["safe"] and origin["disturbance"] == reduction["original"],
                     f"generation {gen}: reduction origin mismatch")
            probe_ids = []
            for trial in reduction["history"]:
                evidence_id = trial["result"]["evidence_id"]
                probe = by_id[evidence_id]
                unsafe = trial["result"]["unsafe"]
                _require(probe["phase"] == "reduction" and probe["origin_evidence_id"] == origin["evidence_id"]
                         and probe["disturbance"] == trial["environment"]
                         and ((not probe["safe"]) == unsafe or
                              (unsafe is None and reduction["status"] == "missing_probe_evidence")),
                         f"generation {gen}: reduction probe mismatch")
                probe_ids.append(evidence_id)
            _require(probe_ids == [case["evidence_id"] for case in executed if case["phase"] == "reduction"],
                     f"generation {gen}: reduction history does not match executed probes")
            if reduction["status"] == "verified":
                retained = by_id[reduction["reduced_result"]["evidence_id"]]
                _require(retained["evidence_id"] in probe_ids and not retained["safe"]
                         and retained["disturbance"] == reduction["reduced"],
                         f"generation {gen}: retained reduction lacks a verified violation")
    return {
        "status": "PASS", "generations": len(report["generations"]),
        "rollouts_by_phase": dict(phases), "control_steps": total_steps,
        "report_digest": hash_dict(report), "archive_digest": hash_dict(archive),
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.scenario_dir / "demo_results.json").read_text())
    archive = json.loads((args.scenario_dir / "counterexamples.json").read_text())
    print(json.dumps(audit_gate(report, archive), indent=2))


if __name__ == "__main__":
    main()
