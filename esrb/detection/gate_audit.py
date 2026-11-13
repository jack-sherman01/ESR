"""Verify structured gate evidence from saved benchmark artifacts."""

import argparse
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path

from esrb.agent_api.contracts import SCHEMA, identity
from esrb.agent_api.controlled import ControlledProfile
from esrb.agent_api.profiles import BY_INSTANCE as PROPOSAL_INSTANCES
from esrb.benchmark.experiment_config import LegacyPegConfig, validate_experiment_config
from esrb.detection.gate_cases import boundary_catalog, control_steps, gate_case
from esrb.oracles import ForceTorqueOracle, HeightClearanceOracle, ProximityOracle, VelocityOracle
from esrb.oracles.multiconstraint import SCENARIO_LIMITS, MultiConstraintOracle, scenario_oracle, verdict_safety
from esrb.tasks.agentic import GraspForceTask, SensorDriftTask, StochasticForceTask
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
    StochasticForceTask,
)}
_ORACLES = {
    "peak_tip_speed_ms": VelocityOracle,
    "min_obstacle_dist_m": ProximityOracle,
    "peak_transit_height_m": HeightClearanceOracle,
    "peak_force_n": ForceTorqueOracle,
}


def _audit_proposal(row, scenario, history, active_version, manifest_id):
    receipt = row["proposal_receipt"]
    proposal = receipt["proposal"]
    manifest = receipt["manifest"]
    parent = receipt["parent_components"]
    child = receipt["candidate_components"]
    feedback = receipt["public_feedback"]
    plan = row["gate_plan"]
    profile = PROPOSAL_INSTANCES.get(scenario)
    _require(profile is not None, "proposal has no registered scenario consumer")
    validators = profile.validators
    consumers = {role: item for role, (item, _) in validators.items()}
    _require(manifest["schema"] == proposal["schema"] == SCHEMA
             and list(manifest["consumers"]) == [asdict(item) for item in consumers.values()]
             and set(proposal["replacements"]) == set(consumers)
             and proposal["formats"] == {role: item.format for role, item in consumers.items()},
             "proposal consumer mismatch")
    _require(identity(manifest) == receipt["manifest_id"] == proposal["manifest_id"]
             and (manifest_id is None or manifest_id == receipt["manifest_id"]),
             "proposal manifest identity mismatch")
    _require(identity(parent) == receipt["base_version"] == proposal["base_version"]
             and (active_version is None or active_version == receipt["base_version"]),
             "proposal parent version does not follow the active version")
    reconstructed = {**parent, **{
        role: validators[role][1](value) for role, value in proposal["replacements"].items()
    }}
    _require(identity(child) == receipt["candidate_version"]
             and child == reconstructed
             and profile.policy_digest(child) == plan["candidate_policy_digest"]
             and identity(proposal) == receipt["proposal_id"], "proposal candidate identity mismatch")
    _require(identity(feedback) == receipt["feedback_id"] and feedback["task_id"] == scenario
             and feedback["generation"] == row["generation"] + 1
             and list(feedback["history"]) == history, "proposal public feedback mismatch")
    temporary = isinstance(profile, ControlledProfile) and profile.nominal_task_config is not None
    _require((child["data"] if temporary else None) == plan.get("candidate_task_config"),
             "proposal task configuration mismatch")
    promoted = row["accepted"] and not temporary
    expected_active = receipt["candidate_version"] if promoted else receipt["base_version"]
    _require(receipt["accepted"] is row["accepted"] and receipt.get("promoted", promoted) is promoted
             and receipt["active_version_after"] == expected_active,
             "proposal promotion record mismatch")
    history.append({
        "generation": row["generation"] + 1, "slot": 0,
        "task_scores": [case["raw_result"]["task_score"] for case in plan["executed"]
                        if case["phase"] != "reduction"],
    })
    return expected_active, receipt["manifest_id"]


def _audit_measurement(raw, oracle, generation, evidence_id):
    trace = raw["episode_data"]
    if isinstance(oracle, MultiConstraintOracle):
        steps = trace.get("control_step_count")
        _require(type(steps) is int and steps > 0, f"generation {generation}: invalid measured step count")
    if "fault_draw" in raw["case"]["disturbance"]:
        draw = raw["case"]["disturbance"]["fault_draw"]
        probability = trace["applied_task_config"]["fault_prob"]
        _require(trace["fault_prob"] == probability and trace["has_fault"] is (draw < probability),
                 f"generation {generation}: stochastic fault does not match the recorded draw")
    measured = oracle.evaluate(trace, generation)
    _require(raw["oracle"] == asdict(measured) and evidence_id == hash_dict(raw),
             f"generation {generation}: invalid raw evidence")
    return measured


def _audit_baseline(row, oracle, task, profile, args):
    gen = row["generation"]
    digest = profile.policy_digest(row["proposal_receipt"]["parent_components"])
    task_config = profile.nominal_task_config if isinstance(profile, ControlledProfile) else None
    measurements = row["baseline_measurements"]
    _require(len(measurements) == args["rollouts"], f"generation {gen}: baseline measurement count mismatch")
    verdicts = []
    for offset, observed in enumerate(measurements):
        raw = observed["raw_result"]
        seed = args["seed"] + gen * 1000 + offset
        case = gate_case(task, seed, "parent")
        trace = raw["episode_data"]
        _require(raw["policy_digest"] == digest and raw["case"] == case
                 and trace["reset_readback"] == case["disturbance"]
                 and trace.get("applied_task_config") == task_config,
                 f"generation {gen}: baseline measurement binding mismatch")
        verdicts.append(_audit_measurement(raw, oracle, gen, observed["evidence_id"]))
    return verdicts


def _audit_measurement_panel(row, role, verdicts):
    labels = [verdict_safety(verdict) for verdict in verdicts]
    aggregate = False if False in labels else True if labels and None not in labels else None
    _require(row[f"{role}_rollout_safe"] == labels and row[f"{role}_safe"] is aggregate,
             f"generation {row['generation']}: {role} measurement labels mismatch")


def _audit_measurement_blocker(row, baseline, candidate):
    missing = any(verdict.missing_required for verdict in baseline + candidate)
    blocked = "measurement_unknown" in row["gate_plan"]["blocking_tests"]
    _require(missing == blocked and not (missing and row["accepted"]),
             f"generation {row['generation']}: missing measurement blocker mismatch")


def audit_gate(report, archive):
    legacy_peg = archive["reset_mode"] == "explicit_xy_offset" and report["scenario"] == "scripted_peg"
    _require(archive["reset_mode"] == "structured" or legacy_peg,
             "expected structured gate artifacts")
    task = PegInsertionTask if legacy_peg else _TASKS[archive["task"]]
    signal = report.get("signal_key", "peak_tip_speed_ms")
    oracle = _ORACLES[signal](report["threshold"])
    primary_test = "force_safety" if isinstance(oracle, ForceTorqueOracle) else "dynamics_safety"
    args = report.get("args", {})
    multi = args.get("multi_constraint", False)
    settings = report.get("experiment_config")
    _require(("experiment_config" in report) == (args.get("config") is not None),
             "experiment config missing or unexpected")
    limits = {}
    if "experiment_config" in report:
        _require(isinstance(settings, dict), "invalid experiment config")
        config = validate_experiment_config(settings["settings"])
        if isinstance(config, LegacyPegConfig):
            _require(multi and report["scenario"] == "scripted_peg", "unsupported legacy configured scenario")
        _require(settings["digest"] == hash_dict(settings["settings"])
                 and report["selector_alpha"] == config.selector.alpha,
                 "experiment config digest or selector mismatch")
        limits = config.limits_for(report["scenario"])
        _require(multi or not limits, "configured oracle limits require multi-constraint measurements")
    _require(("measurement_protocol" in report) == multi, "measurement protocol missing or unexpected")
    if multi:
        scenario_task, primary_signal, default_threshold = SCENARIO_LIMITS[report["scenario"]]
        expected_threshold = limits.get(primary_signal, default_threshold)
        _require((task.name, signal, report["threshold"]) == (scenario_task, primary_signal, expected_threshold),
                 "measurement scenario or primary limit mismatch")
        oracle = scenario_oracle(report["scenario"], limits=limits)
        _require(report["measurement_protocol"] == oracle.protocol(), "measurement protocol mismatch")
        primary_test = oracle.primary_test
    boundaries = {case["id"]: case for case in boundary_catalog(task, primary_test)}
    bank = {cx["id"]: cx for cx in archive["counterexamples"]}
    _require(len(bank) == len(archive["counterexamples"]), "duplicate counterexample ID")
    phases = Counter()
    total_steps = 0
    history = []
    active_version = manifest_id = None
    has_proposals = any("proposal_receipt" in row for row in report["generations"])
    for row in report["generations"]:
        gen = row["generation"]
        plan = row["gate_plan"]
        if has_proposals:
            _require("proposal_receipt" in row, "missing proposal receipt")
            active_version, manifest_id = _audit_proposal(
                row, archive["scenario"], history, active_version, manifest_id,
            )
        baseline_verdicts = []
        candidate_verdicts = []
        if multi:
            baseline_verdicts = _audit_baseline(
                row, oracle, task, PROPOSAL_INSTANCES[archive["scenario"]], report["args"],
            )
            _audit_measurement_panel(row, "baseline", baseline_verdicts)
        executed = plan["executed"]
        count = len(executed)
        _require(count == plan["used_rollouts"] == len(plan["cases"]), f"generation {gen}: rollout count mismatch")
        if plan["blocked_reason"]:
            _require(count == 0 and not row["accepted"], f"generation {gen}: blocked gate executed or accepted")
            if multi:
                _audit_measurement_panel(row, "candidate", [])
                _audit_measurement_blocker(row, baseline_verdicts, [])
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
            evidence_id = observed["evidence_id"]
            verdict = _audit_measurement(raw, oracle, gen, evidence_id)
            safe = verdict_safety(verdict)
            candidate_verdicts.append(verdict)
            _require(observed["safe"] is safe and observed["signal_values"] == verdict.signal_values,
                     f"generation {gen}: invalid measured label")
            _require(observed["control_steps"] == control_steps(trace),
                     f"generation {gen}: invalid control-step count")
            by_id[evidence_id] = observed
            new_id = observed["new_counterexample_id"]
            if safe is not False:
                _require(new_id is None, f"generation {gen}: rollout archived without a verified violation")
            else:
                cx = bank[new_id]
                source = cx["source"]
                _require(cx["generation_found"] == gen and cx["episode_data"] == trace
                         and cx["violated_constraints"] == verdict.violated_constraints
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
        if multi:
            _audit_measurement_panel(row, "candidate", candidate_verdicts)
            _audit_measurement_blocker(row, baseline_verdicts, candidate_verdicts)
        steps = sum(case["control_steps"] for case in executed)
        _require(steps == plan["used_control_steps"], f"generation {gen}: total control steps mismatch")
        total_steps += steps
        reduction = plan["reduction"]
        _require(used["reduction"] == (reduction["evaluations"] if reduction else 0),
                 f"generation {gen}: reduction count mismatch")
        if used["reduction"]:
            origin = by_id[reduction["origin_evidence_id"]]
            _require(origin["safe"] is False and origin["disturbance"] == reduction["original"],
                     f"generation {gen}: reduction origin mismatch")
            probe_ids = []
            for trial in reduction["history"]:
                evidence_id = trial["result"]["evidence_id"]
                probe = by_id[evidence_id]
                unsafe = trial["result"]["unsafe"]
                _require(probe["phase"] == "reduction" and probe["origin_evidence_id"] == origin["evidence_id"]
                         and probe["disturbance"] == trial["environment"]
                         and ((probe["safe"] is not None and (not probe["safe"]) is unsafe)
                              or (unsafe is None and reduction["status"] == "missing_probe_evidence"
                                  and (not multi or probe["safe"] is None))),
                         f"generation {gen}: reduction probe mismatch")
                probe_ids.append(evidence_id)
            _require(probe_ids == [case["evidence_id"] for case in executed if case["phase"] == "reduction"],
                     f"generation {gen}: reduction history does not match executed probes")
            if reduction["status"] == "verified":
                retained = by_id[reduction["reduced_result"]["evidence_id"]]
                _require(retained["evidence_id"] in probe_ids and retained["safe"] is False
                         and retained["disturbance"] == reduction["reduced"],
                         f"generation {gen}: retained reduction lacks a verified violation")
    result = {
        "status": "PASS", "generations": len(report["generations"]),
        "rollouts_by_phase": dict(phases), "control_steps": total_steps,
        "report_digest": hash_dict(report), "archive_digest": hash_dict(archive),
    }
    if has_proposals:
        result["proposals"] = len(history)
        result["active_version"] = active_version
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("scenario_dir", type=Path)
    args = parser.parse_args()
    report = json.loads((args.scenario_dir / "demo_results.json").read_text())
    archive = json.loads((args.scenario_dir / "counterexamples.json").read_text())
    print(json.dumps(audit_gate(report, archive), indent=2))


if __name__ == "__main__":
    main()
