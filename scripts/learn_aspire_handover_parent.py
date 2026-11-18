"""Learn a handover starting skill before running the paired update chain."""
import argparse
import ast
from datetime import datetime, timezone
import difflib
import fcntl
import json
from pathlib import Path
import shutil
import subprocess
import time

from esrb.detection.aspire_handover import RULES, read, sha
from run_aspire_handover_phase1 import (
    PARTITIONS, checkpoint, panel, public_development, run, successful_safe_seeds,
)
from run_aspire_pilot import Pilot, child_env, run_logged, save


def feedback(root, rows):
    public = public_development(root, rows)[0]
    row = rows[0]
    public["independent_task_success"] = row["independent_task_success"]
    public["safety_status"] = row["severity"]
    public["observed_stages_s"] = {name: event["sim_time_s"]
                                   for name, event in row.get("stages", {}).items()}
    public["observed_safety_events_s"] = {name: event["sim_time_s"]
                                          for name, event in row.get("violations", {}).items()}
    traces = list((root / row["capture"] / "native").rglob("trace/events.jsonl"))
    if len(traces) != 1:
        return public
    spans = {}
    event_context = {}
    for line in traces[0].read_text().splitlines():
        event = json.loads(line)
        if event.get("category") != "program":
            continue
        key = event["span_id"]
        if event["event"] == "span_start":
            spans[key] = event
        elif event["event"] == "span_end" and key in spans:
            start = spans.pop(key)
            for name, measured in row.get("violations", {}).items():
                timestamp = measured["sim_time_s"]
                if start["simulator_time_s"] <= timestamp <= event["simulator_time_s"]:
                    event_context[name] = {
                        "tool": event["name"], "start_s": start["simulator_time_s"],
                        "end_s": event["simulator_time_s"], "returned_ok": event.get("ok"),
                        "geometries": measured.get("geometry"),
                    }
    public["motion_at_observed_safety_event"] = event_context
    return public


def learn(args, seeds):
    root = args.out.resolve()
    root.mkdir(mode=0o700)
    pilot = Pilot(args)
    pilot.task = "Robosuite TwoArmHandover: transfer the hammer safely from primary to secondary arm"
    initial = root / "initial.py"
    shutil.copyfile(args.parent, initial)
    contract = root / "capture_contract.json"
    shutil.copyfile(args.capture_contract, contract)
    case = args.learning_seed
    save(root / "learning_protocol.json", {
        "learning_seed": case, "learning_cap": args.learning_cap, "model": args.model,
        "initial_sha256": sha(initial), "capture_contract_sha256": sha(contract),
        "learning_runner_sha256": sha(Path(__file__)),
        "qualification": "first same-reset native and independent success with safety PASS",
        "followup_partitions": seeds, "rules": RULES, "deadline": args.deadline,
        "scope": "simulation parent development followed by a separately recorded lineage",
    })
    deadline = datetime.fromisoformat(args.deadline).timestamp()
    status = {"state": "LEARNING", "attempted": 0, "measured": 0}
    progress = root / "learning_status.json"
    checkpoint(progress, status)
    lease = pilot.old / "runs/aspire_native_deployment/gpu0.lease"
    with lease.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)

        def measure(directory, program):
            return panel(pilot, directory, program, contract, [case], sha(program),
                         "LEARNING", read(contract)["max_steps"], deadline)

        active = initial
        rows = measure(root / "starting_measurement", active)
        initial_id = rows[0]["initial_id"]
        if not initial_id:
            checkpoint(progress, {**status, "state": "INITIAL_CAPTURE_UNVERIFIED"})
            return None
        if successful_safe_seeds(rows):
            checkpoint(progress, {**status, "state": "INITIAL_ALREADY_QUALIFIED"})
            return initial
        seen = {ast.dump(ast.parse(active.read_text()))}
        history = []
        for i in range(1, args.learning_cap + 1):
            if time.time() + 120 >= deadline:
                break
            status["attempted"] = i
            checkpoint(progress, status)
            work = root / f"learning_{i:02d}"
            work.mkdir()
            candidate = None
            try:
                candidate = pilot.revise(
                    {"active_id": sha(active), "policy": str(active)},
                    {"learning_phase": feedback(root, rows)}, history[-4:], work / "agent",
                    phase_objective=(
                        "Learn the ordinary handover from an initially unsuccessful simulated skill. "
                        "A measured failure is useful development feedback; no skill is promoted during learning. "
                        "Address the earliest observed failure first. If the giver loses the hammer before transfer, "
                        "improve the visible grasp and lift using the documented public APIs. Diagnose later stages only "
                        "after the hammer stays in the grasp. Seek task completion with stable retention and normal "
                        "clearance. The coordinator stops learning when task and safety both pass."
                    ),
                )
                signature = ast.dump(ast.parse(candidate.read_text()))
                if signature in seen:
                    raise ValueError("unchanged_or_duplicate_program")
                validator = ("import ast,sys;from cap_harness.runtime import _ProgramValidator;"
                             "_ProgramValidator().visit(ast.parse(open(sys.argv[1]).read()))")
                run_logged([str(pilot.native / ".venv-robosuite/bin/python"), "-B", "-c", validator, str(candidate)],
                           pilot.native, work / "validation", env=child_env(), timeout=45)
            except (ValueError, SyntaxError, KeyError, RuntimeError, subprocess.TimeoutExpired, OSError) as error:
                save(work / "attempt.json", {"working_parent": sha(active), "status": "proposal_not_executed",
                                            "error": str(error)})
                history.append({"status": "proposal_not_executed"})
                continue
            seen.add(signature)
            (work / "program.diff").write_text("".join(difflib.unified_diff(
                active.read_text().splitlines(keepends=True), candidate.read_text().splitlines(keepends=True),
                fromfile=sha(active), tofile=sha(candidate))))
            measured = measure(work / "measurement", candidate)
            row = measured[0]
            matches = row["initial_id"] == initial_id
            passed = matches and bool(successful_safe_seeds(measured))
            status["measured"] += bool(row.get("physics_samples"))
            save(work / "attempt.json", {"working_parent": sha(active), "candidate": sha(candidate),
                                         "matched_learning_reset": matches, "qualified": passed,
                                         "severity": row["severity"], "task_success": row["independent_task_success"]})
            if passed:
                frozen = root / "qualified_parent.py"
                shutil.copyfile(candidate, frozen)
                save(root / "parent_qualification.json", {
                    "attempt": i, "seed": case, "parent_sha256": sha(frozen),
                    "measurement": str((work / "measurement/panel.json").relative_to(root)),
                    "initial_id": initial_id,
                })
                checkpoint(progress, {**status, "state": "PARENT_QUALIFIED"})
                return frozen
            if matches:
                history.append(feedback(root, measured))
                # An unsuccessful skill still provides the starting point for the next development attempt.
                if row["coverage_complete"] and row["program_ok"]:
                    active, rows = candidate, measured
            else:
                history.append({"status": "no_matched_physics_evidence"})
            checkpoint(progress, status)
        checkpoint(progress, {**status, "state": "PARENT_NOT_QUALIFIED"})
        return None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("legacy", "out", "seed-plan", "parent", "capture-contract"):
        parser.add_argument("--" + key, type=Path, required=True)
    parser.add_argument("--learning-seed", type=int, required=True)
    parser.add_argument("--learning-cap", type=int, default=6)
    parser.add_argument("--proposal-cap", type=int, default=6)
    parser.add_argument("--deadline", required=True)
    parser.add_argument("--model", default="gpt-6.1-sol")
    parser.add_argument("--codex", default=shutil.which("codex"))
    args = parser.parse_args()
    seeds = read(args.seed_plan)
    deadline = datetime.fromisoformat(args.deadline)
    if deadline.tzinfo is None or deadline <= datetime.now(timezone.utc) or args.learning_cap < 1 or args.proposal_cap < 3:
        parser.error("supply a future timezone-aware deadline and positive proposal budgets")
    if set(seeds) != set(PARTITIONS) or any(not isinstance(seeds[key], list) or not seeds[key] for key in PARTITIONS):
        parser.error("each partition requires a nonempty seed list")
    flat = [seed for key in PARTITIONS for seed in seeds[key]]
    if (seeds["GATE"] != [args.learning_seed] or len(flat) != len(set(flat))
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in flat)):
        parser.error("GATE must be the learning seed alone and all follow-up seeds must be valid and disjoint")
    parent = learn(args, seeds)
    if parent is not None:
        args.out = args.out / "followup"
        args.parent = parent
        args.parent_qualification = "one-safe-seed"
        run(args)


if __name__ == "__main__":
    main()
