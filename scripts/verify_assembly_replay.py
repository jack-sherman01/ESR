"""Save and replay one controlled continuous-assembly engineering witness.

Usage: python scripts/verify_assembly_replay.py --out-dir /tmp/esrb-assembly-witness
The output directory must be new. Full parent/candidate telemetry is retained;
the re-executed candidate must match every recorded sample and stage boundary.
"""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from esrb.agent.assembly_policy import AssemblyPolicy
from esrb.oracles.assembly import validate_assembly_measurements
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator
from esrb.utils.hashing import hash_dict


def execute(policy, seed, disturbance):
    task = LongHorizonAssemblyTask()
    try:
        observation, done = task.reset(seed, disturbance=disturbance), False
        while not done:
            observation, _, done, trace = task.step(policy.act(observation))
        validate_assembly_measurements(trace)
        result = task.evaluate(trace)
        return {"policy": policy.state_dict(), "trace": trace, "trace_digest": hash_dict(trace),
                "task_success": result.success, "task_score": result.task_score,
                "oracle": asdict(scenario_oracle(task.name).evaluate(trace, 0))}
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", required=True, type=Path)
    parser.add_argument("--seed", default=0, type=int)
    args = parser.parse_args()
    root = args.out_dir.absolute()
    if args.seed < 0 or root.exists() or root.is_symlink() or not root.parent.is_dir():
        parser.error("provide a nonnegative seed and a new directory under an existing parent")
    point = LongHorizonAssemblyTask.disturbance_for_seed(args.seed)
    parent = execute(AssemblyPolicy(), args.seed, point)
    child_policy = AssemblyOrientationOperator().apply(AssemblyPolicy(), args.seed).candidate_policy
    child = execute(child_policy, args.seed, point)
    replay = execute(child_policy, args.seed+100000, point)
    if not parent["task_success"] or not parent["oracle"]["safe"]:
        raise RuntimeError("the selected parent did not complete safe assembly")
    stages = {row["stage"]: row for row in child["trace"]["stage_records"]}
    if (stages.get("transport", {}).get("status") != "completed"
            or stages.get("insert", {}).get("status") != "failed"
            or child["trace"]["failure_reason"] != "insertion_blocked_orientation"
            or child["oracle"]["status"] != "VIOLATION"):
        raise RuntimeError("the selected case did not reproduce the carry-then-jam witness")
    if child["trace_digest"] != replay["trace_digest"]:
        raise RuntimeError("the structured witness did not replay identically")
    root.mkdir()
    for label, result in (("parent", parent), ("candidate", child)):
        (root/f"{label}.json").write_text(json.dumps(result, indent=2)+"\n")
    summary = {
        "status": "PASS", "scope": "controlled_engineering_only", "seed": args.seed,
        "disturbance": point, "parent_success": parent["task_success"], "parent_safe": parent["oracle"]["safe"],
        "candidate_success": child["task_success"], "candidate_failure": child["trace"]["failure_reason"],
        "candidate_violations": child["oracle"]["violated_constraints"],
        "candidate_final_depth_m": child["trace"]["insertion_depth_m"][-1],
        "candidate_stage_reports": child["oracle"]["stage_reports"],
        "parent_final_depth_m": parent["trace"]["insertion_depth_m"][-1],
        "parent_released_stable_hold_s": parent["trace"]["verification_hold_s"][-1],
        "reset_count_per_episode": 1, "physics_rollouts": 3,
        "control_steps": sum(len(result["trace"]["stages"]) for result in (parent, child, replay)),
        "replay_seed": args.seed+100000, "replay_exact_match": True,
        "candidate_trace_digest": child["trace_digest"], "replay_trace_digest": replay["trace_digest"],
        "model_calls": 0, "hardware_actions": 0,
    }
    (root/"summary.json").write_text(json.dumps(summary, indent=2)+"\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
