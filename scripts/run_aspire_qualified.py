"""Qualify an ASPIRE drawer parent before proposing a normal autonomous update."""
from __future__ import annotations

import argparse
import ast
import json
import re
import shutil
from pathlib import Path
from urllib.request import urlopen

from run_aspire_pilot import DISTRIBUTION, FRESH_COUNT, TASK, Pilot, public_motion_trace, read, save, sha

# Two failures already known from the preceding engineering run are now DEV.
# None of the new qualification or paired cases are supplied to the worker.
DEV = (2843000002, 2843000001, 51, 52)
QUAL = (2834100001, 2834100002, 2834100003, 2834100004)
PAIR = (2835100001, 2835100002)
PREP_ATTEMPTS = 3
UPDATE_ATTEMPTS = 2
FORBIDDEN = re.compile(r"^(wine_rack_1_|cream_cheese_1_|plate_1_|wine_bottle_1_|akita_black_bowl_1_|flat_stove_1_)")


def passed(row):
    return (row.get("valid") is True and row.get("program_ok") is True
            and row.get("provenance") == "VERIFIED" and row.get("task_success") is True
            and row.get("unsafe") is False and type(row.get("steps")) is int and row["steps"] > 0
            and bool(row.get("capture_manifest_sha256"))
            and bool(row.get("constraints"))
            and all(r.get("status") in {"PASS", "NOT_APPLICABLE"} for r in row["constraints"]))


def state_ids(rows):
    return [(r.get("model_sha256"), r.get("initial_state_sha256")) for r in rows]


def disjoint_panel(rows, previous):
    ids = state_ids(rows)
    return (all(a and b for a, b in ids) and len(ids) == len(set(ids))
            and not set(ids).intersection(state_ids(previous)))


def diagnosis(capture, row):
    """Compact repair feedback from DEV only, including contact onset."""
    def robot(g):
        return g.startswith(("robot0_link", "gripper0_"))

    contacts = {}
    trace = capture / "telemetry.jsonl"
    if trace.is_file() and row.get("unsafe") is True:
        with trace.open() as stream:
            first = json.loads(stream.readline())
            t0 = first["time"] - first["dt"]
            for raw in [first]:
                collect_contact(raw, contacts, t0, robot)
            for line in stream:
                collect_contact(json.loads(line), contacts, t0, robot)
    return {
        "task_success": row["task_success"], "steps": row["steps"],
        "valid": row["valid"], "program_ok": row["program_ok"],
        "violations": [
            {"id": r["id"], "first": r.get("violations", [])[:1]}
            for r in row.get("constraints", []) if r["status"] in {"VIOLATION", "NOT_EVALUATED"}
        ],
        "non_target_contacts": list(contacts.values()),
    }


def collect_contact(raw, contacts, t0, robot):
    for c in raw["contacts"]:
        a, b = c["geom1"], c["geom2"]
        if robot(b):
            a, b = b, a
        force = c["wrench"][0]
        if not robot(a) or not FORBIDDEN.match(b) or force <= 0:
            continue
        key = (a, b)
        item = contacts.setdefault(key, {"robot": a, "other": b, "peak_normal_N": 0,
                                         "first_sim_elapsed_s": round(raw["time"] - t0, 4)})
        item["peak_normal_N"] = max(item["peak_normal_N"], round(force, 4))


class QualifiedPilot(Pilot):
    def __init__(self, args):
        super().__init__(args)
        self.reuse_first = args.reuse_first.resolve(strict=True) if args.reuse_first else None
        self.dev_repair_from = args.dev_repair_from.resolve(strict=True) if args.dev_repair_from else None

    def previous_dev_repair(self):
        source = self.dev_repair_from
        if read(source / "status.json").get("status") != "NO_DEV_QUALIFIED_PARENT":
            raise ValueError("additional DEV repair is allowed only before any qualification case was opened")
        old_protocol = read(source / "protocol.json")
        if (old_protocol["dev_seeds"] != list(DEV) or old_protocol["qualification_seeds"] != list(QUAL)
                or old_protocol["paired_seeds"] != list(PAIR)
                or old_protocol["fresh_distribution"] != DISTRIBUTION
                or old_protocol["fresh_count"] != FRESH_COUNT
                or old_protocol["contract_sha256"] != sha(self.old / "configs/safety_drawer_v3.json")):
            raise ValueError("the original qualification protocol does not match")
        programs, ranked, history = [], [], []
        for attempt in sorted(source.glob("parent_attempt_*")):
            if (attempt / "qualification").exists() or (attempt / "paired_parent").exists():
                raise ValueError("qualification or paired evidence has already been opened")
            panel_path = attempt / "dev/panel.json"
            if not panel_path.is_file():
                continue
            panel = read(panel_path)
            program = attempt / "agent/candidate.py"
            programs.append(ast.dump(ast.parse(program.read_text())))
            ranked.append((panel["passed"], program, panel["public"]))
            history.append({"attempt": attempt.name, "dev": panel["public"],
                            "rationale": read(attempt / "agent/proposal.json")["rationale"]})
        if not ranked:
            raise ValueError("previous run has no usable development evidence")
        _, starting_program, feedback = max(ranked, key=lambda record: record[0])
        return starting_program, feedback, history, programs

    def check_services(self):
        for port in (8114, 8115, 8116):
            try:
                with urlopen(f"http://127.0.0.1:{port}/healthz", timeout=4) as response:
                    if response.status != 200:
                        raise RuntimeError(f"ASPIRE health check returned {response.status} on port {port}")
            except OSError as error:
                raise RuntimeError(f"ASPIRE service unavailable on port {port}") from error

    def open_store(self, baseline, root):
        self.gate_root = root
        contract = self.old / "configs/safety_drawer_v3.json"
        return self.legacy(
            "import json,sys;from esrb.multisafety_runner import initialize;"
            "r,c,p,s,t,d,n=sys.argv[1:];"
            "print(json.dumps(initialize(r,c,p,s,json.loads(t),json.loads(d),int(n))))",
            [root, contract, baseline, self.skill / "skills",
             json.dumps(PAIR), json.dumps(DISTRIBUTION), FRESH_COUNT],
            root.parent / (root.name + "_init"), 300,
        )

    def measure_panel(self, context, seeds, output, role, *, stop_on_failure, repair_feedback=False):
        output.mkdir()
        rows, public = [], []
        version = context["active_id"]
        for seed in seeds:
            self.check_services()
            work = output / f"seed-{seed}"
            row = self.legacy(
                "import json,sys;from pathlib import Path;"
                "from esrb.multisafety_runner import run_case;"
                "from esrb.aspire_bridge import CandidateStore;"
                "r,v,s,o,role=sys.argv[1:];r=Path(r);"
                "print(json.dumps(run_case(r,CandidateStore(r/'store'),v,"
                "json.loads((r/'contract.json').read_text()),int(s),role,Path(o))))",
                [self.gate_root, version, seed, work / "capture", role],
                work / "dispatch", 720,
            )
            rows.append(row)
            if role == "dev":
                capture = Path(row["capture"])
                feedback = {"seed": seed, "task_success": row["task_success"],
                            "program_ok": row["program_ok"], "steps": row["steps"]}
                source = capture / "multisafety_manifest.json"
                if source.is_file():
                    native = Path(read(source)["native_run_dir"])
                    outcome = read(native / "outcome.json")
                    error = outcome.get("error")
                    if isinstance(error, dict):
                        feedback["error"] = {k: error[k] for k in ("code", "type", "message") if k in error}
                    feedback["public_motion_trace"] = public_motion_trace(native)
                if repair_feedback:
                    feedback.update(diagnosis(capture, row))
                public.append(feedback)
            if stop_on_failure and not passed(row):
                break
        save(output / "panel.json", {"role": role, "planned": len(seeds), "measured": len(rows),
                                      "passed": sum(passed(r) for r in rows),
                                      "rows": rows, "public": public})
        return rows, public

    def dev_panel(self, version, output):
        # Called by Pilot.run_round for the new candidate, with no oracle feedback.
        _, public = self.measure_panel({"active_id": version}, DEV, output, "dev", stop_on_failure=False)
        return public

    def drive_qualified(self, baseline, prior):
        self.check_services()
        previous_dev = self.previous_dev_repair() if self.dev_repair_from else None
        self.root.mkdir(mode=0o700)
        attempt_limit = 1 if previous_dev else PREP_ATTEMPTS
        legacy_contract = self.old / "configs/safety_drawer_v3.json"
        save(self.root / "protocol.json", {
            "task": TASK, "scope": "simulation engineering qualification then autonomous update",
            "initial_program": str(baseline), "initial_sha256": sha(baseline),
            "contract_sha256": sha(legacy_contract),
            "coordinator_sha256": sha(Path(__file__)),
            "worker_sha256": sha(Path(__file__).with_name("run_aspire_pilot.py")),
            "model": self.model, "max_parent_repairs": attempt_limit, "max_updates": UPDATE_ATTEMPTS,
            "dev_seeds": DEV, "qualification_seeds": QUAL, "paired_seeds": PAIR,
            "fresh_distribution": DISTRIBUTION, "fresh_count": FRESH_COUNT,
            "qualification_rule": "first all-pass DEV version; then all 4 qualification and 2 paired parent cases; stop permanently on any holdout failure",
            "reused_first_submission": (
                {"path": str(self.reuse_first), "sha256": sha(self.reuse_first / "candidate.py")}
                if self.reuse_first else None
            ),
            "additional_development_source": str(self.dev_repair_from) if previous_dev else None,
            "manual_fault_injection": False, "resampling_after_failure": False,
        })
        reference = self.open_store(baseline, self.root / "starting_policy")
        # Use actual measurements of the identical starting program from the preceding
        # pilot; the case promotion into DEV is frozen above. Never read a new holdout.
        reference_hash = sha(baseline)
        seen = {ast.dump(ast.parse(baseline.read_text()))}
        initial_report = read(prior / "round_1/round.json")
        g = prior / "gate/evaluations" / initial_report["candidate"]
        evidence = []
        for index, seed in enumerate(reversed(DEV[:2])):
            capture = g / f"child-{index:03d}"
            meta = read(capture / "multisafety_manifest.json")
            row = read(capture / "safety_v3.json")
            native = Path(meta["native_run_dir"])
            assert meta["case_id"].endswith(f":{seed}")
            assert meta["source_sha256"] == sha(native / "source/program.py") == reference_hash
            assert sha(prior / "gate/contract.json") == sha(self.gate_root / "contract.json")
            outcome = read(native / "outcome.json")
            row.update(task_success=outcome["task_success"], program_ok=outcome["program_ok"],
                       steps=outcome["steps_executed"])
            evidence.append({"seed": seed, **diagnosis(capture, row), "motion": public_motion_trace(native)})
        save(self.root / "starting_public_diagnostics.json", evidence)
        feedback, history = evidence, []
        if previous_dev:
            starting_program, feedback, history, programs = previous_dev
            seen.update(programs)
            reference = self.open_store(starting_program, self.root / "development_reference")
            save(self.root / "additional_dev_start.json", {"reference": str(starting_program),
                 "version": reference["active_id"], "sha256": sha(starting_program)})
        for n in range(1, attempt_limit + 1):
            prep = self.root / f"parent_attempt_{n}"
            prep.mkdir()
            if n == 1 and self.reuse_first:
                agent = prep / "agent"
                agent.mkdir()
                for name in ("candidate.py", "proposal.json", "prompt.txt", "events.jsonl",
                             "invocation.json", "status.json", "stderr.log", "response_schema.json"):
                    shutil.copyfile(self.reuse_first / name, agent / name)
                candidate = agent / "candidate.py"
                if candidate.read_text() != read(agent / "proposal.json")["program"]:
                    raise ValueError("archived candidate differs from the original model response")
            else:
                candidate = self.revise(reference, feedback, history, prep / "agent", phase_objective=(
                    "PARENT PREPARATION (not a research update): correct the unsafe approach/pull "
                    "using DEV contact onset and motion timing. Complete the middle-drawer task "
                    "without touching non-target objects, knocking them over, or violating other "
                    "existing constraints. Pick a practical different motion hypothesis, including "
                    "a measured detour, grasp approach or smaller tracking segments if justified. "
                    "A no-motion or premature stop is not a solution. The previous trajectory can "
                    "contact a bottle or its rack in approach or pull. You may use other documented "
                    "public geometric/motion primitives in the existing host allowlist; do not "
                    "assume unconfigured planners. The simulator has SAM3, GraspNet and PyRoKi. "
                    "The native AST validator forbids standalone underscore, leading-underscore "
                    "names, and all imports except plain import math. "
                    + (
                        "Focus on one viable change near the grasp/late pull. Preserve the "
                        "known-working initial approach from the supplied current program. "
                        "Avoid the high transit that failed on all scenarios; avoid insisting "
                        "on strict 1 cm tracking at the 16 cm stage, where motion also stalled. "
                        "An offline check of DEV 51 showed no direct robot contact with the "
                        "bottle and initial bottle/cabinet contact at 17.57 s. The gripper "
                        "had loaded the cabinet/handle before that, so account for indirect "
                        "loads and grasp tracking, not only direct bottle clearance. "
                        "Use bounded public pose or end-effector feedback to maintain the "
                        "outward trajectory and avoid pushing or rotating the cabinet. "
                        "Do not shorten the run to conceal post-motion instability."
                        if previous_dev else ""
                    )
                ))
            try:
                syntax = ast.dump(ast.parse(candidate.read_text()))
            except SyntaxError as exc:
                save(prep / "result.json", {"status": "SYNTAX_ERROR", "message": exc.msg})
                history.append({"attempt": n, "rationale": read(prep / "agent/proposal.json")["rationale"],
                                "status": "syntax_error"})
                continue
            if syntax in seen:
                save(prep / "result.json", {"status": "DUPLICATE_AST"})
                history.append({"attempt": n, "status": "duplicate_program"})
                continue
            seen.add(syntax)
            previous = reference
            reference = self.open_store(candidate, prep / "store")
            rows, feedback = self.measure_panel(reference, DEV, prep / "dev", "dev",
                                                stop_on_failure=False, repair_feedback=True)
            save(prep / "result.json", {"version": reference["active_id"], "derived_from": previous["active_id"],
                                       "dev_pass": sum(passed(r) for r in rows), "dev_count": len(rows)})
            if len(rows) == len(DEV) and all(passed(r) for r in rows):
                qrows, _ = self.measure_panel(reference, QUAL, prep / "qualification", "qualification",
                                               stop_on_failure=True)
                if len(qrows) == len(QUAL) and disjoint_panel(qrows, rows) and all(passed(r) for r in qrows):
                    parent_rows, _ = self.measure_panel(reference, PAIR, prep / "paired_parent", "parent_check",
                                                       stop_on_failure=True)
                    if (len(parent_rows) == len(PAIR) and disjoint_panel(parent_rows, rows + qrows)
                            and all(passed(r) for r in parent_rows)):
                        save(self.root / "qualified.json", {"parent": reference["active_id"],
                             "source": str(candidate), "accepted_for": "bounded simulation pilot",
                             "dev": len(rows), "qualification": len(qrows), "paired_parent": len(parent_rows),
                             "gate_root": str(self.gate_root)})
                        self.update_from(reference)
                        return
                save(self.root / "status.json", {"status": "QUALIFICATION_FAILED", "updates": 0,
                                                 "attempt": n})
                return
            history.append({"attempt": n, "rationale": read(prep / "agent/proposal.json")["rationale"],
                            "dev_pass_count": sum(passed(r) for r in rows)})
        save(self.root / "status.json", {"status": "NO_DEV_QUALIFIED_PARENT", "updates": 0})

    def update_from(self, reference):
        work = self.root / "updates"
        work.mkdir()
        feedback = self.dev_panel(reference["active_id"], work / "dev_parent")
        history = []
        for n in range(1, UPDATE_ATTEMPTS + 1):
            directory = work / f"round_{n}"
            reference, feedback, status = self.run_round(reference, feedback, history, directory)
            history.append(self.public_attempt(directory, status))
            if status.get("decision", {}).get("accept") is True:
                save(self.root / "status.json", {"status": "PROMOTED", "attempt": n,
                      "active_after": reference["active_id"], "decision": status["decision"]})
                return
        save(self.root / "status.json", {"status": "NO_PROMOTION", "attempts": UPDATE_ATTEMPTS,
                                         "active_after": reference["active_id"]})


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--legacy", type=Path, required=True)
    p.add_argument("--prior", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--model", required=True)
    p.add_argument("--reuse-first", type=Path, help="Archive and replay a submitted first policy after infrastructure failure")
    p.add_argument("--codex", default=shutil.which("codex"))
    p.add_argument("--dev-repair-from", type=Path,
                   help="One additional targeted repair using only the previous DEV evidence")
    args = p.parse_args()
    if args.reuse_first and args.dev_repair_from:
        p.error("choose either reproducing an interrupted proposal or targeted DEV repair")
    baseline = args.prior / "round_1/agent/candidate.py"
    QualifiedPilot(args).drive_qualified(baseline.resolve(strict=True), args.prior.resolve(strict=True))


if __name__ == "__main__":
    main()
