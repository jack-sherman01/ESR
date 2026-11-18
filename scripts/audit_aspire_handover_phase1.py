"""Offline verification of a Three-partition ASPIRE handover run."""
import argparse
from collections import Counter
import json
from pathlib import Path

from esrb.detection.aspire_handover import audit, read, sha
from run_aspire_handover_phase1 import decision, paired, successful_safe_seeds


def verify_panel(root, rows, seeds, phase, candidate, rules):
    if [row["seed"] for row in rows] != seeds:
        raise ValueError(f"{phase} panel does not match the frozen seed plan")
    for row in rows:
        if row["phase"] != phase or row["candidate_id"] != candidate:
            raise ValueError(f"{phase} panel identity mismatch")
        folder = (root / row["capture"]).resolve()
        if folder != root and root not in folder.parents:
            raise ValueError("capture must be within the run directory")
        if row["initial_id"] is None:
            if row["severity"] != "UNKNOWN" or not row.get("error"):
                raise ValueError("missing evidence without a recorded exception")
            continue
        manifest = read(folder / "multisafety_manifest.json")
        if (manifest["source_sha256"] != candidate
                or manifest["case_id"] != f"robosuite:two_arm_handover:0:{row['seed']}"):
            raise ValueError("executed program or native case binding mismatch")
        measurement = {key: value for key, value in row.items()
                       if key not in ("capture", "phase", "seed", "elapsed_s")}
        if measurement != audit(folder, rules):
            raise ValueError("archived assessment no longer matches the raw evidence")


def verify(root):
    protocol = read(root / "protocol.json")
    state = read(root / "status.json")
    plan, rules = protocol["seeds"], protocol["rules"]
    parent = protocol["parent_source_sha256"]
    if sha(root / "parent.py") != parent or sha(root / "capture_contract.json") != protocol["capture_contract_sha256"]:
        raise ValueError("parent source or capture contract mismatch")
    active_gate = read(root / "baseline/GATE/panel.json")
    active_dev = read(root / "baseline/DEV/panel.json")
    unique = set()
    result = {"status": state["state"], "attempts": 0, "decisions": 0, "measured": 0,
              "reference_pairs": 0}
    for phase, rows in (("DEV", active_dev), ("GATE", active_gate)):
        verify_panel(root, rows, plan[phase], phase, parent, rules)
        unique.update(row["capture"] for row in rows)
    result["baseline"] = {phase: dict(Counter(row["severity"] for row in rows))
                          for phase, rows in (("DEV", active_dev), ("GATE", active_gate))}
    qualification = root / "baseline/qualification.json"
    mode = protocol.get("parent_qualification", "strict")
    if qualification.exists() and read(qualification) != {
        "rule": mode, "successful_safe_gate_seeds": successful_safe_seeds(active_gate)
    }:
        raise ValueError("parent qualification evidence mismatch")
    for attempt in sorted(root.glob("attempt_*")):
        result["attempts"] += 1
        proposal = read(attempt / "attempt.json")
        if proposal["parent"] != parent:
            raise ValueError("proposal does not target the active version")
        if proposal["status"] != "candidate":
            continue
        active_before = parent
        candidate_path = attempt / "agent/candidate.py"
        child = proposal["candidate"]
        if sha(candidate_path) != child:
            raise ValueError("candidate source mismatch")
        gate = read(attempt / "GATE/panel.json")
        dev = read(attempt / "DEV/panel.json")
        verify_panel(root, gate, plan["GATE"], "GATE", child, rules)
        verify_panel(root, dev, plan["DEV"], "DEV", child, rules)
        unique.update(row["capture"] for row in gate+dev)
        saved = read(attempt / "decision.json")
        expected = decision(active_gate, gate, parent_qualification=mode)
        if saved["reason"] == "protocol_integrity":
            raise ValueError("candidate rejected with changed protocol")
        if saved != {**expected, "parent": parent, "candidate": child,
                     "active_after": child if expected["accept"] else parent}:
            raise ValueError("gate decision mismatch")
        result["decisions"] += 1
        result["measured"] += any(row.get("physics_samples", 0) > 0 for row in gate)
        reference = read(attempt / "REFERENCE/paired_audit.json")
        if reference["parent"] != active_before or reference["candidate"] != child:
            raise ValueError("reference panel version mismatch")
        p, c = reference["parent_measurements"], reference["candidate_measurements"]
        for v, rows in ((active_before, p), (child, c)):
            if any(row["capture"] not in unique for row in rows):
                verify_panel(root, rows, plan["REFERENCE"], "REFERENCE", v, rules)
            unique.update(row["capture"] for row in rows)
        ids = {x["initial_id"] for x in p if x["initial_id"]}
        exposed = {x["initial_id"] for x in active_gate + active_dev + gate + dev if x["initial_id"]}
        disjoint = bool(ids) and len(ids) == len(p) and not bool(ids & exposed)
        matched = bool(paired(p, c))
        if reference["matched_resets"] != matched or reference["distinct_from_dev_and_gate"] != disjoint:
            raise ValueError("reference partition binding mismatch")
        result["reference_pairs"] += matched and disjoint
        if saved["accept"]:
            parent, active_gate, active_dev = child, gate, dev
    if (parent != state["active"] or result["reference_pairs"] != state["audited"]
            or result["measured"] != state["measured"]):
        raise ValueError("checkpoint does not agree with verified evidence")
    return {**result, "active": parent, "capture_records": len(unique)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    print(json.dumps(verify(args.run.resolve()), indent=2))


if __name__ == "__main__":
    main()
