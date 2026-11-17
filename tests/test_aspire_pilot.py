"""Orchestration contracts only: doubles are not native ASPIRE evidence."""
import importlib.util
from pathlib import Path

import pytest


def load_pilot():
    script = Path(__file__).resolve().parents[1] / "scripts/run_aspire_pilot.py"
    spec = importlib.util.spec_from_file_location("aspire_pilot", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("decisions", [(False, True), (True, False)])
def test_round_two_uses_actual_active_parent(tmp_path, decisions):
    pilot = load_pilot()
    seen, proposed, evaluated = [], [], []
    obj = pilot.Pilot.__new__(pilot.Pilot)
    obj.root = tmp_path
    obj.gate_root = tmp_path / "gate"
    obj.initialize = lambda baseline: {"active_id": "parent"}
    obj.dev_panel = lambda version, path: [{"task_score": 1, "version": version}]

    def worker(ctx, dev, history, output):
        seen.append((ctx["active_id"], dev[0]["version"], list(history)))
        output.mkdir()
        program = output / "candidate.py"
        program.write_text("pass\n")
        return program

    def propose(ctx, program, output):
        version = f"child{len(proposed) + 1}"
        proposed.append((ctx["active_id"], version))
        return version

    def gate(source, arguments, output, timeout):
        version = arguments[1]
        previous = dict((child, parent) for parent, child in proposed)[version]
        accept = decisions[len(evaluated)]
        evaluated.append(version)
        return {
            "active_after": {"active_id": version if accept else previous},
            "decision": {"accept": accept, "reason": "hidden", "private_seed": 999},
        }

    obj.revise, obj.propose, obj.legacy = worker, propose, gate
    obj.public_attempt = lambda d, row: {"candidate_development": row["dev_candidate"]}
    obj.drive(Path("unused"))
    second_parent = "child1" if decisions[0] else "parent"
    assert seen[1][:2] == (second_parent, second_parent)
    assert seen[1][2][0]["candidate_development"][0]["version"] == "child1"
    assert "private_seed" not in str(seen)
    assert evaluated == ["child1", "child2"]
    assert (tmp_path / "round_1/agent/candidate.py").is_file()
    assert (tmp_path / "round_2/agent/candidate.py").is_file()


def test_worker_history_contains_diff_not_gate_inputs(tmp_path):
    import json

    pilot_cls = load_pilot().Pilot
    pilot = pilot_cls.__new__(pilot_cls)
    worker = tmp_path / "agent"
    worker.mkdir()
    parent = tmp_path / "parent.py"
    parent.write_text("attempts = 2\n")
    (tmp_path / "parent_context.json").write_text(json.dumps(
        {"policy": str(parent), "bank_sha256": "hidden_bank"}
    ))
    (worker / "proposal.json").write_text(json.dumps(
        {"program": "attempts = 3\n", "rationale": "Try a new approach."}
    ))
    public = pilot.public_attempt(tmp_path, {
        "candidate": "v2", "active_before": "v1", "active_after": "v1",
        "dev_candidate": [{"steps": 13}],
        "decision": {"reason": "secret_rule", "seed": 999},
    })
    assert "-attempts = 2" in public["program_diff"]
    assert "+attempts = 3" in public["program_diff"]
    assert public["dev"] == [{"steps": 13}] and not public["became_active"]
    assert all(secret not in str(public) for secret in ("secret_rule", "hidden_bank", "999"))
