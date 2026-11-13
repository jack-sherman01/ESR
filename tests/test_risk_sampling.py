import json
import sys

import pytest

from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_sampling import risk_cases, risk_summary, sampling_protocol
from esrb.tasks.agentic import StochasticForceTask
from esrb.tasks.contact_rich import PegInsertionTask
from experiments import run_benchmark


def test_draw_schedule_is_reproducible_and_generation_specific():
    bounds = {"x_offset_m": [-0.002, 0.002], "y_offset_m": [0.001, 0.001]}
    protocol = sampling_protocol(PegInsertionTask, 427, 20, bounds)
    first = risk_cases(PegInsertionTask, protocol, "scripted_peg", 0)
    second = risk_cases(PegInsertionTask, protocol, "scripted_peg", 1)
    assert first == risk_cases(PegInsertionTask, protocol, "scripted_peg", 0)
    assert len({case["seed"] for case in first + second}) == 40
    for case in first + second:
        assert -0.002 <= case["disturbance"]["x_offset_m"] <= 0.002
        assert case["disturbance"]["y_offset_m"] == 0.001
    with pytest.raises(ValueError, match="risk bounds"):
        sampling_protocol(PegInsertionTask, 427, 20, {"x_offset_m": [-1.0, 1.0]})


def test_replay_and_boundary_failures_do_not_enter_ucb():
    cases = [
        {"phase": "risk", "safe": True, "evidence_id": "draw"},
        {"phase": "replay", "safe": False, "evidence_id": "replay"},
        {"phase": "boundary", "safe": False, "evidence_id": "boundary"},
        {"phase": "fresh", "safe": False, "evidence_id": "fresh"},
    ]
    summary = risk_summary(cases, 0, 0.1)
    assert summary["upper_bound"] == risk_summary(cases[:1], 0, 0.1)["upper_bound"]
    assert summary["violations"] == 0
    assert summary["regression_violations"] == ["replay", "boundary"]


def test_stochastic_force_risk_gate_and_offline_audit(monkeypatch, tmp_path):
    profile = tmp_path / "risk.yaml"
    profile.write_text("""
benchmark:
  scenarios: [stochastic_force]
  num_generations: 2
  rollouts_per_generation: 24
gate:
  fresh_min: 2
risk_sampling:
  seed: 427
  samples: 20
""")
    out = tmp_path / "run"
    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--config", str(profile), "--out-dir", str(out)])
    run_benchmark.main()
    scenario = out / "stochastic_force"
    report = json.loads((scenario / "demo_results.json").read_text())
    archive = json.loads((scenario / "counterexamples.json").read_text())
    assert report["risk_sampling"]["bounds"] == {"fault_draw": [0.0, 1.0]}
    for row in report["generations"]:
        plan = row["gate_plan"]
        draws = [case for case in plan["executed"] if case["phase"] == "risk"]
        summary = plan["independent_risk"]
        assert plan["used_rollouts"] == 24
        assert summary["sample_count"] == len(draws) == 20
        assert summary["violations"] == sum(case["disturbance"]["fault_draw"] < 0.5 for case in draws)
        assert not row["accepted"]
    assert any(case["phase"] == "replay" for case in report["generations"][1]["gate_plan"]["executed"])
    assert audit_gate(report, archive)["status"] == "PASS"
    report["generations"][0]["gate_plan"]["independent_risk"]["violations"] += 1
    with pytest.raises(ValueError, match="risk summary mismatch"):
        audit_gate(report, archive)


def test_stochastic_force_replays_the_recorded_disturbance():
    task = StochasticForceTask()
    task.configure(fault_prob=0.5)
    for seed in (42, 123):
        task.reset(seed=seed, disturbance={"fault_draw": 0.25})
        _, _, _, trace = task.step([10])
        assert trace["has_fault"]
        assert trace["reset_readback"] == {"fault_draw": 0.25}
