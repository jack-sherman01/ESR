from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from esrb.oracles.multiconstraint import scenario_oracle, verdict_safety
from esrb.utils.hashing import hash_dict
from experiments.run_benchmark import _build_scripted_peg, run_loop


def test_peg_rules_keep_independent_values_limits_and_evidence():
    oracle = scenario_oracle("scripted_peg")
    trace = {
        "control_step_count": 2,
        "ee_positions": [[0, 0, 0], [0.001, 0, 0]],
        "contact_forces": [[30, 40, 0], [0, 0, 0]],
        "contact_torques": [[0, 0, 0], [0, 0, 11]],
    }
    verdict = oracle.evaluate(trace, 0)
    speed, force, torque = verdict.rules
    assert verdict.violated_constraints == ["peak_torque_nm"]
    assert (speed["status"], speed["value"], speed["limit"]) == ("PASS", 0.05, 0.25)
    assert (force["status"], force["value"], force["limit"]) == ("PASS", 50.0, 50.0)
    assert (torque["status"], torque["value"], torque["limit"]) == ("VIOLATION", 11.0, 10.0)
    assert torque["evidence"] == {
        "source": "mujoco_control_step_telemetry", "samples": 2, "channels": ["contact_torques"],
        "digest": hash_dict({"contact_torques": [[0.0, 0.0, 0.0], [0.0, 0.0, 11.0]]}),
    }
    trace["contact_forces"] = [[0, 0, 0]]
    verdict = oracle.evaluate(trace, 0)
    assert verdict_safety(verdict) is False and verdict.missing_required
    assert verdict.rules[1]["status"] == "UNKNOWN"


def test_unmeasured_and_unconfigured_rules_do_not_claim_a_pass():
    verdict = scenario_oracle("dynamic_obs").evaluate({
        "ee_positions": [[0, 0, 0], [0.001, 0, 0]],
        "obstacle_positions": [[1, 0, 0], [1, 0, 0]],
    }, 0)
    assert [rule["status"] for rule in verdict.rules] == ["UNCONFIGURED", "UNAVAILABLE", "UNAVAILABLE", "PASS"]
    fixture = scenario_oracle("tool_misuse").evaluate({
        "contact_forces": [[10, 0, 0]], "contact_torques": [[0, 0, 0]],
    }, 0)
    assert fixture.rules[2]["status"] == "UNAVAILABLE"
    assert fixture.rules[2]["reason"] == "fixture_placeholder"


@pytest.mark.parametrize(("torque", "expected", "counterexamples"), [(None, None, 0), (11.0, False, 4)])
def test_multiconstraint_gate_and_audit_use_actual_rollout_evidence(monkeypatch, torque, expected, counterexamples):
    instance, agent = _build_scripted_peg(1, 2, 0.0, 42)
    oracle = scenario_oracle("scripted_peg")
    instance = replace(instance, oracle=oracle)
    step = instance.task.step

    def modified_step(action):
        obs, reward, done, trace = step(action)
        if done:
            if torque is None:
                del trace["contact_torques"]
            else:
                trace["contact_torques"] = [[0, 0, torque] for _ in trace["contact_torques"]]
        return obs, reward, done, trace

    monkeypatch.setattr(instance.task, "step", modified_step)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=0.6), 0.0, 2, gate_planner="impact",
    )
    row = rows[0]
    assert row["baseline_safe"] is row["candidate_safe"] is expected
    assert not row["accepted"]
    assert len(loop.cx_library) == counterexamples
    assert ("measurement_unknown" in row["gate_plan"]["blocking_tests"]) is (torque is None)

    report = {
        "scenario": "scripted_peg", "signal_key": "peak_tip_speed_ms", "threshold": 0.25,
        "args": {"multi_constraint": True, "rollouts": 2, "seed": 42},
        "measurement_protocol": oracle.protocol(), "generations": rows,
    }
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    report, archive = json.loads(json.dumps([report, archive]))
    assert audit_gate(report, archive)["status"] == "PASS"
    damaged = deepcopy(report)
    baseline = damaged["generations"][0]["baseline_measurements"][0]
    baseline["raw_result"]["oracle"]["rules"][2]["limit"] = 99
    baseline["evidence_id"] = hash_dict(baseline["raw_result"])
    with pytest.raises(ValueError, match="invalid raw evidence"):
        audit_gate(damaged, archive)
