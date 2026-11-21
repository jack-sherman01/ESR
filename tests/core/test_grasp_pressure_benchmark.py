"""Numerical grasp updates bind task configuration, policy and saved evidence."""

from copy import deepcopy
from dataclasses import asdict, replace
import json

import numpy as np
import pytest

from esrb.agent_api.contracts import ProposalError
from esrb.agent_api.profiles import BY_INSTANCE
from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.runtime.benchmark import run_loop
from esrb.tasks.agentic import GraspPressureTask
from experiments.builders import _build_grasp_pressure


def test_structured_reset_replays_initial_force_and_reports_missing_force():
    task = GraspPressureTask()
    point = {"initial_pressure_n": 1.5}
    task.reset(seed=1, disturbance=point)
    _, _, _, first = task.step(np.array([2.0]))
    task.reset(seed=999, disturbance=point)
    _, _, _, replay = task.step(np.array([2.0]))
    assert replay == first
    assert replay["reset_readback"] == point
    assert replay["contact_forces"] == [[3.5, 0.0, 0.0]]
    oracle = scenario_oracle("grasp_pressure")
    assert oracle.protocol()["source"] == "deterministic_force_fixture"
    assert oracle.evaluate(replay, 0).status == "PASS"
    del replay["contact_forces"]
    assert oracle.evaluate(replay, 0).status == "UNKNOWN"
    with pytest.raises(ValueError, match="initial_pressure_n"):
        task.reset(disturbance={"initial_pressure_n": 3.0})


@pytest.fixture(scope="module")
def trained_grasp():
    pytest.importorskip("torch")
    return _build_grasp_pressure(2, 5, 1.0, 42, 1000)


@pytest.mark.parametrize(("severity", "accepted", "target"), [(0.0, True, 15.0), (1.0, False, 30.0)])
def test_trained_target_update_and_replay_audit(trained_grasp, severity, accepted, target):
    original, policy = trained_grasp
    agent = deepcopy(policy)
    initial_state = agent.state_dict()
    instance = replace(original, task=GraspPressureTask(), oracle=scenario_oracle("grasp_pressure"))
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=0.6), severity, 5,
        gate_planner="impact", fresh_min=2, boundary_cases=1,
    )
    for row in rows:
        receipt = row["proposal_receipt"]
        assert row["baseline_safe"] is True
        assert row["candidate_safe"] is accepted
        assert row["accepted"] is accepted
        assert receipt["candidate_components"]["policy"] == receipt["parent_components"]["policy"]
        assert receipt["proposal"]["replacements"] == {"data": {"target_pressure_n": target}}
        assert receipt["promoted"] is False
        assert receipt["active_version_after"] == receipt["base_version"]
    assert agent.state_dict() == initial_state
    assert instance.task._target == 15.0
    report = {
        "scenario": "grasp_pressure", "signal_key": "peak_force_n", "threshold": 25.0,
        "args": {"multi_constraint": True, "rollouts": 5, "seed": 42},
        "measurement_protocol": instance.oracle.protocol(), "generations": rows,
    }
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    report, archive = json.loads(json.dumps([report, archive]))
    assert audit_gate(report, archive)["status"] == "PASS"


def test_target_configuration_rejects_nonfinite_or_negative_values():
    profile = BY_INSTANCE["grasp_pressure_target_shift"]
    for target in (-1.0, float("nan"), float("inf")):
        with pytest.raises(ProposalError):
            profile.validate("data", {"target_pressure_n": target})


def test_policy_roundtrip_keeps_deterministic_actions_and_digest_is_rng_neutral():
    torch = pytest.importorskip("torch")
    from esrb.agent.rl_grasp_policy import RLGraspPolicy

    policy = RLGraspPolicy(seed=42, deterministic=True)
    replica = type(policy)(seed=17)
    replica.load_state_dict(json.loads(json.dumps(policy.state_dict())))
    observation = GraspPressureTask().reset(seed=99)
    assert np.array_equal(replica.act(observation), policy.act(observation))
    assert replica.deterministic is True
    profile = BY_INSTANCE["grasp_pressure_target_shift"]
    before = torch.random.get_rng_state().clone()
    assert profile.policy_digest(profile.snapshot(replica)) == profile.policy_digest(profile.snapshot(policy))
    assert torch.equal(before, torch.random.get_rng_state())
