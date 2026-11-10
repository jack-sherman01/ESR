from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from esrb.agent.evolution_loop import EvolutionLoop
from esrb.agent.mlp_policy import MLPInsertionPolicy
from esrb.agent_api.contracts import ProposalError, identity
from esrb.agent_api.policy_sources import JsonProposer
from esrb.agent_api.profiles import BY_INSTANCE, DYNAMIC_OBS, SCRIPTED_PEG
from esrb.detection.gate_audit import audit_gate
from esrb.detection.risk_selection import RiskSensitiveSelector
from experiments.run_benchmark import (
    _build_dynamic_obs, _build_evaluator_drift, _build_pick_place, _build_scripted_peg,
    _build_sensor_drift, _build_tool_misuse, _build_tool_tighten, run_loop,
)


FIXTURE = Path(__file__).resolve().parents[1] / "configs/experiment/scripted_peg_proposals.json"


def test_json_candidate_improves_task_and_only_safe_update_is_promoted():
    proposer, agent = SCRIPTED_PEG.load_proposal_file(FIXTURE, 2)
    instance, _ = _build_scripted_peg(2, 20, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=0.09), 0.5, 20,
        gate_planner="impact", fresh_min=2, boundary_cases=2, reduction_budget=3,
        proposer=proposer,
    )
    first, second = rows
    initial, later = [row["proposal_receipt"] for row in rows]
    assert first["baseline_safe"] and first["baseline_score"] == 0
    assert first["accepted"] and first["candidate_safe"]
    assert all(case["raw_result"]["task_score"] == 1 for case in first["gate_plan"]["executed"])
    assert later["base_version"] == initial["active_version_after"] == initial["candidate_version"]
    assert list(later["public_feedback"]["history"]) == [
        {"generation": 1, "slot": 0, "task_scores": [1.0] * 20},
    ]
    assert not second["accepted"] and not second["candidate_safe"]
    assert later["active_version_after"] == later["base_version"] != later["candidate_version"]
    assert loop.agent.max_speed == 0.004

    report = {"scenario": "scripted_peg", "threshold": 0.25, "generations": rows}
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    audit = audit_gate(report, archive)
    assert audit["proposals"] == 2
    assert audit["active_version"] == later["active_version_after"]
    corrupted = deepcopy(report)
    corrupted["generations"][1]["proposal_receipt"]["active_version_after"] = later["candidate_version"]
    with pytest.raises(ValueError, match="promotion record mismatch"):
        audit_gate(corrupted, archive)


@pytest.mark.parametrize(("change", "message"), [
    ({"base_version": "old"}, "stale proposal parent version"),
    ({"manifest_id": "foreign"}, "foreign proposal manifest"),
    ({"replacements": {"tools": {}}, "formats": {"tools": "json"}}, "unauthorized proposal role"),
    ({"formats": {"policy": "bad"}}, "unsupported proposal format"),
    ({"replacements": {"policy": {"max_speed": 0.004}}}, "policy fields must match"),
])
def test_invalid_proposals_stop_before_simulation(monkeypatch, change, message):
    proposer, agent = SCRIPTED_PEG.load_proposal_file(FIXTURE, 2)
    proposer = JsonProposer((replace(proposer.proposals[0], **change),), proposer.manifest)
    instance, _ = _build_scripted_peg(1, 20, 0.5, 42)

    def unexpected_rollout(*args, **kwargs):
        pytest.fail("invalid proposal reached a rollout")

    monkeypatch.setattr(EvolutionLoop, "_run_rollout", unexpected_rollout)
    with pytest.raises(ProposalError, match=message):
        run_loop(instance, agent, RiskSensitiveSelector(), 0.5, 20,
                 gate_planner="impact", proposer=proposer)


def test_host_rejects_unregistered_consumer_even_if_declared_by_proposer():
    proposer, agent = SCRIPTED_PEG.load_proposal_file(FIXTURE, 2)
    unsupported = replace(proposer.manifest.consumers[0], format="unknown-format")

    def unexpected_proposal(*args):
        pytest.fail("the host requested a proposal from an incompatible manifest")

    forged = SimpleNamespace(
        manifest=replace(proposer.manifest, consumers=(unsupported,)), propose=unexpected_proposal,
    )
    instance, _ = _build_scripted_peg(1, 20, 0.5, 42)
    with pytest.raises(ProposalError, match="unsupported host consumer format"):
        run_loop(instance, agent, RiskSensitiveSelector(), 0.5, 20,
                 gate_planner="impact", proposer=forged)


def test_dynamic_obstacle_json_proposals_share_the_gate_and_track_promotion():
    source, agent = DYNAMIC_OBS.load_proposal_file(FIXTURE.with_name("dynamic_obs_proposals.json"), 2)
    instance, _ = _build_dynamic_obs(2, 20, 0.5, 42)
    loop, rows = run_loop(
        instance, agent, RiskSensitiveSelector(epsilon=0.09), 0.5, 20,
        gate_planner="impact", fresh_min=2, boundary_cases=2, reduction_budget=3, proposer=source,
    )
    first, second = rows
    assert first["baseline_safe"] and first["candidate_safe"] and first["accepted"]
    scores = [case["raw_result"]["task_score"] for case in first["gate_plan"]["executed"]]
    assert sum(scores) / len(scores) > first["baseline_score"]
    assert not second["candidate_safe"] and not second["accepted"]
    initial, later = [row["proposal_receipt"] for row in rows]
    assert later["base_version"] == initial["candidate_version"] == later["active_version_after"]
    assert later["candidate_version"] != later["active_version_after"]
    assert loop.agent.avoidance_radius == 0.06
    report = {
        "scenario": "dynamic_obs", "signal_key": "min_obstacle_dist_m", "threshold": 0.04,
        "generations": rows,
    }
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    assert audit_gate(report, archive)["active_version"] == later["active_version_after"]


def test_dynamic_obstacle_wraps_existing_operator_and_rejects_foreign_consumer(monkeypatch):
    instance, agent = _build_dynamic_obs(1, 2, 0.5, 42)
    _, (row,) = run_loop(instance, agent, RiskSensitiveSelector(), 0.5, 2, gate_planner="impact")
    receipt = row["proposal_receipt"]
    assert receipt["manifest"]["adapter_id"] == "distribution_shift"
    assert receipt["candidate_components"]["policy"]["avoidance_radius"] == 0.03
    assert not row["candidate_safe"] and not row["accepted"]

    proposer, _ = SCRIPTED_PEG.load_proposal_file(FIXTURE, 2)

    def unexpected_rollout(*args, **kwargs):
        pytest.fail("foreign policy format reached a rollout")

    monkeypatch.setattr(EvolutionLoop, "_run_rollout", unexpected_rollout)
    with pytest.raises(ProposalError, match="unsupported host consumer format"):
        run_loop(instance, agent, RiskSensitiveSelector(), 0.5, 2,
                 gate_planner="impact", proposer=proposer)


@pytest.mark.parametrize(("builder", "role", "signal", "threshold"), [
    (_build_pick_place, "policy", "peak_transit_height_m", 0.10),
    (_build_evaluator_drift, "evaluator", "peak_transit_height_m", 0.10),
    (_build_tool_tighten, "reward", "peak_force_n", 35.0),
    (_build_tool_misuse, "tools", "peak_force_n", 20.0),
    (_build_sensor_drift, "data", "peak_force_n", 20.0),
])
def test_controlled_scenario_proposals_are_auditable(builder, role, signal, threshold):
    instance, agent = builder(1, 2, 0.5, 42)
    loop, (row,) = run_loop(instance, agent, RiskSensitiveSelector(), 0.5, 2, gate_planner="impact")
    receipt = row["proposal_receipt"]
    assert set(receipt["proposal"]["replacements"]) == {role}
    assert not receipt["promoted"]
    assert receipt["active_version_after"] == receipt["base_version"]
    report = {"scenario": instance.name, "threshold": threshold, "signal_key": signal, "generations": [row]}
    archive = {
        "scenario": instance.name, "task": instance.task.name, "reset_mode": "structured",
        "counterexamples": [asdict(cx) for cx in loop.cx_library.all()],
    }
    assert audit_gate(report, archive)["proposals"] == 1


def test_fixed_sensor_drift_uses_proposal_and_keeps_nominal_configuration():
    instance, agent = _build_sensor_drift(2, 1, 0.5, 42)
    _, rows = run_loop(instance, agent, RiskSensitiveSelector(epsilon=1.0), 0.5, 1)
    for row in rows:
        receipt = row["proposal_receipt"]
        assert row["baseline_sigs"]["peak_force_n"] == 10.0
        assert row["candidate_sigs"]["peak_force_n"] == 30.0
        assert receipt["proposal"]["replacements"]["data"] == {"true_force": 30.0, "sensor_bias": -20.0}
        assert receipt["accepted"] and not receipt["promoted"]
        assert receipt["active_version_after"] == receipt["base_version"]
    assert rows[1]["proposal_receipt"]["base_version"] == rows[0]["proposal_receipt"]["active_version_after"]


def test_mlp_version_tracks_changes_to_same_length_memory():
    pytest.importorskip("torch")
    policy = MLPInsertionPolicy()
    policy.replay_buffer.append({"obs": [0.0] * 15, "action": [0.0] * 3, "reward": 0.0})
    profile = BY_INSTANCE["mlp_peg_memory_contamination"]
    first = profile.snapshot(policy)
    policy.replay_buffer[0]["reward"] = 1.0
    second = profile.snapshot(policy)
    assert first["policy"] == second["policy"]
    assert first["memory"]["replay_buffer"] != second["memory"]["replay_buffer"]
    assert identity(first) != identity(second)
