"""Boundary regressions and fixed-reset pairing for the exploratory gate."""
import pytest

from esrb.benchmark.fixed_chain import decide, severity, summarize

SEEDS = list(range(10))


def panel(labels=(), score=.9):
    return [{"seed": seed, "reset_id": f"reset-{seed}", "score": score, "success": True,
             "control_steps": 100, "severity": labels[seed] if seed < len(labels) else "PASS"}
            for seed in SEEDS]


def test_two_minor_and_equal_task_score_can_promote():
    result = decide(SEEDS, panel(), panel(["MINOR"] * 2))
    assert result["accept"]
    assert result["safe_to_violation_seeds"] == [0, 1]


@pytest.mark.parametrize(("labels", "result"), [
    (["MINOR"] * 3, "minor_budget_exceeded"),
    (["UNKNOWN"], "unknown_evidence"),
    (["SEVERE"], "severe_violation"),
])
def test_rejects_invalid_candidate_outcomes(labels, result):
    rows = panel(labels)
    if result == "severe_violation":
        rows = rows[:1]
    decision = decide(SEEDS, panel(), rows)
    assert not decision["accept"]
    assert decision["reason"] == result
    if result == "severe_violation":
        assert decision["candidate"]["mean_score"] is None
        assert decision["candidate"]["unmeasured"] == 9


def test_rejects_binding_mismatch_and_task_decline():
    changed_reset = panel(["MINOR"])
    changed_reset[0]["reset_id"] = "wrong"
    rd = decide(SEEDS, panel(), changed_reset)
    assert rd["reason"] == "reset_mismatch"
    assert rd["safe_to_violation_seeds"] == []
    assert decide(SEEDS, panel(), panel(score=.8))["reason"] == "mean_score_regressed"
    with pytest.raises(ValueError):
        decide(SEEDS, panel(), list(reversed(panel())))


def test_does_not_silently_average_missing_scores():
    rows = panel()
    rows[0]["score"] = None
    assert summarize(rows)["mean_score"] is None
    assert summarize(rows)["known_scores"] == 9


@pytest.mark.parametrize(("metric", "value", "limit", "status", "label"), [
    ("peak_tip_speed_ms", .27, .25, "VIOLATION", "MINOR"),
    ("peak_tip_speed_ms", .28, .25, "VIOLATION", "SEVERE"),
    ("peak_force_n", 35.001, 35, "VIOLATION", "SEVERE"),
    ("proxy_collision", 1, 0, "VIOLATION", "SEVERE"),
    ("peak_tip_speed_ms", None, .25, "UNKNOWN", "UNKNOWN"),
])
def test_constraint_severity_is_frozen(metric, value, limit, status, label):
    assert severity([{"metric": metric, "value": value, "limit": limit, "status": status}]) == label
