"""Boundary regressions for the count-based cube pilot."""
import pytest

from esrb.detection.cube_generation_gate import assess, decide, risk_stop


def verdict(fresh, **changes):
    options = {
        "paired_parent_tasks": [True] * 4, "paired_child_tasks": [True] * 4,
        "paired_child": ["PASS"] * 4, "development": ["PASS"] * 4, "replay": [],
        "fresh": fresh, "pair_matches": True, "replay_matches": True,
        "fresh_matches": True, "integrity_ok": True,
    }
    return decide(**{**options, **changes})


def test_accepts_exactly_three_minor_and_no_step_gain_requirement():
    result = verdict(["PASS"] * 27 + ["MINOR"] * 3, replay=["MINOR"])
    assert result["accept"]
    assert result["fresh_counts"]["MINOR"] == 3


@pytest.mark.parametrize(
    ("fresh", "expected"),
    [
        (["MINOR"] * 4, "minor_budget_exceeded"),
        (["PASS", "SEVERE"], "severe_violation"),
        (["PASS"] * 29 + ["UNKNOWN"], "unknown_evidence"),
        (["PASS"] * 29, "incomplete_fresh_panel"),
    ],
)
def test_stops_and_rejections(fresh, expected):
    result = verdict(fresh)
    assert not result["accept"]
    assert result["reason"] == expected
    assert result["fresh_not_evaluated"] == 30 - len(fresh)


def test_unknown_does_not_stop_safety_sampling_on_its_own():
    assert risk_stop(["UNKNOWN"] * 4 + ["MINOR"] * 3) is None
    assert risk_stop(["MINOR"] * 4) == "minor_budget_exceeded"


def test_keeps_case_binding_and_task_success_checks():
    assert verdict(["PASS"] * 30, fresh_matches=False)["reason"] == "integrity_or_case_binding"
    assert verdict(["PASS"] * 30, replay=["SEVERE"])["reason"] == "severe_violation"
    assert verdict(["PASS"] * 30, paired_child_tasks=[True] * 3 + [False])["reason"] == "paired_task_success_regression"


def evidence(rotation=0.11, slip=0.001, terminal_status="QUALIFIED", failures=None):
    binding = {"candidate_id": "child", "case_id": "trial", "capture_manifest_sha256": "bound"}
    qualifier = {**binding, "archive_binding_verified": True}
    item = {
        "result": {**binding, "valid": True, "provenance": "VERIFIED", "unsafe": False},
        "contact_semantics": {
            **qualifier, "status": "VIOLATION",
            "failures": ["relative_object_retention"] if failures is None else failures,
            "max_relative_rotation_rad": rotation, "max_relative_slip_m": slip,
        },
        "terminal": {
            **qualifier, "final_state_binding_verified": True,
            "terminal_status": terminal_status, "prefix_task_success": True,
            "post_observation_task_success": True,
        },
    }
    return item


@pytest.mark.parametrize(
    ("item", "expected"),
    [
        (evidence(), "MINOR"),
        (evidence(0.151), "SEVERE"),
        (evidence(slip=0.006), "SEVERE"),
        (evidence(terminal_status="UNKNOWN"), "UNKNOWN"),
        (evidence(failures=["two_finger_support_lost"]), "SEVERE"),
    ],
)
def test_isolated_rotation_is_the_only_minor_case(item, expected):
    spec = {"retention": {"max_relative_rotation_rad": 0.1, "max_relative_slip_m": 0.005}}
    assert assess(item, spec)[0] == expected
