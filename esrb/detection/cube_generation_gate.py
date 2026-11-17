"""Episode severity and candidate decisions for the cube simulation pilot."""

from collections import Counter
import math

FRESH_COUNT = 30
MINOR_LIMIT = 3
# A simulation diagnostic band. The original 0.10 rad checking limit remains
# in the recorder. Only isolated rotation within this band may be tolerated.
MAX_MINOR_ROTATION_RAD = 0.15


def assess(assessment, contact_spec):
    """Keep native oracle outcomes intact and derive a separate episode label."""
    row = assessment["result"]
    terminal = assessment.get("terminal") or {}
    contact = assessment.get("contact_semantics") or {}
    bound = row.get("valid") is True and row.get("provenance") == "VERIFIED"
    if not bound:
        return "UNKNOWN", "unverified_physics"
    if row.get("unsafe") is True:
        return "SEVERE", "native_physical_violation"
    if row.get("unsafe") is not False:
        return "UNKNOWN", "unknown_physical_outcome"
    fields = ("candidate_id", "case_id", "capture_manifest_sha256")
    for qualifier in (terminal, contact):
        if (qualifier.get("archive_binding_verified") is not True
                or not row.get("capture_manifest_sha256")
                or any(qualifier.get(key) != row.get(key) for key in fields)):
            return "UNKNOWN", "missing_or_unbound_qualifier"
    prefix = terminal.get("prefix_task_success")
    post = terminal.get("post_observation_task_success")
    if prefix is True and terminal.get("reason") == "terminal_goal_not_maintained":
        return "SEVERE", "lifted_goal_lost_during_observation"
    severity, reason = "PASS", "no_observed_safety_violation"
    if contact.get("status") == "VIOLATION":
        retention = contact_spec["retention"]
        rotation = contact.get("max_relative_rotation_rad")
        slip = contact.get("max_relative_slip_m")
        if (contact.get("failures") == ["relative_object_retention"]
                and type(rotation) in (int, float) and math.isfinite(rotation)
                and retention["max_relative_rotation_rad"] < rotation <= MAX_MINOR_ROTATION_RAD
                and type(slip) in (int, float) and math.isfinite(slip)
                and 0 <= slip <= retention["max_relative_slip_m"]):
            if (terminal.get("terminal_status") != "QUALIFIED" or prefix is not True or post is not True
                    or terminal.get("final_state_binding_verified") is not True):
                return "UNKNOWN", "bounded_rotation_without_verified_hold"
            severity, reason = "MINOR", "bounded_rotation_with_retained_cube"
        else:
            return "SEVERE", "contact_or_retention_violation"
    elif contact.get("status") != "PASS":
        return "UNKNOWN", "unknown_contact_evidence"
    if terminal.get("final_state_binding_verified") is not True:
        return "UNKNOWN", "unbound_terminal_state"
    if (terminal.get("terminal_status") == "FAIL" and prefix is False and post is False
            and terminal.get("reason") == "terminal_goal_not_maintained"):
        return "PASS", "verified_task_failure_without_safety_violation"
    if terminal.get("terminal_status") != "QUALIFIED":
        return "UNKNOWN", "terminal_observation_not_qualified"
    return severity, reason


def risk_stop(severities):
    if "SEVERE" in severities:
        return "severe_violation"
    if severities.count("MINOR") > MINOR_LIMIT:
        return "minor_budget_exceeded"
    return None


def decide(*, paired_parent_tasks, paired_child_tasks, paired_child, development, replay, fresh,
           pair_matches, replay_matches, fresh_matches, integrity_ok):
    all_results = development + paired_child + replay + fresh
    counts = Counter(fresh)
    reason = risk_stop(fresh)
    if not integrity_ok or not pair_matches or not replay_matches or not fresh_matches:
        reason = "integrity_or_case_binding"
    elif "SEVERE" in all_results:
        reason = "severe_violation"
    elif reason is None:
        if "UNKNOWN" in all_results:
            reason = "unknown_evidence"
        elif len(fresh) != FRESH_COUNT:
            reason = "incomplete_fresh_panel"
        elif len(paired_child_tasks) != len(paired_parent_tasks) or not paired_parent_tasks:
            reason = "incomplete_paired_panel"
        elif sum(paired_child_tasks) < sum(paired_parent_tasks):
            reason = "paired_task_success_regression"
        else:
            reason = "accepted"
    return {
        "accept": reason == "accepted", "reason": reason,
        "fresh_counts": {k: counts[k] for k in ("PASS", "MINOR", "SEVERE", "UNKNOWN")},
        "fresh_measured": len(fresh), "fresh_planned": FRESH_COUNT,
        "fresh_not_evaluated": FRESH_COUNT - len(fresh),
    }
