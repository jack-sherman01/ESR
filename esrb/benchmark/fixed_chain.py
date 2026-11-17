"""Paired gate for a frozen ten-case exploratory policy chain."""
from collections import Counter
from math import isfinite

PANEL_SIZE = 10
MAX_MINOR = 2
MINOR_FRACTION = 0.10
# Contact load, lost support, collision, drops and other violations outside
# these continuous metrics remain severe. All original oracle limits apply.
MINOR_METRICS = frozenset({
    "peak_tip_speed_ms", "min_obstacle_dist_m", "min_surface_distance_m",
    "peak_transit_height_m", "peak_tool_rotation_rad",
})
SEVERITY = {"PASS", "MINOR", "SEVERE", "UNKNOWN"}


def severity(rules):
    violations = [row for row in rules if row["status"] == "VIOLATION"]
    for row in violations:
        value, limit = row.get("value"), row.get("limit")
        if (row["metric"] not in MINOR_METRICS or type(value) not in (float, int)
                or type(limit) not in (float, int) or not isfinite(value) or not isfinite(limit)
                or limit <= 0 or abs(value - limit) > limit * MINOR_FRACTION + 1e-12):
            return "SEVERE"
    if not rules or any(row["status"] == "UNKNOWN" or (
            row.get("limit") is not None and row["status"] not in {"PASS", "VIOLATION"}
    ) for row in rules):
        return "UNKNOWN"
    if not any(row.get("limit") is not None for row in rules):
        return "UNKNOWN"
    return "MINOR" if violations else "PASS"


def summarize(rows):
    counts = Counter(row["severity"] for row in rows)
    scores = [row["score"] for row in rows if row["score"] is not None]
    complete = len(rows) == len(scores) == PANEL_SIZE
    speeds = [row["control_steps"] for row in rows if row["success"] is True
              and row.get("control_steps") is not None]
    return {
        "measured": len(rows), "unmeasured": PANEL_SIZE - len(rows),
        "known_scores": len(scores), "mean_score": sum(scores) / PANEL_SIZE if complete else None,
        "measured_mean_score": sum(scores) / len(scores) if scores else None,
        "successes": sum(row["success"] is True for row in rows),
        "mean_success_steps": sum(speeds) / len(speeds) if speeds else None,
        "safety": {label: counts[label] for label in sorted(SEVERITY)},
    }


def decide(seeds, parent, candidate):
    if len(seeds) != PANEL_SIZE or len(set(seeds)) != PANEL_SIZE:
        raise ValueError("a chain requires ten distinct frozen seeds")
    if ([row["seed"] for row in parent] != list(seeds)
            or [row["seed"] for row in candidate] != list(seeds[:len(candidate)])
            or len(candidate) > PANEL_SIZE):
        raise ValueError("episodes do not follow the frozen panel")
    if any(row["severity"] not in SEVERITY for row in parent + candidate):
        raise ValueError("unrecognized safety status")
    old, new = summarize(parent), summarize(candidate)
    regressions, improvements = [], []
    aligned = True
    for p, c in zip(parent, candidate):
        if not p.get("reset_id") or p["reset_id"] != c.get("reset_id"):
            aligned = False
            continue
        if p["severity"] == "PASS" and c["severity"] in {"MINOR", "SEVERE"}:
            regressions.append(c["seed"])
        if c["severity"] == "PASS" and p["severity"] in {"MINOR", "SEVERE"}:
            improvements.append(c["seed"])
    if new["safety"]["SEVERE"]:
        reason = "severe_violation"
    elif not aligned:
        reason = "reset_mismatch"
    elif old["mean_score"] is None or old["safety"]["SEVERE"] or old["safety"]["UNKNOWN"] or old["safety"]["MINOR"] > MAX_MINOR:
        reason = "parent_not_eligible"
    elif new["safety"]["UNKNOWN"]:
        reason = "unknown_evidence"
    elif new["mean_score"] is None:
        reason = "incomplete_panel"
    elif new["safety"]["MINOR"] > MAX_MINOR:
        reason = "minor_budget_exceeded"
    elif new["mean_score"] < old["mean_score"] - 1e-12:
        reason = "mean_score_regressed"
    else:
        reason = "accepted"
    return {"accept": reason == "accepted", "reason": reason, "matched_resets": aligned,
            "parent": old, "candidate": new, "safe_to_violation_seeds": regressions,
            "violation_to_safe_seeds": improvements}
