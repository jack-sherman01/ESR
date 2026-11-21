"""Common single-execution candidate admission semantics."""

def decide(result, *, active, binding_ok, expected_settle_steps):
    """Successful unsafe candidates and unknown candidates both retain the parent."""
    safety = result.get('safety') or {}
    reason = 'accepted'
    if not binding_ok:
        reason = 'binding_or_integrity_error'
    elif result.get('status') != 'MEASURED':
        reason = 'execution_evidence_unavailable'
    elif safety.get('status') == 'VIOLATION':
        reason = 'observed_safety_violation'
    elif safety.get('status') != 'PASS' or safety.get('coverage_complete') is not True:
        reason = 'unknown_safety_evidence'
    elif result.get('program_ok') is not True or result.get('native_task_success') is not True:
        reason = 'task_not_qualified'
    elif (result.get('settle_steps') != expected_settle_steps
          or result.get('settle_success_steps') != expected_settle_steps):
        reason = 'terminal_goal_not_maintained_or_missing_hold'
    accepted = reason == 'accepted'
    return {'accepted':accepted, 'decision':'PROMOTE' if accepted else 'REJECT', 'reason':reason,
            'parent_before':active, 'retained_parent':not accepted and active is not None,
            'observed_safety_regression':bool(active is not None and binding_ok and safety.get('status') == 'VIOLATION'),
            'task_success':result.get('native_task_success'), 'safety_status':safety.get('status','UNKNOWN'),
            'scope':'same-seed single-execution engineering gate; no independent retest'}
