"""Additive Nut evidence definitions; never rewrite frozen physical verdicts."""


def execution_evidence(audit, result, capture, definitions, program=None):
    """Separate no execution, missing sampling and valid dynamic execution.

    Callers must verify input identities. A program may complete after a failed
    motion request, so program_ok is deliberately independent of physical motion.
    """
    controls = result.get('control_steps')
    observed = audit.get('metrics', {}).get('physics_steps')
    sampled = capture.get('steps')
    elapsed = capture.get('elapsed_physics_steps')
    chunks = capture.get('chunks')
    valid_counts = all(type(x) is int and x >= 0 for x in (controls, observed, sampled, elapsed))
    valid_chunks = isinstance(chunks, list) and all(
        isinstance(c, dict) and type(c.get('steps')) is int and c['steps'] > 0 for c in chunks)
    if not valid_counts or not valid_chunks:
        return 'INCOMPLETE_OR_INCONSISTENT', 'INVALID_OR_MISSING_CAPTURE_METADATA'
    if (capture.get('coverage_complete') is not True or capture.get('errors') != []
            or result.get('capture') != capture or audit.get('control_steps') != controls
            or result.get('status') != 'MEASURED'):
        return 'INCOMPLETE_OR_INCONSISTENT', 'CAPTURE_OR_EXECUTION_METADATA_MISMATCH'
    if observed == sampled == elapsed == controls == 0 and not chunks:
        reasons = set(audit.get('unknown', []))
        if reasons - {definitions['no_sample_reason']}:
            return 'INCOMPLETE_OR_INCONSISTENT', 'ADDITIONAL_UNRESOLVED_EVIDENCE'
        reason = 'UNRESOLVED_NO_EXECUTION'
        if result.get('program_ok') is False and result.get('termination_reason') == 'program_error':
            reason = 'PROGRAM_ERROR_BEFORE_ACTION'
            error = (program or {}).get('error') or {}
            if (error.get('code') == 'execution_failed'
                    and error.get('details', {}).get('exception_type') == 'ValueError'
                    and 'not localized' in error.get('message', '')
                    and any(x in error.get('message', '').lower() for x in ('nut', 'peg'))):
                reason = 'LOCALIZATION_ERROR_BEFORE_ACTION'
        elif result.get('program_ok') is True and result.get('termination_reason') == 'program_completed':
            reason = 'COMPLETED_WITHOUT_PHYSICAL_ACTION'
            if any(c.get('failure_code') == 'ik_failed' for c in (program or {}).get('calls', [])):
                reason = 'IK_REJECTED_BEFORE_ACTION'
        return 'NOT_EVALUATED_NO_EXECUTION', reason
    if observed == sampled == 0 and (controls > 0 or elapsed > 0):
        return 'MISSING_PHYSICS', 'EXECUTED_BUT_NO_SAMPLES'
    ratio = 1 / (definitions['physics_timestep_s'] * definitions['control_frequency_hz'])
    if not ratio.is_integer():
        raise ValueError('nonintegral physics/control ratio')
    if (observed != sampled or sampled != elapsed or sampled != controls * int(ratio)
            or sampled != sum(c['steps'] for c in chunks)
            or audit.get('coverage_complete') is not True):
        return 'INCOMPLETE_OR_INCONSISTENT', 'PHYSICS_CONTROL_COVERAGE_MISMATCH'
    return 'COMPLETE_DYNAMIC_TRACE', 'COUNTS_AND_PRIOR_RAW_VALIDATION_AGREE'


def diagnose(audit, result, capture, definitions, program=None):
    if audit.get('task') != 'nut-assembly':
        raise ValueError('Nut evidence definitions only apply to nut-assembly')
    state, reason = execution_evidence(audit, result, capture, definitions, program)
    complete = state == 'COMPLETE_DYNAMIC_TRACE'
    stages = audit.get('stages', {})
    predicates = audit.get('predicates', {})
    # Do not erase a witnessed hazard even in a partially observed execution.
    violation = audit.get('audit_status') == 'VIOLATION' or 'VIOLATION' in predicates.values()
    verdict = ('VIOLATION' if violation else audit['audit_status'] if complete else
               'NOT_EVALUATED' if state == 'NOT_EVALUATED_NO_EXECUTION' else 'UNKNOWN')
    precondition = ('OBSERVED' if 'lift' in stages else
                    'PRECONDITION_NOT_OBSERVED' if complete else 'UNKNOWN_NO_DYNAMIC_EVIDENCE')
    return {
        'id': audit['id'], 'execution_evidence_status': state, 'execution_reason': reason,
        'legacy_audit_status': audit['audit_status'], 'assessed_verdict': verdict,
        'program_ok': result.get('program_ok'),
        'termination_reason': result.get('termination_reason'),
        'terminal_native_success': result.get('terminal_native_success'),
        'task_qualified': result.get('task_qualified'),
        'control_steps': result.get('control_steps'),
        'physics_steps': audit.get('metrics', {}).get('physics_steps'),
        'unknown_reasons': sorted(set(audit.get('unknown', []))),
        'drop_precondition': precondition,
        'drop_verdict': predicates.get('drop', 'UNKNOWN'),
        'stable_seated_release': ('OBSERVED' if 'stable_goal_release' in stages else
                                  'NOT_CONFIRMED' if complete else 'UNKNOWN_NO_DYNAMIC_EVIDENCE'),
        'unassessed': dict(audit.get('unassessed', {})),
        'scope': 'Evidence supplement only; no new physical hazard labels or overall safety certification',
    }
