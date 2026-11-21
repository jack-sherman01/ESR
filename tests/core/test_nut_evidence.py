"""No-execution, missing-physics and task/safety counterexamples."""
from copy import deepcopy
from pathlib import Path
import pytest
from esrb.runtime.io import read
from esrb.detection.nut_evidence import diagnose

D = read(Path(__file__).resolve().parents[2] / 'configs/evaluation/nut_evidence_v1.json')


def sample(steps=0, controls=0):
    capture = dict(steps=steps, elapsed_physics_steps=steps, coverage_complete=True,
                   errors=[], chunks=[dict(steps=steps)] if steps else [])
    result = dict(status='MEASURED', control_steps=controls, capture=deepcopy(capture),
                  program_ok=True, termination_reason='program_completed',
                  terminal_native_success=False, task_qualified=False)
    audit = dict(id='nut/1', task='nut-assembly', control_steps=controls,
                 metrics=dict(physics_steps=steps), coverage_complete=bool(steps),
                 audit_status='PASS' if steps else 'UNKNOWN',
                 unknown=[] if steps else ['no physics samples'] * 2,
                 stages={}, predicates={'drop': 'PASS' if steps else 'UNKNOWN'},
                 unassessed={'qualified_jam': 'UNKNOWN'})
    return audit, result, capture


def test_empty_but_complete_capture_is_no_execution_never_safe_pass():
    a, r, c = sample()
    out = diagnose(a, r, c, D)
    assert out['execution_evidence_status'] == 'NOT_EVALUATED_NO_EXECUTION'
    assert out['assessed_verdict'] == 'NOT_EVALUATED'
    assert out['legacy_audit_status'] == 'UNKNOWN'
    assert out['unknown_reasons'] == ['no physics samples']


def test_ignored_ik_error_can_coexist_with_completed_program():
    a, r, c = sample()
    out = diagnose(a, r, c, D, {'calls': [{'ok': False, 'failure_code': 'ik_failed'}]})
    assert out['execution_reason'] == 'IK_REJECTED_BEFORE_ACTION'
    assert out['program_ok'] is True


def test_verified_localization_exception_is_separate_from_other_program_errors():
    a, r, c = sample()
    r.update(program_ok=False, termination_reason='program_error')
    program = {'error': {'code': 'execution_failed', 'details': {'exception_type': 'ValueError'},
                         'message': 'Nut was not localized'}}
    assert diagnose(a, r, c, D, program)['execution_reason'] == 'LOCALIZATION_ERROR_BEFORE_ACTION'
    assert diagnose(a, r, c, D)['execution_reason'] == 'PROGRAM_ERROR_BEFORE_ACTION'


def test_controls_without_physics_are_missing_data_not_no_execution():
    a, r, c = sample(controls=1)
    out = diagnose(a, r, c, D)
    assert out['execution_evidence_status'] == 'MISSING_PHYSICS'
    assert out['assessed_verdict'] == 'UNKNOWN'


@pytest.mark.parametrize('change', ['errors', 'chunks', 'coverage', 'elapsed', 'missing_count', 'extra_unknown'])
def test_bad_capture_cannot_be_excused_as_clean_zero_execution(change):
    a, r, c = sample()
    if change == 'errors': c['errors'] = ['disk full']
    elif change == 'chunks': c['chunks'] = [{'steps': 1}]
    elif change == 'coverage': c['coverage_complete'] = False
    elif change == 'elapsed': c['elapsed_physics_steps'] = 25
    elif change == 'missing_count': del c['steps']
    else: a['unknown'].append('identity mismatch')
    r['capture'] = deepcopy(c)
    assert diagnose(a, r, c, D)['assessed_verdict'] == 'UNKNOWN'


def test_task_failure_can_have_no_assessed_physical_violation():
    a, r, c = sample(25, 1)
    out = diagnose(a, r, c, D)
    assert out['assessed_verdict'] == 'PASS' and out['task_qualified'] is False
    assert out['drop_precondition'] == 'PRECONDITION_NOT_OBSERVED'
    assert out['unassessed']['qualified_jam'] == 'UNKNOWN'


def test_native_success_does_not_manufacture_stable_release_or_remove_collision():
    a, r, c = sample(25, 1)
    r.update(task_qualified=True, terminal_native_success=True)
    a.update(audit_status='VIOLATION', stages={'lift': .02})
    a['predicates']['wrong_peg_contact'] = 'VIOLATION'
    out = diagnose(a, r, c, D)
    assert out['assessed_verdict'] == 'VIOLATION'
    assert out['stable_seated_release'] == 'NOT_CONFIRMED'


def test_incomplete_data_does_not_erase_witnessed_violation_or_mutate_input():
    a, r, c = sample(25, 1)
    a['predicates']['wrong_peg_contact'] = 'VIOLATION'
    c['errors'] = ['lost final chunk']
    before = deepcopy((a, r, c))
    assert diagnose(a, r, c, D)['assessed_verdict'] == 'VIOLATION'
    assert (a, r, c) == before


def test_phase_evidence_is_not_created_without_a_marker():
    a, r, c = sample(25, 1)
    a['stages']['stable_goal_release'] = .02
    assert diagnose(a, r, c, D)['stable_seated_release'] == 'OBSERVED'


def test_unresolved_trace_end_drop_is_not_relabelled_safe():
    a, r, c = sample(25, 1)
    a.update(audit_status='UNKNOWN', unknown=['unconfirmed airborne grasp loss at trace end'])
    a['predicates']['drop'] = 'UNKNOWN'
    out = diagnose(a, r, c, D)
    assert out['execution_evidence_status'] == 'COMPLETE_DYNAMIC_TRACE'
    assert out['assessed_verdict'] == 'UNKNOWN'


def test_wrong_scene_is_rejected():
    a, r, c = sample()
    a['task'] = 'stack'
    with pytest.raises(ValueError, match='only apply'):
        diagnose(a, r, c, D)
