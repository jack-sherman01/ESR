"""Scientific counting and lineage checks; no simulator or model is used."""
from pathlib import Path

import pytest

from esrb.runtime.io import digest, fingerprint, read, save
from esrb.tools.analysis.natural_study import (
    Inputs, accepted_intervals, analyze, load_study, matching_events, predicate_changes,
)


def result(version, seed, status='PASS', qualified=True, predicates=None):
    return {'id': f'stack/chain_01/v{version:02d}/seed_{seed}', 'task': 'stack', 'chain': 1,
            'version': version, 'seed': seed, 'audit_status': status, 'task_qualified': qualified,
            'predicates': predicates or {'drop': 'VIOLATION' if status == 'VIOLATION' else 'PASS'},
            'initial_state_id': str(seed), 'model_sha256': 'model', 'control_steps': 100+version,
            'events': [], 'coverage_complete': status != 'NOT_EVALUATED'}


def test_unknown_and_unexecuted_cannot_become_new_or_resolved():
    old, new = result(1, 1), result(2, 1, 'VIOLATION')
    assert predicate_changes(old, new)['new'] == ['drop']
    old['audit_status'] = 'NOT_EVALUATED'
    assert not predicate_changes(old, new)['new']
    assert predicate_changes(old, new)['unknown'] == ['drop']
    assert not predicate_changes(new, old)['resolved']
    old['audit_status'] = 'UNKNOWN'; old['predicates']['drop'] = 'UNKNOWN'
    assert not predicate_changes(old, new)['new']


def test_known_predicate_can_be_compared_when_another_rule_is_unknown():
    old = result(1, 1, 'UNKNOWN', predicates={'drop': 'PASS', 'collision': 'UNKNOWN'})
    new = result(2, 1, 'VIOLATION', predicates={'drop': 'VIOLATION', 'collision': 'UNKNOWN'})
    assert predicate_changes(old, new)['new'] == ['drop']


def test_same_contact_aliases_do_not_duplicate_witness():
    r = result(1, 1, 'VIOLATION')
    r['events'] = [{'kind': 'robot_fixture_contact', 'groups': ['gripper0', 'table'],
                    'chunk': 'chunk_000.npz', 'contact_row': 8, 'physics_index': 12}]*2
    assert len(matching_events(r, ['robot_fixture_contact', 'contact:gripper0|table'])) == 1


def test_failed_proposal_is_not_the_next_parent_and_initial_hazard_is_not_regression():
    events = [dict(version=1, accepted=True, parent_before=None, generation_base=None),
              dict(version=2, accepted=False, parent_before={'version': 1}, generation_base={'version': 1}),
              dict(version=3, accepted=True, parent_before={'version': 1}, generation_base={'version': 1})]
    chain = {'task': 'stack', 'chain': 1, 'seeds': [1, 2, 3, 4, 5], 'events': events}
    lookup, native = {}, {}
    for e in events:
        for s in chain['seeds']:
            r = result(e['version'], s, 'VIOLATION' if s == 1 else 'PASS', qualified=e['version'] != 2 or s != 5)
            lookup[('stack', 1, e['version'], s)] = r
            native[('stack', 1, e['version'], s)] = {'termination_reason': 'failed'}
    tables = analyze([chain], lookup, native)
    assert len(tables['updates']) == 2
    assert all(u['parent_version'] == 1 for u in tables['updates'])
    assert not any(u['new_violation'] for u in tables['updates'])
    assert tables['chains'][0]['first_parent_violation_cases'] == 1
    assert tables['chains'][0]['task_accepted'] == 2
    assert tables['lineage_intervals'][0]['accepted_parent_states'] == 2


def test_predicate_regression_can_occur_under_already_unsafe_panel():
    old = result(1, 1, 'VIOLATION', predicates={'drop': 'VIOLATION', 'collision': 'PASS'})
    new = result(2, 1, 'VIOLATION', predicates={'drop': 'VIOLATION', 'collision': 'VIOLATION'})
    changes = predicate_changes(old, new)
    assert changes['new'] == ['collision']
    assert changes['persistent'] == ['drop']


def test_intervals_use_accepted_states_and_unknown_breaks_recurrence():
    codes = ['VIOLATION', 'PASS', 'VIOLATION', 'UNKNOWN', 'VIOLATION']
    lookup = {('stack', 1, i, 1): result(i, 1, code, predicates={'drop': code}) for i, code in enumerate(codes, 1)}
    events = [dict(version=i, accepted=True) for i in range(1, 6)]
    intervals = accepted_intervals(events, lookup, 'stack', 1, [1])
    assert len(intervals) == 3
    assert intervals[0]['resolution_version'] == 2
    assert intervals[1]['recurrence_after_known_pass']
    assert not intervals[2]['recurrence_after_known_pass']
    assert intervals[-1]['end_status'] == 'RIGHT_CENSORED_AT_FINAL_PARENT'


@pytest.fixture
def campaign(tmp_path):
    c, a, root = [tmp_path / name for name in ['campaign', 'audit', 'chain']]
    for p in [c, a, root]: p.mkdir()
    save(root / 'protocol.json', {'seeds': [1, 2, 3, 4, 5], 'candidate_budget': 1})
    save(root / 'artifacts.json', {})
    program = root / 'candidates/v01/program.py'; program.parent.mkdir(parents=True); program.write_text('pass\n')
    save(program.parent / 'proposal.json', {})
    sha = digest(program)
    items, cases = [], []
    for seed in range(1, 6):
        import numpy as np
        execution = tmp_path / f'execution-{seed}'; execution.mkdir()
        (execution / 'model.xml').write_text('model')
        model_sha = digest(execution / 'model.xml')
        initial = np.array([float(seed)]); np.save(execution / 'initial_state.npy', initial)
        initial_id = fingerprint({'model': model_sha, 'state': initial.tolist()})
        save(execution / 'initial_identity.json', {'seed': seed, 'initial_state_id': initial_id, 'model_sha256': model_sha})
        raw = dict(program_ok=True, terminal_native_success=True, task_qualified=True, program_sha256=sha,
                   initial_state_id=initial_id, native_task_success=True, capture={'coverage_complete': True})
        save(execution / 'result.json', raw)
        item = dict(id=f'stack/chain_01/v01/seed_{seed}', task='stack', chain=1, version=1, seed=seed,
                    execution=str(execution), result_sha256=digest(execution / 'result.json'),
                    program_sha256=sha, initial_state_id=initial_id, model_sha256=model_sha,
                    task_qualified=True, task_accepted=True, parent_version=None)
        audit_result = dict(result(1, seed), **{k: v for k, v in item.items()})
        path = a / 'episodes/stack/chain_01' / f'v01_seed_{seed}.json'; path.parent.mkdir(parents=True, exist_ok=True)
        save(path, audit_result)
        items.append(item); cases.append({'seed': seed, 'task_qualified': True})
    active = dict(version=1, program='candidates/v01/program.py', program_sha256=sha)
    event = dict(version=1, accepted=True, generation_base=None, parent_before=None, parent_after=active,
                 program='candidates/v01/program.py', program_sha256=sha, cases=cases, previous_event_sha256=None)
    event['event_sha256'] = fingerprint(event)
    save(root / 'state.json', dict(events=[event], active=active, previous_event_sha256=event['event_sha256']))
    entry = dict(task='stack', chain=1, root=str(root))
    save(c / 'manifest.json', dict(chains=[entry])); save(c / 'completion.json', {})
    save(a / 'inputs.json', dict(campaign=str(c), episodes=items, chains=[dict(entry,
        events=[event], state_sha256=digest(root/'state.json'), protocol_sha256=digest(root/'protocol.json'),
        artifact_manifest_sha256=digest(root/'artifacts.json'))]))
    save(a / 'rules.json', {})
    save(a / 'protocol.json', dict(inputs_sha256=digest(a/'inputs.json'), rules_sha256=digest(a/'rules.json')))
    return c, a, root


def test_bound_campaign_loads(campaign):
    c, a, _ = campaign
    chains, lookup, _, episodes, _, _ = load_study(c, a, Inputs())
    assert len(chains) == 1 and len(lookup) == len(episodes) == 5


def test_tampered_program_is_rejected(campaign):
    c, a, root = campaign
    (root / 'candidates/v01/program.py').write_text('changed\n')
    with pytest.raises(ValueError, match='SHA256 mismatch'):
        load_study(c, a, Inputs())


def test_actual_initial_identity_mismatch_is_rejected(campaign):
    c, a, _ = campaign
    item = read(a / 'inputs.json')['episodes'][0]
    p = Path(item['execution']) / 'initial_identity.json'
    data = read(p); data['initial_state_id'] = 'different'; save(p, data)
    with pytest.raises(ValueError, match='initial identity mismatch'):
        load_study(c, a, Inputs())


def test_failed_fifth_seed_cannot_be_accepted(campaign):
    c, a, root = campaign
    state = read(root/'state.json'); event = state['events'][0]
    event['cases'][-1]['task_qualified'] = False
    event['event_sha256'] = fingerprint({k: v for k, v in event.items() if k != 'event_sha256'})
    state['previous_event_sha256'] = event['event_sha256']; save(root/'state.json', state)
    index = read(a/'inputs.json'); index['chains'][0]['events'] = [event]
    index['chains'][0]['state_sha256'] = digest(root/'state.json'); save(a/'inputs.json', index)
    protocol = read(a/'protocol.json'); protocol['inputs_sha256'] = digest(a/'inputs.json'); save(a/'protocol.json', protocol)
    with pytest.raises(ValueError, match='not 5/5'):
        load_study(c, a, Inputs())


def test_raw_initial_array_is_checked_in_addition_to_recorded_identity(campaign):
    import numpy as np
    c, a, _ = campaign
    item = read(a / 'inputs.json')['episodes'][0]
    np.save(Path(item['execution']) / 'initial_state.npy', np.array([1234.0]))
    with pytest.raises(ValueError, match='Raw initial state'):
        load_study(c, a, Inputs())


def test_duplicate_episode_id_is_rejected(campaign):
    c, a, _ = campaign
    index = read(a/'inputs.json'); index['episodes'].append(index['episodes'][0]); save(a/'inputs.json', index)
    protocol = read(a/'protocol.json'); protocol['inputs_sha256'] = digest(a/'inputs.json'); save(a/'protocol.json', protocol)
    with pytest.raises(ValueError, match='Duplicate episode'):
        load_study(c, a, Inputs())


def test_raw_witness_checks_force_frame_geometry_and_time(tmp_path):
    import numpy as np
    from esrb.tools.analysis.natural_study_evidence import verify_witness
    p = tmp_path / 'physics'; p.mkdir()
    row = np.zeros((1, 21)); row[0, :2] = [0, 1]; row[0, 6:15] = np.eye(3).flatten(); row[0, 15] = 2.0
    np.savez_compressed(p/'chunk_000000.npz', contacts=row, contact_offsets=np.array([0, 1]), time=np.array([[0.0, 0.002]]))
    save(p/'geometry.json', [{'geom_name': 'a', 'body_name': 'body-a'}, {'geom_name': 'b', 'body_name': 'body-b'}])
    save(p/'schema.json', {'physics_timestep_s': 0.002})
    event = dict(kind='collision', chunk='chunk_000000.npz', contact_row=0, geom_ids=[0, 1],
                 geom_names=['a', 'b'], body_names=['body-a', 'body-b'], time_start_s=0.0, time_end_s=0.002,
                 point_world_m=[0.0]*3, wrench_contact_raw=[2.0, 0, 0, 0, 0, 0],
                 force_world_on_geom2_n=[2.0, 0, 0], normal_force_n=2.0)
    assert verify_witness(tmp_path, event, Inputs())['status'] == 'PASS'
    event['time_start_s'] = 0.1
    with pytest.raises(AssertionError):
        verify_witness(tmp_path, event, Inputs())
