"""Revised offline decisions must retain original evidence and lineage."""
import json
import sys
from pathlib import Path

import pytest

from experiments.studies.measurement_validation import read,save
from experiments.studies.measurement_reprocessing import reuse_tree,sha
from experiments.studies.measurement_reaudit import worker,recompute_gates,refresh_validation_witnesses
from esrb.measurement.spec import event_profile
from tests.core.test_measurement_validation import physical_trace


def test_offline_worker_replaces_linked_result_without_touching_original(physical_trace,tmp_path,monkeypatch):
    from esrb.measurement import episodes
    monkeypatch.setattr(episodes,'run',lambda *a,**kw:pytest.fail('offline audit started a physical rollout'))
    origin=tmp_path/'original';relative='validation/constrained_push/baseline-seed0000'
    (origin/relative).parent.mkdir(parents=True);reuse_tree(physical_trace,origin/relative)
    save(origin/relative/'result.json',dict(scene='constrained_push',case='baseline',status='RECORDED',seed=0,
        file=relative,wall_s=12.5,measurement=dict(primary_status='UNKNOWN',missing=['old rule']),task_score=1.))
    receipts={str(p.relative_to(origin)):sha(p) for p in origin.rglob('*') if p.is_file()}
    root=tmp_path/'new';(root/relative).parent.mkdir(parents=True);reuse_tree(origin/relative,root/relative)
    save(root/'protocol.json',dict(event_profile=event_profile(),offline_reaudit=dict(origin=str(origin))))
    worker(root,relative)
    result=read(root/relative/'result.json')
    assert result['measurement']['primary_status']=='PASS'
    assert result['reference']['status']=='PASS'
    assert result['wall_s']==12.5
    assert result['offline_reaudit']['physical_rollouts_started']==0
    assert result['offline_reaudit']['previous_primary_status']=='UNKNOWN'
    assert receipts=={str(p.relative_to(origin)):sha(p) for p in origin.rglob('*') if p.is_file()}
    assert (root/relative/'raw/manifest.json').stat().st_ino==(origin/relative/'raw/manifest.json').stat().st_ino
    assert (root/relative/'result.json').stat().st_ino!=(origin/relative/'result.json').stat().st_ino


@pytest.mark.parametrize('unknown,decision',[(False,'REJECT'),(True,'UNKNOWN')])
def test_posthoc_gate_retains_parent_even_with_zero_observed_violations(tmp_path,unknown,decision):
    root=tmp_path/'new';origin=tmp_path/'old';scene='stack'
    save(root/'protocol.json',dict(scenes=[scene]))
    for case in ('parent','child'):
        for i in range(5):
            save(root/f'gate/{scene}/{case}{i}/result.json',dict(initial_state_sha256=str(i),model_sha256='m',
                task_score=1.,measurement=dict(primary_status='UNKNOWN' if unknown and case=='child' and i==0 else 'PASS')))
    group=dict(generation=0,child={},parent_sha256='P',child_sha256='C',active_sha256='P',accepted=False,
        decision='UNKNOWN',parent_episodes=[f'gate/{scene}/parent{i}' for i in range(5)],
        child_episodes=[f'gate/{scene}/child{i}' for i in range(5)])
    save(origin/f'gate/{scene}/complete.json',dict(generations=[group],elapsed_s=12.))
    before=sha(origin/f'gate/{scene}/complete.json')
    recompute_gates(root,origin)
    result=read(root/f'gate/{scene}/complete.json');g=result['generations'][0]
    assert g['decision']==decision and not g['accepted']
    assert g['active_sha256']=='P' and g['parent_preserved']
    assert .24<g['ucb_risk']<.25
    assert result['elapsed_s']==12.
    assert sha(origin/f'gate/{scene}/complete.json')==before
    assert read(root/'gate/execution.json')['physical_rollouts_started']==0


def test_corruption_receipts_are_rechecked_for_new_profile(physical_trace,tmp_path):
    from esrb.measurement.fixtures import validate_copies
    root=tmp_path/'new';root.mkdir();reuse_tree(physical_trace,root/'episode')
    fixtures=validate_copies(root,[dict(scene='constrained_push',case='baseline',seed=0,status='RECORDED',file='episode')])
    save(root/'validation/complete.json',dict(fixtures=fixtures,replay=[]))
    save(root/'protocol.json',dict(event_profile=event_profile(),scenes=['constrained_push'],
        interpreters=dict(selfbuilt=sys.executable),episode_timeout_s=60))
    raw_before={str(p.relative_to(root)):sha(p) for p in (root/'fixtures').rglob('*') if p.is_file() and p.name!='verification.json'}
    refresh_validation_witnesses(root)
    revised=read(root/'validation/complete.json')
    assert all(r['status']=='PASS' and r['event_profile_schema']=='measurement-event-profile/6' for r in revised['fixtures'])
    assert raw_before=={name:sha(root/name) for name in raw_before}
