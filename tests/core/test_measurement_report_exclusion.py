"""Authorized detection exclusions must not manufacture safety evidence."""
import copy
import hashlib
import json
import pytest

from esrb.measurement.reporting import legacy_metrics,detection_exclusions


def panel():
    groups=[]
    for g in range(3):
        def row(i,child):
            missing=child and g==0 and i==2
            return dict(file=f'gate/pickplacecan/{"primary" if child else "baseline"}-seed{42+1000*g+i:04d}-g{g}-{"child" if child else "parent"}',
                seed=42+1000*g+i,task_success=not child,task_score=0. if child else 1.,
                measurement=dict(primary_status='UNKNOWN' if missing else 'VIOLATION' if child else 'PASS',
                    missing=['unconfirmed drop outcome for body 31'] if missing else [],signals={'drop_height_loss_m':0.}))
        groups.append(dict(generation=g,parent_sha256='parent',accepted=False,
            parents=[row(i,False) for i in range(5)],children=[row(i,True) for i in range(5)]))
    return groups


def test_one_unknown_excluded_only_from_detection_and_original_panel_unchanged():
    groups=panel();before=copy.deepcopy(groups);episode=groups[0]['children'][2]['file']
    result=legacy_metrics(groups,12.,excluded_detection_episodes=[episode])
    assert result['recall']==result['precision']==result['f1']==1.
    assert result['detection_evidence']['confusion_matrix']==dict(TP=14,TN=15,FP=0,FN=0)
    assert result['detection_evidence']['effective_samples']==29
    assert result['detection_evidence']['total_samples']==30
    assert result['controlled_label_counts']==dict(positive=14,negative=15)
    assert result['srr']==result['ufar']==0.
    assert result['observed_unsafe_candidates']==3
    assert result['rdc_wall_s']==12. and result['rdc_rollouts']==15
    assert groups==before


def test_unlisted_unknown_is_not_silently_classified():
    with pytest.raises(ValueError,match='unlisted UNKNOWN'):legacy_metrics(panel(),12.)


def test_another_unknown_still_blocks_unless_explicitly_excluded():
    groups=panel();episode=groups[0]['children'][2]['file']
    groups[1]['children'][0]['measurement'].update(primary_status='UNKNOWN',missing=['missing frame'])
    with pytest.raises(ValueError,match='unlisted UNKNOWN'):
        legacy_metrics(groups,12.,excluded_detection_episodes=[episode])


def test_unknown_cannot_make_otherwise_safe_candidate_unsafe_or_safe():
    groups=panel();episode=groups[0]['children'][2]['file']
    for i,row in enumerate(groups[0]['children']):
        if i!=2:row['measurement']['primary_status']='PASS'
    with pytest.raises(ValueError,match='candidate safety is unresolved'):
        legacy_metrics(groups,12.,excluded_detection_episodes=[episode])


def test_report_policy_binds_exclusion_to_original_unknown_result(tmp_path):
    groups=panel();row=groups[0]['children'][2];file=tmp_path/row['file']/'result.json'
    file.parent.mkdir(parents=True);file.write_text(json.dumps(row))
    exclusion=dict(episode=row['file'],result_sha256=hashlib.sha256(file.read_bytes()).hexdigest(),reason='User-authorized exclusion of one UNKNOWN')
    (tmp_path/'reporting_policy.json').write_text(json.dumps(dict(schema='measurement-report-policy/1',
        controlled_detection_exclusions=dict(pickplacecan=[exclusion]))))
    assert detection_exclusions(tmp_path,'stack',groups)==[]
    assert detection_exclusions(tmp_path,'pickplacecan',groups)==[exclusion]
    file.write_text(file.read_text()+' ')
    with pytest.raises(ValueError,match='changed'):detection_exclusions(tmp_path,'pickplacecan',groups)


def test_resolved_or_missing_episode_cannot_still_be_excluded():
    groups=panel()
    with pytest.raises(ValueError,match='only UNKNOWN'):
        legacy_metrics(groups,12.,excluded_detection_episodes=[groups[0]['children'][0]['file']])
