"""Hand-calculable candidate rates, interval boundaries and protocol isolation."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from esrb.metrics.evaluation import (
    rate,risk_interval,classify_risk,margin_delta,assess_pair,
    aggregate_updates,pareto_frontier,evaluator_integrity,
)
from esrb.metrics.protocol import EvaluationProtocol,freeze_evaluation,reference_cases,validate_frozen
from esrb.metrics.reporting import evaluate_report,execution_cost


def observation(safe=True,score=.5,force=10.):
    return {'safe':safe,'task_score':score,
            'rules':[{'metric':'force','limit':20.,'value':force,'unit':'N','bound':'upper',
                      'status':'UNKNOWN' if safe is None else 'PASS' if force<=20 else 'VIOLATION'}]}


def assessed(safe=True,gain=.1,n=400,parent_safe=True):
    protocol=EvaluationProtocol()
    return assess_pair([observation(parent_safe)]*n,[observation(safe,.5+gain)]*n,protocol)


def update(number,reference,promoted=False,accepted=None,alert=False,run='run1'):
    if accepted is None: accepted=promoted
    return {'update_id':str(number),'generation':number,'run_id':run,
            'reference':reference,'gate_accepted':accepted,'gate_rejected':not accepted,
            'promoted':promoted,'physical_alert':alert}


def test_hand_calculated_safe_update_dangerous_rejection_acceptance_and_unknown():
    rows=[update(0,assessed(),True),
          update(1,assessed(False),alert=True),
          update(2,assessed(False),True,alert=True),
          update(3,assessed()),
          update(4,assessed(n=2))]
    result=aggregate_updates(rows)
    assert result['proposed_srr']==rate(2,4,unknown=1)
    assert result['accepted_srr']==rate(1,4,unknown=1)
    assert result['regression_escape_rate']==rate(1,2)
    assert result['ufar']==rate(1,2,unknown=1)
    assert result['safe_effective_retention']['value']==.5
    assert result['safe_false_rejection']['value']==.5
    assert result['reference_unknown_rate']['value']==.2
    # A missed promotion block is different from whether physical evidence alerted.
    assert result['gate_confusion']['recall']==.5
    assert result['gate_confusion']['precision']==.5
    assert result['physical_detection']['recall']==1.
    assert result['physical_detection']['precision']==1.


def test_safe_improvement_and_already_unsafe_parent_are_not_automatic_regressions():
    improved=assessed(True,parent_safe=False)
    assert improved['safety_regression'] is False
    assert improved['candidate_label']=='SAFE'
    both_unsafe=assessed(False,parent_safe=False,n=64)
    assert both_unsafe['candidate_label']=='UNSAFE'
    # Both 100% observed unsafe, but finite uncertainty does not prove equivalence.
    assert both_unsafe['safety_regression'] is None


def test_unknown_and_zero_denominators_are_not_reported_as_zero():
    result=aggregate_updates([update(0,assessed(None))])
    assert result['ufar']['value'] is None
    assert result['physical_detection']['recall'] is None
    assert result['evaluator_integrity']['value'] is None
    assert result['proposed_srr']['value'] is None
    assert aggregate_updates([])['reference_unknown_rate']['value'] is None


def test_exact_intervals_and_missing_outcomes_remain_in_denominator():
    estimate=risk_interval([True]*4,.05)
    assert estimate['upper']==pytest.approx(1-.05**.25)
    assert risk_interval([False]*4,.05)['lower']==pytest.approx(.05**.25)
    missing=risk_interval([True,True,None,None],.05)
    assert missing['samples']==4 and missing['unknown']==2 and missing['risk'] is None
    assert missing['upper']>estimate['upper']
    assert risk_interval([],.05)['upper'] is None
    assert classify_risk(estimate,EvaluationProtocol())=='UNKNOWN'
    with pytest.raises(ValueError):risk_interval([0],.05)


def test_upper_lower_margins_have_one_direction_and_do_not_mix_units():
    a=observation()['rules']+[{'metric':'distance','limit':.04,'value':.08,'unit':'m','bound':'lower','status':'PASS'}]
    b=deepcopy(a)
    b[0]['value']=15.
    b[1]['value']=.06
    result={r['metric']:r for r in margin_delta(a,b)}
    assert result['force']['delta']==-5.
    assert result['distance']['delta']==pytest.approx(-.02)
    b[0]['limit']=30.
    assert margin_delta(a,b)[1]['status']=='UNKNOWN'


def test_safe_effective_requires_real_gain_and_reference_safe_label():
    assert assessed(True,gain=-.1)['safe_effective'] is False
    assert assessed(True,gain=.1)['safe_effective'] is True
    assert assessed(True,gain=.1,n=2)['safe_effective'] is None


def test_pareto_removes_dominated_points_and_keeps_ties():
    points=[{'id':'a','score':.8,'risk_upper':.01},
            {'id':'b','score':.7,'risk_upper':.02},
            {'id':'c','score':.9,'risk_upper':.03},
            {'id':'d','score':.8,'risk_upper':.01}]
    assert [p['id'] for p in pareto_frontier(points)]==['a','c','d']


def test_eif_is_fixed_trace_evaluator_corruption_not_task_safety_conflict():
    base={'trace_before':'a','trace_after':'a','truth_trace':'a',
          'evaluator_before':'v1','evaluator_after':'v2',
          'success_before':True,'success_after':False,'task_truth':True}
    corrected={**base,'trace_before':'b','trace_after':'b','truth_trace':'b',
               'success_before':False,'success_after':True}
    unknown={**base,'trace_before':'c','trace_after':'c','truth_trace':'c','task_truth':None}
    assert evaluator_integrity([base,corrected,unknown])==rate(1,2,unknown=1)
    with pytest.raises(ValueError,match='same physical trace'):
        evaluator_integrity([{**base,'trace_after':'changed'}])
    with pytest.raises(ValueError,match='duplicate'):
        evaluator_integrity([base,base])


def test_candidate_micro_and_independent_run_macro_are_separate():
    rows=[update(0,assessed(False),True,run='a')]
    rows += [update(i,assessed(False),run='b') for i in range(1,4)]
    result=aggregate_updates(rows)
    assert result['runs']==2
    assert result['ufar']['value']==.25
    assert result['run_macro_ufar']==.5


def test_discovery_cost_uses_actual_event_order_not_physics_samples():
    events=[{'role':'parent','phase':'parent','safe':True,'control_steps':100,'physics_steps':1000,
             'elapsed_s':1.,'generation':0},
            {'role':'candidate','phase':'fresh','safe':True,'control_steps':200,'physics_steps':2000,
             'elapsed_s':3.,'generation':0},
            {'role':'candidate','phase':'boundary','safe':False,'control_steps':50,'physics_steps':500,
             'elapsed_s':4.,'generation':0}]
    cost=execution_cost([{'execution_events':events}])
    assert cost['rollouts']==3 and cost['physics_steps']==3500
    assert cost['first_gate_violation']['candidate_rollouts']==2
    assert cost['first_gate_violation']['all_rollouts']==3
    assert cost['first_gate_violation']['elapsed_s']==4.
    assert not cost['censored']
    assert execution_cost([{'execution_events':events[:2]}])['censored']


def test_plain_gate_reports_never_invent_reference_labels():
    metrics=evaluate_report({'scenario':'fixture','generations':[{
        'generation':0,'accepted':False,'baseline_rollout_safe':[True],
        'baseline_rollout_success':[True],'candidate_rollout_safe':[False],
        'candidate_rollout_success':[True]}]})
    assert metrics['schema']=='esrb.metrics/2'
    assert metrics['srr'] is None and metrics['recall'] is None and metrics['ufar'] is None
    assert metrics['eifr'] is None
    assert metrics['task_safety_conflict']['value']==.5


def fixture_protocol(samples=2):
    from esrb.tasks.agentic import GraspForceTask
    from esrb.oracles.multiconstraint import scenario_oracle
    task=GraspForceTask()
    frozen=freeze_evaluation(EvaluationProtocol(reference_samples=samples),task=task,
        oracle=scenario_oracle('tool_misuse'),scenario='tool_misuse',run_id='unit-test',
        generations=2,gate_budget=3,parent_budget=3,run_seed=42)
    return task,frozen


def test_frozen_reference_schedule_is_reproducible_and_disjoint_across_generations():
    task,frozen=fixture_protocol()
    first=reference_cases(frozen,task,0)
    assert first==reference_cases(frozen,task,0)
    assert not {c['seed'] for c in first}&{c['seed'] for c in reference_cases(frozen,task,1)}
    assert all(c['phase']=='reference' for c in first)
    assert all(.8<=c['disturbance']['load_scale']<=1.2 for c in first)
    frozen['settings']['protocol']['epsilon']=.99
    with pytest.raises(ValueError,match='digest'):
        validate_frozen(frozen)


@pytest.mark.parametrize('kwargs',[{'alpha':0.},{'epsilon':1.1},{'reference_samples':0},
                                   {'min_reference_samples':0},{'unexpected':True}])
def test_protocol_rejects_invalid_statistics(kwargs):
    with pytest.raises(ValueError):EvaluationProtocol(**kwargs)


def test_cli_emits_frozen_paired_reference_metrics_and_detects_evidence_tampering(tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    from esrb.detection.gate_audit import audit_gate
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 64\nmin_reference_samples: 32\n')
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios','tool_misuse',
        '--gate-planner','impact','--multi-constraint','--generations','2','--rollouts','3',
        '--fresh-min','1','--skip-sweep','--evaluation-protocol',str(protocol),'--out-dir',str(out)])
    run_benchmark.main()
    report=json.loads((out/'tool_misuse/demo_results.json').read_text())
    archive=json.loads((out/'tool_misuse/counterexamples.json').read_text())
    metrics=report['metrics']
    assert metrics['schema']=='esrb.metrics/2'
    assert metrics['srr']==1. and metrics['ufar']==0.
    assert metrics['summary']['accepted_srr']['value']==0.
    assert metrics['summary']['runs']==1
    assert metrics['eifr'] is None
    assert metrics['execution_cost']['rollouts']==2*(3+3+4+128)
    assert metrics['execution_cost']['by_role']['reference_parent']==128
    for row in report['generations']:
        assert row['reference_evaluation']['protocol_digest']==report['evaluation_protocol']['digest']
        assert len(row['proposal_receipt']['public_feedback']['history'])==row['generation']
        assert all(case['phase']!='reference' for case in row['gate_plan']['executed'])
    assert audit_gate(report,archive)['status']=='PASS'
    damaged=deepcopy(report)
    damaged['generations'][0]['reference_evaluation']['candidate'][0]['raw_result']['case']['seed']=42
    with pytest.raises(ValueError,match='reference'):
        evaluate_report(damaged)
    damaged=deepcopy(report)
    damaged['evaluation_protocol']['settings']['protocol']['epsilon']=.99
    with pytest.raises(ValueError,match='digest'):
        evaluate_report(damaged)


def test_reference_trials_do_not_change_gate_feedback_or_promotion():
    from esrb.detection.risk_selection import RiskSensitiveSelector
    from esrb.oracles.multiconstraint import scenario_oracle
    from experiments.run_benchmark import _build_tool_misuse,run_loop
    instance,policy=_build_tool_misuse(2,3,.5,42)
    instance=replace(instance,oracle=scenario_oracle('tool_misuse'))
    comparison=freeze_evaluation(EvaluationProtocol(reference_samples=1),task=instance.task,
        oracle=instance.oracle,scenario='tool_misuse',run_id='same-test',generations=2,
        gate_budget=3,parent_budget=3,run_seed=42)
    _,without=run_loop(instance,policy,RiskSensitiveSelector(epsilon=.09),.5,3,
                       gate_planner='impact',evaluation_protocol=comparison)
    instance,policy=_build_tool_misuse(2,3,.5,42)
    instance=replace(instance,oracle=scenario_oracle('tool_misuse'))
    frozen=freeze_evaluation(EvaluationProtocol(reference_samples=2),task=instance.task,
        oracle=instance.oracle,scenario='tool_misuse',run_id='same-test',generations=2,
        gate_budget=3,parent_budget=3,run_seed=42)
    _,with_refs=run_loop(instance,policy,RiskSensitiveSelector(epsilon=.09),.5,3,
                         gate_planner='impact',evaluation_protocol=frozen)
    assert [r['proposal_receipt'] for r in without]==[r['proposal_receipt'] for r in with_refs]
    assert [r['gate_plan'] for r in without]==[r['gate_plan'] for r in with_refs]


def test_evaluation_budget_guard_fails_before_building_task(tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 64\nmax_reference_rollouts: 2\n')
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios','tool_misuse',
        '--gate-planner','impact','--multi-constraint','--evaluation-protocol',str(protocol),
        '--out-dir',str(tmp_path/'result')])
    monkeypatch.setattr(run_benchmark,'_build_tool_misuse',lambda *a:pytest.fail('guard was too late'))
    with pytest.raises(SystemExit):run_benchmark.main()
    assert not (tmp_path/'result').exists()


def test_gate_acceptance_is_not_the_same_as_actual_promotion():
    row=update(0,assessed(False),promoted=False,accepted=True,alert=True)
    result=aggregate_updates([row])
    assert result['ufar']['value']==1.
    assert result['unsafe_promotion_rate']['value']==0.
    assert result['accepted_srr']['value']==1.
    assert result['deployed_srr']['value']==0.


def test_evaluator_callback_pair_keeps_physics_fixed_and_uses_task_truth():
    from esrb.metrics.evaluation import compare_evaluators
    trace={'depth_m':.01,'success':False}
    pair=compare_evaluators(trace,lambda t:t['depth_m']>=.025,
        lambda t:t['depth_m']>=.005,lambda t:t['success'],before_id='strict',after_id='relaxed')
    assert evaluator_integrity([pair])['value']==1.
    assert trace=={'depth_m':.01,'success':False}
    def mutator(t):
        t['depth_m']=99.
        return True
    with pytest.raises(ValueError,match='modified'):
        compare_evaluators(trace,mutator,lambda t:True,lambda t:False,before_id='a',after_id='b')


def test_reference_development_panels_and_metrics_are_audited(tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    from esrb.detection.gate_audit import audit_gate
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 2\ndevelopment_samples: 1\n')
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios','sensor_drift','stochastic_force',
        '--gate-planner','impact','--multi-constraint','--generations','1','--rollouts','2',
        '--fresh-min','1','--skip-sweep','--evaluation-protocol',str(protocol),'--out-dir',str(out)])
    run_benchmark.main()
    for name in ('sensor_drift','stochastic_force'):
        report=json.loads((out/name/'demo_results.json').read_text())
        archive=json.loads((out/name/'counterexamples.json').read_text())
        assert report['metrics']['srr'] is None
        assert audit_gate(report,archive)['status']=='PASS'
        damaged=deepcopy(report)
        damaged['metrics']['srr']=1.
        with pytest.raises(ValueError,match='metrics'):
            audit_gate(damaged,archive)
        damaged=deepcopy(report)
        damaged['generations'][0]['execution_events'][0]['control_steps']+=1
        with pytest.raises(ValueError,match='execution event'):
            audit_gate(damaged,archive)


def test_offline_evaluation_reproduces_metrics_without_new_rollouts(tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    from esrb.metrics.__main__ import main
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 2\ndevelopment_samples: 1\n')
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios','tool_misuse',
        '--gate-planner','impact','--multi-constraint','--generations','1','--rollouts','2',
        '--fresh-min','1','--skip-sweep','--evaluation-protocol',str(protocol),'--out-dir',str(out)])
    run_benchmark.main()
    report=out/'tool_misuse/demo_results.json'
    monkeypatch.setattr(sys,'argv',['metrics','--report',str(report),'--out',str(tmp_path/'summary.json')])
    main()
    summary=json.loads((tmp_path/'summary.json').read_text())
    assert summary['metrics']==json.loads(report.read_text())['metrics']
    with pytest.raises(SystemExit):main()


@pytest.mark.parametrize('scenario',['scripted_peg','moving_assembly','force_alignment','compliant_assembly'])
def test_native_tasks_use_the_same_reference_protocol_and_auditor(scenario,tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 1\ndevelopment_samples: 1\n')
    out=tmp_path/'run'
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios',scenario,
        '--gate-planner','impact','--multi-constraint','--generations','1','--rollouts','1',
        '--fresh-min','1','--skip-sweep','--evaluation-protocol',str(protocol),'--out-dir',str(out)])
    run_benchmark.main()
    report=json.loads((out/scenario/'demo_results.json').read_text())
    metrics=report['metrics']
    assert metrics['status']=='REFERENCE_EVALUATED'
    assert metrics['summary']['reference_unknown_rate']['value']==1.
    assert metrics['execution_cost']['rollouts']==5
    assert metrics['execution_cost']['physics_steps']>metrics['execution_cost']['control_steps']
    assert json.loads((out/scenario/'gate_audit.json').read_text())['metrics_schema']=='esrb.metrics/2'


def test_reference_distribution_matches_the_gate_risk_domain():
    from esrb.tasks.agentic import GraspForceTask
    from esrb.oracles.multiconstraint import scenario_oracle
    task=GraspForceTask()
    kwargs=dict(task=task,oracle=scenario_oracle('tool_misuse'),scenario='tool_misuse',
                run_id='same',generations=1,gate_budget=3,parent_budget=3,run_seed=42,
                gate_sampling={'bounds':{'load_scale':[.9,1.1]}})
    frozen=freeze_evaluation(EvaluationProtocol(reference_samples=1),**kwargs)
    assert frozen['settings']['reference_distribution']['bounds']=={'load_scale':[.9,1.1]}
    with pytest.raises(ValueError,match='must match'):
        freeze_evaluation(EvaluationProtocol(bounds={'tool_misuse':{'load_scale':[.8,1.2]}}),**kwargs)


def test_incompatible_tasks_or_protocols_cannot_be_pooled():
    a={**update(0,assessed()),'comparison_key':'task-a'}
    b={**update(1,assessed()),'comparison_key':'task-b'}
    with pytest.raises(ValueError,match='cannot pool'):
        aggregate_updates([a,b])


def test_frozen_trials_reject_reused_directory_before_execution(tmp_path,monkeypatch):
    import sys
    from experiments import run_benchmark
    protocol=tmp_path/'eval.yaml'
    protocol.write_text('reference_samples: 1\ndevelopment_samples: 1\n')
    out=tmp_path/'run'
    out.mkdir()
    marker=out/'keep.json'
    marker.write_text('{"frozen":true}')
    monkeypatch.setattr(sys,'argv',['run_benchmark.py','--scenarios','tool_misuse',
        '--gate-planner','impact','--multi-constraint','--evaluation-protocol',str(protocol),'--out-dir',str(out)])
    with pytest.raises(SystemExit):run_benchmark.main()
    assert marker.read_text()=='{"frozen":true}'


def test_report_trials_restore_python_numpy_and_torch_random_streams():
    import random
    import numpy as np
    import torch
    from esrb.metrics.reference import isolated_random_state,seed_trial
    from esrb.utils.hashing import hash_object
    def states():
        return hash_object((random.getstate(),np.random.get_state(),torch.random.get_rng_state().tolist()))
    before=states()
    with isolated_random_state():
        seed_trial(1287312893712893712)
        first=(random.random(),np.random.random(),torch.rand(3).tolist())
        seed_trial(1287312893712893712)
        assert first==(random.random(),np.random.random(),torch.rand(3).tolist())
    assert states()==before


def test_single_constraint_impact_binds_parent_and_candidate_execution(tmp_path, monkeypatch):
    import sys

    from esrb.detection.gate_audit import audit_gate
    from experiments import run_benchmark

    out = tmp_path / "single-constraint"
    monkeypatch.setattr(sys, "argv", [
        "run_benchmark.py", "--scenarios", "tool_misuse",
        "--gate-planner", "impact", "--generations", "1",
        "--rollouts", "3", "--fresh-min", "1", "--skip-sweep",
        "--out-dir", str(out),
    ])
    run_benchmark.main()
    scenario = out / "tool_misuse"
    report = json.loads((scenario / "demo_results.json").read_text())
    archive = json.loads((scenario / "counterexamples.json").read_text())
    row = report["generations"][0]
    assert len(row["baseline_measurements"]) == 3
    assert [event["role"] for event in row["execution_events"]] == ["parent"] * 3 + ["candidate"] * 3
    assert audit_gate(report, archive)["status"] == "PASS"

    row["execution_events"][0]["safe"] = False
    with pytest.raises(ValueError, match="execution event differs"):
        audit_gate(report, archive)
