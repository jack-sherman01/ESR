"""Bind candidate metrics to audited reference evidence and actual execution costs."""
from collections import Counter
from math import isfinite

from esrb.utils.hashing import hash_dict
from .evaluation import assess_pair, aggregate_updates
from .protocol import EvaluationProtocol, validate_frozen, reference_cases, evaluation_cases


def require(value,message):
    if not value:
        raise ValueError(message)


def audit_reference(report, task, profile):
    """Recompute held-out labels; a different phase string alone is not independence."""
    from esrb.oracles.multiconstraint import scenario_oracle, verdict_safety
    from esrb.detection.gate_audit import _audit_measurement
    frozen=report['evaluation_protocol']
    protocol=validate_frozen(frozen)
    settings=frozen['settings']
    require(settings['scenario']==report['scenario'], 'reference scenario mismatch')
    require(settings['run_id']==report['run_id'],'reference run identity mismatch')
    require(settings['gate_sampling']==report.get('risk_sampling'),'frozen gate distribution mismatch')
    require(settings['gate_selector']['alpha']==report.get('selector_alpha',.1)
            and settings['gate_selector']['epsilon']==report['args']['epsilon'],
            'frozen gate selector mismatch')
    require(settings['generations']==len(report['generations']), 'reference generation count mismatch')
    require(settings['oracle']==report['measurement_protocol'], 'reference oracle protocol mismatch')
    limits={rule['metric']:rule['limit'] for rule in settings['oracle']['rules'] if rule['limit'] is not None}
    oracle=scenario_oracle(report['scenario'],limits=limits)
    require(oracle.protocol()==settings['oracle'],'reference rules differ from frozen oracle')
    panels={}
    all_reference_seeds=set()
    exposed=set()
    for row in report['generations']:
        for observed in row.get('baseline_measurements',[]):
            exposed.add(observed['raw_result']['case']['seed'])
        exposed.update(c['seed'] for c in row.get('gate_plan',{}).get('executed',[]))
        exposed.update(item['raw_result']['case']['seed'] for item in row.get('reference_evaluation',{}).get('development',[]))
    for row in report['generations']:
        gen=row['generation']
        receipt=row['proposal_receipt']
        panel=row.get('reference_evaluation')
        require(panel is not None and panel['protocol_digest']==frozen['digest'],'missing or foreign reference panel')
        cases=reference_cases(frozen,task,gen)
        require(not ({case['seed'] for case in cases}&exposed),'reference cases overlap gate/development measurements')
        seeds={case['seed'] for case in cases}
        require(len(seeds)==len(cases) and not seeds&all_reference_seeds,'duplicate reference trials')
        all_reference_seeds |= seeds
        paired={}
        for role in ('parent','candidate','development'):
            wrappers=panel[role]
            expected_cases=evaluation_cases(frozen,task,gen,'development') if role=='development' else cases
            require(len(wrappers)==len(expected_cases),'reference/development sample count mismatch')
            components=receipt[('candidate' if role=='development' else role)+'_components']
            digest=profile.policy_digest(components)
            normalized=[]
            for observed,case in zip(wrappers,expected_cases,strict=True):
                raw=observed['raw_result']
                require(observed['evidence_id']==hash_dict(raw),'reference evidence digest mismatch')
                require(raw['case']==case and raw['policy_digest']==digest,'reference case or policy mismatch')
                trace=raw['episode_data']
                require(trace['reset_readback']==case['disturbance'],'reference reset mismatch')
                config=components.get('data') if getattr(profile,'nominal_task_config',None) is not None else None
                require(trace.get('applied_task_config')==config,'reference task configuration mismatch')
                verdict=_audit_measurement(raw,oracle,gen,observed['evidence_id'],task)
                evaluator=task()
                try:
                    if config is not None:
                        evaluator.configure(**config)
                    assessed_task=evaluator.evaluate(trace)
                    require(raw['task_success'] is assessed_task.success and raw['task_score']==assessed_task.task_score,
                            'reference task truth does not match physical task evaluation')
                finally:
                    if hasattr(evaluator,'close'):
                        evaluator.close()
                normalized.append({'safe':verdict_safety(verdict),'task_score':raw['task_score'],
                                   'rules':verdict.rules})
            if role!='development':
                paired[role]=normalized
        panels[gen]=assess_pair(paired['parent'],paired['candidate'],protocol)
    return panels


def execution_cost(rows):
    """All episodes count as executions; only candidate events discover gate violations."""
    events=[event for row in rows for event in row.get('execution_events',[])]
    if not events:
        return {'status':'NOT_MEASURED','first_gate_violation':None,'censored':True}
    for i,event in enumerate(events):
        require(type(event['control_steps']) is int and event['control_steps']>=0,'invalid execution step count')
        require(type(event['elapsed_s']) in (int,float) and isfinite(event['elapsed_s'])
                and event['elapsed_s']>=0 and (i==0 or event['elapsed_s']>=events[i-1]['elapsed_s']),
                'execution clock must be finite and monotonic')
    first=next((i for i,e in enumerate(events) if e['role']=='candidate' and e['safe'] is False),None)
    prefix=events[:first+1] if first is not None else []
    known_physics=[e['physics_steps'] for e in events if e.get('physics_steps') is not None]
    return {'status':'MEASURED','rollouts':len(events),
            'by_role':dict(Counter(e['role'] for e in events)),
            'candidate_by_phase':dict(Counter(e['phase'] for e in events if e['role']=='candidate')),
            'control_steps':sum(e['control_steps'] for e in events),
            'physics_steps':sum(known_physics) if len(known_physics)==len(events) else None,
            'physics_steps_known':sum(known_physics),'physics_unknown_episodes':len(events)-len(known_physics),
            'last_observation_elapsed_s':events[-1]['elapsed_s'],
            'censored':first is None,
            'first_gate_violation':None if first is None else {
                'all_rollouts':len(prefix), 'candidate_rollouts':sum(e['role']=='candidate' for e in prefix),
                'control_steps':sum(e['control_steps'] for e in prefix),
                'elapsed_s':prefix[-1]['elapsed_s'], 'generation':prefix[-1]['generation'],
                'scope':'from_run_loop_start_including_parent_trials_and_any_earlier_reference_trials'}}


def evaluate_report(report):
    """Without a frozen reference panel, research rates remain unavailable."""
    frozen=report.get('evaluation_protocol')
    protocol=validate_frozen(frozen) if frozen else EvaluationProtocol()
    require(not any('reference_evaluation' in row for row in report['generations']) or frozen is not None,
            'reference panels without frozen protocol')
    assessed={}
    if frozen:
        from esrb.detection.gate_audit import TASKS_BY_NAME
        from esrb.agent_api.profiles import BY_INSTANCE
        from esrb.tasks.extensions.settings import configured_task
        task_type=TASKS_BY_NAME[report['task_name']]
        task_type=configured_task(task_type,frozen['settings']['task_settings']) if frozen['settings']['task_settings'] else task_type
        assessed=audit_reference(report,task_type,BY_INSTANCE[report['instance_name']])
    rows=[]
    run_id=frozen['settings']['run_id'] if frozen else report.get('run_id','unidentified_run')
    for row in report['generations']:
        receipt=row.get('proposal_receipt',{})
        gen=row['generation']
        physical=row.get('candidate_rollout_safe',[])
        alert=True if False in physical else False if physical and None not in physical else None
        blockers=list(row.get('gate_plan',{}).get('blocking_tests',[]))
        accepted=row['accepted']
        promoted=receipt.get('promoted',accepted)
        if not accepted and not blockers:
            blockers=['risk_ucb_or_no_eligible_candidate']
        reference=assessed.get(gen,assess_pair([],[],protocol))
        comparison_key=hash_dict({'scenario':report['scenario'],'reference_protocol':protocol.model_dump(mode='json'),
                                  'oracle':frozen['settings']['oracle'] if frozen else None,
                                  'distribution':frozen['settings']['reference_distribution'] if frozen else None})
        rows.append({'run_id':run_id,'generation':gen,'slot':0,'comparison_key':comparison_key,
                     'update_id':hash_dict({'run_id':run_id,'scenario':report['scenario'],'generation':gen,
                                           'parent':receipt.get('base_version'), 'candidate':receipt.get('candidate_version')}),
                     'parent_version':receipt.get('base_version'),'candidate_version':receipt.get('candidate_version'),
                     'origin':receipt.get('manifest',{}).get('origin','unidentified'),
                     'gate_accepted':accepted,'gate_rejected':not accepted,'promoted':promoted,
                     'physical_alert':alert,'rejection_reasons':blockers,
                     'reference':reference})
    summary=aggregate_updates(rows)
    cost=execution_cost(report['generations'])
    observed=Counter()
    conflicts=Counter()
    for row in report['generations']:
        for role in ('baseline','candidate'):
            safe=row.get(role+'_rollout_safe',[])
            success=row.get(role+'_rollout_success',[])
            observed.update({role+'_violations':safe.count(False),role+'_unknown':safe.count(None),
                             role+'_rollouts':len(safe)})
            conflicts.update({'task_success_and_violation':sum(s and v is False for s,v in zip(success,safe)),
                              'task_safety_known':sum(v is not None for v in safe)})
    return {'schema':'esrb.metrics/2','scenario':report['scenario'],'run_id':run_id,'status':'REFERENCE_EVALUATED' if frozen else 'NO_INDEPENDENT_REFERENCE',
            'statistical_unit':'candidate_update','replication_unit':'independent_run',
            'protocol_digest':frozen['digest'] if frozen else None,
            'label_source':frozen['settings']['label_source'] if frozen else None,
            'confidence_scope':'per_update_only; no simultaneous guarantee across generations',
            'updates':rows,'summary':summary,'execution_cost':cost,'observed_gate_panel':dict(observed),
            'task_safety_conflict':rate_conflict(conflicts),
            'eifr':summary['evaluator_integrity']['value'],
            'srr':summary['proposed_srr']['value'],'ufar':summary['ufar']['value'],
            'recall':summary['physical_detection']['recall'],
            'precision':summary['physical_detection']['precision'],'f1':summary['physical_detection']['f1'],
            'pareto':summary['pareto'],
            'notes':['EIF requires same-trace evaluator rescoring with independent task truth; not collected by this runner.',
                     'Safe-effective means reference-safe with observed mean task gain; it is not a significance claim.',
                     'The reference oracle is isolated from selection, not an independent physical sensor.',
                     'A distinct run UUID alone does not establish independent optimization runs.']}


def rate_conflict(counts):
    from .evaluation import rate
    return rate(counts['task_success_and_violation'],counts['task_safety_known'])


def print_metrics(metrics):
    summary=metrics['summary']
    def fmt(value):
        return 'N/A' if value is None else f'{value:.3f}'
    print(f"\n  Evaluation v2: {metrics['status']}")
    for label,key in (('Proposed SRR','proposed_srr'),('Accepted SRR','accepted_srr'),
                      ('UFAR','ufar'),('Safe-effective retention','safe_effective_retention')):
        measure=summary[key]
        print(f"  {label}: {fmt(measure['value'])} ({measure['numerator']}/{measure['denominator']}; unknown={measure['unknown']})")
    print('  Rejection reasons:',summary['rejection_reasons'])
    print('  Execution cost:',metrics['execution_cost'].get('rollouts','N/A'),'episodes')
    print('  EIF: N/A (requires fixed-trace independent task comparisons)')


def audit_execution_events(report):
    """Bind cost events to the actual recorded episodes, including separate DEV/REF."""
    from types import SimpleNamespace
    from esrb.detection.gate_cases import control_steps
    from esrb.detection.gate_checks import has_oracle_evidence
    from esrb.oracles.base import OracleVerdict
    for row in report['generations']:
        expected=[]
        for wrapper in row.get('baseline_measurements',[]):
            expected.append(('parent',wrapper['raw_result']))
        for case in row['gate_plan']['executed']:
            expected.append(('candidate',case['raw_result']))
        panel=row.get('reference_evaluation')
        if panel:
            expected.extend(('development_candidate',w['raw_result']) for w in panel['development'])
            for parent,candidate in zip(panel['parent'],panel['candidate'],strict=True):
                expected.extend([('reference_parent',parent['raw_result']),('reference_candidate',candidate['raw_result'])])
        events=row.get('execution_events',[])
        require(len(events)==len(expected),'execution event count does not match raw episodes')
        for event,(role,raw) in zip(events,expected,strict=True):
            oracle = raw['oracle']
            if 'status' in oracle:
                status = oracle['status']
                require(status in {'PASS', 'VIOLATION', 'UNKNOWN'}, 'invalid oracle status')
                safe = False if status == 'VIOLATION' else True if status == 'PASS' else None
            else:
                verdict = OracleVerdict(**oracle)
                observation = SimpleNamespace(oracle_verdict=verdict, episode_data=raw['episode_data'])
                safe = verdict.safe if has_oracle_evidence(observation) else None
            require(event['role']==role and event['phase']==raw['case']['phase']
                    and event['seed']==raw['case']['seed'] and event['generation']==row['generation']
                    and event['safe'] is safe and event['control_steps']==control_steps(raw['episode_data']),
                    'execution event differs from physical episode')
        if 'evaluation_protocol' in report:
            budget=report['evaluation_protocol']['settings']['budgets']
            require(len(row['baseline_measurements'])==budget['parent_per_generation']
                    and len(row['gate_plan']['executed'])<=budget['candidate_per_generation'],
                    'execution count exceeds frozen budget')
    execution_cost(report['generations'])
