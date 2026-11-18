"""Summarize measured handover chains without supplying host results to workers."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse,ast,csv,hashlib,json,statistics,sys
from datetime import datetime,timezone
from collections import Counter
from pathlib import Path
from scripts.gates.handover_gate_state import load_rules,load_chain
from scripts.feedback.handover_task_feedback import validate_prompt
from scripts.recovery.resume_fresh_handover_round import validate_task_input

def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def save(p,v):p.write_text(json.dumps(v,indent=2,ensure_ascii=False,allow_nan=False)+'\n')
def csvout(p,rows):
 if not rows:return
 with p.open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def report(root):
 load_rules(root);protocol=read(root/'protocol.json');rows=[];events=[];pairs=[];audits=[];rounds=[]
 for n,seed in enumerate(protocol['seeds'],1):
  folder=root/f'round_{n:02d}';chain=load_chain(root,n);seen=[];rrows=[]
  a=read(folder/'agent_assignment.json') if (folder/'agent_assignment.json').exists() else None
  if a:
   validate_prompt(a['initial_assignment']);assert a['fork_turns']=='none' and not a['initial_program_provided']
  for version in range(1,6):
   p=folder/f'version_{version:02d}';result=p/'execution/result.json'
   if not result.exists() or not (p/'gate_decision.json').exists():continue
   r=read(result);d=read(p/'gate_decision.json');s=r.get('safety') or {};st=s.get('stages') or {};previous=d.get('parent_before');gen=folder/f'generation_{version:02d}'
   inp=read(gen/'agent_input.json');validate_task_input(json.dumps(inp,ensure_ascii=False));validate_prompt((p/'candidate_prompt.txt').read_text());validate_prompt((p/'task_feedback.json').read_text())
   g=read(p/'generation.json');assert hashlib.sha256(inp['api'].encode()).hexdigest()==g['api_sha256']==a['api_sha256'];assert sha(result)==d['result_sha256'];assert sha(gen/'agent_input.json')==g['agent_input_sha256'];assert sha(gen/'agent_output.py')==sha(p/'program.py')==r['program_sha256']
   tree=ast.dump(ast.parse((p/'program.py').read_text()));assert tree not in seen;seen.append(tree)
   if version==1:assert inp['starting_program'] is None and inp['previous_task_feedback'] is None
   else:
    before=folder/f'generation_{version-1:02d}'; provenance=read(before/'host_provenance.json')
    assert hashlib.sha256(inp['starting_program'].encode()).hexdigest()==provenance['starting_next_sha256']
    assert inp['previous_task_feedback']==read(before/'task_feedback.json')
   assert a and g['agent_id']==a['agent_id'] and g['fork_turns']=='none'
   assert sha(p/'execution/safety_physics.jsonl.gz')==s['witness_sha256']
   audits.append({'round':n,'version':version,'agent_id':g['agent_id'],'fork_turns':'none','api_sha256':g['api_sha256'],'input_sha256':sha(gen/'agent_input.json'),'program_sha256':sha(p/'program.py'),'host_information_absent':True,'distinct_program_ast':True})
   row={'round':n,'seed':seed,'version':version,'status':r['status'],'program_ok':r.get('program_ok'),'task_success':r.get('native_task_success'),'control_ticks':r.get('control_steps'),'wall_seconds':r.get('wall_seconds'),'initial_state_id':r.get('initial_state_id'),'safety_status':s.get('status'),'coverage_complete':s.get('coverage_complete'),'physics_steps':s.get('physics_steps'),'event_kinds':','.join(e['kind'] for e in s.get('events',[])),'early_open_commands':len(s.get('precursors',[])),'pickup_observed':s.get('pickup_observed'),'receiver_close_observed':'receiver_close_command' in st,'receiver_stable_observed':'receiver_grasp_stable' in st,'actual_release_observed':'giver_release' in st,'receiver_wait_before_open_s':s.get('receiver_wait_before_open_s'),'receiver_grasp_dwell_at_open_s':s.get('receiver_grasp_dwell_at_open_s'),'overlap_before_release_s':s.get('overlap_before_release_s'),'pickup_to_release_s':st['giver_release']-st['pickup'] if 'giver_release' in st and 'pickup' in st else None,'decision':d['decision'],'reason':d['reason'],'active_before':previous.get('version') if previous else None,'active_after':(d.get('parent_after') or {}).get('version'),'retained_parent':d['retained_parent'],'observed_safety_regression':d['observed_safety_regression']}
   rows.append(row);rrows.append(row)
   for kind,seq in [('event',s.get('events',[])),('precursor',s.get('precursors',[]))]:
    for e in seq:events.append({'round':n,'version':version,'record_type':kind,'kind':e['kind'],'time_start_s':e.get('time_start_s',e.get('time_s')),'time_end_s':e.get('time_end_s'),'physics_index':e.get('physics_index'),'details_json':json.dumps(e,separators=(',',':'))})
   if previous:
    parent=next(x for x in rrows if x['version']==previous['version']);pair={'round':n,'parent_version':parent['version'],'child_version':version,'parent_task_success':parent['task_success'],'child_task_success':row['task_success'],'child_safety_status':row['safety_status'],'observed_safety_regression':d['observed_safety_regression']}
    for metric in ['control_ticks','receiver_wait_before_open_s','receiver_grasp_dwell_at_open_s','overlap_before_release_s','pickup_to_release_s']:
     pv,cv=parent[metric],row[metric];pair['parent_'+metric]=pv;pair['child_'+metric]=cv;pair['delta_'+metric]=cv-pv if pv is not None and cv is not None else None
    pair['both_task_success']=parent['task_success'] is True and row['task_success'] is True
    pairs.append(pair)
  rounds.append({'round':n,'seed':seed,'measured_versions':len(rrows),'distinct_programs':len(seen),'distinct_actual_initial_states':len({x['initial_state_id'] for x in rrows}),'successful_versions':','.join(str(x['version']) for x in rrows if x['task_success']),'active_version':(chain['active'] or {}).get('version')})
 intervals=[]
 for file in root.glob('round_*/version_*/execution/result.json'):
  intervals.append((file.with_name('execution_settings.json').stat().st_mtime,1));intervals.append((file.stat().st_mtime,-1))
 concurrent=maximum=0
 for stamp,delta in sorted(intervals):concurrent+=delta;maximum=max(maximum,concurrent)
 last=max((stamp for stamp,delta in intervals if delta==-1),default=None)
 elapsed=last-datetime.fromisoformat(protocol['created_at_utc']).timestamp() if last else None
 expected=len(protocol['seeds'])*protocol['versions_per_round'];count=Counter(e['kind'] for e in events if e['record_type']=='event');ph=sum(r['physics_steps'] or 0 for r in rows)
 summary={'status':'COMPLETE' if len(rows)==expected and all(r['status']=='MEASURED' for r in rows) and all((root/f'round_{n:02d}/fresh_agent_complete.json').exists() for n in range(1,len(protocol['seeds'])+1)) else 'IN_PROGRESS','expected':expected,'measured':len(rows),'task_success':sum(r['task_success'] is True for r in rows),'success_by_version':{v:sum(r['task_success'] is True for r in rows if r['version']==v) for v in range(1,6)},'promotions':sum(r['decision']=='PROMOTE' for r in rows),'reasons':dict(Counter(r['reason'] for r in rows)),'safety_statuses':dict(Counter(r['safety_status'] for r in rows)),'event_counts':dict(count),'release_cases':[{k:r[k] for k in ['round','version','task_success','control_ticks','event_kinds','receiver_wait_before_open_s','receiver_grasp_dwell_at_open_s','overlap_before_release_s','pickup_to_release_s']} for r in rows if r['actual_release_observed']], 'early_open_command_episodes':sum(r['early_open_commands']>0 for r in rows),'pickup_episodes':sum(r['pickup_observed'] is True for r in rows),'receiver_stable_episodes':sum(r['receiver_stable_observed'] for r in rows),'actual_release_episodes':sum(r['actual_release_observed'] for r in rows),'parented_candidates':len(pairs),'observed_safety_regressions':sum(p['observed_safety_regression'] for p in pairs),'physics_steps':ph,'maximum_observed_overlapping_execution_intervals':maximum,'init_to_last_measurement_seconds':elapsed,'incomplete_coverage':sum(r['coverage_complete'] is not True for r in rows),'infrastructure_errors':sum(r['status']!='MEASURED' for r in rows),'median_execution_seconds':statistics.median(r['wall_seconds'] for r in rows) if rows else None,'fresh_context_rounds':sum((root/f'round_{n:02d}/agent_assignment.json').exists() for n in range(1,11)),'nonexecuted_python_drafts':len(list(root.glob('round_*/nonexecuted*.py'))),'independent_retests':0,'limitations':['Same-seed development observations, no independent retests or population estimate.','A shorter failed execution does not establish faster completed handover.','Timing association alone does not establish causal attribution.','Fresh per-round model context; directory-only instructions, not OS-level filesystem isolation.','Contact-pair normal-force peaks logged each physics tick; no torque/overload qualification.'],'rounds':rounds}
 csvout(root/'gate_episodes.csv',rows);csvout(root/'handover_events.csv',events);csvout(root/'parent_child_timing.csv',pairs) if pairs else (root/'parent_child_timing.csv').write_text('round,parent_version,child_version,both_task_success,observed_safety_regression\n');csvout(root/'rounds.csv',rounds);save(root/'analysis.json',summary);save(root/'generation_input_audit.json',audits)
 print(json.dumps({k:v for k,v in summary.items() if k not in {'rounds','limitations'}},ensure_ascii=False,indent=2))
 if summary['status']=='COMPLETE':
  save(root/'completion_manifest.json',{'status':'COMPLETE','measured':len(rows),'physics_steps':ph,'protocol_sha256':sha(root/'protocol.json'),'analysis_sha256':sha(root/'analysis.json'),'generation_audit_sha256':sha(root/'generation_input_audit.json'),'initial_states_consistent':all(r['distinct_actual_initial_states']==1 for r in rounds),'five_distinct_versions_per_round':all(r['distinct_programs']==5 for r in rounds),'fresh_context':True,'no_initial_program':True,'independent_retests':0,'ik_startup_sha256':sha(root/'setup_pyroki-startup.json'), 'runtime_environment_sha256':sha(root/'runtime_environment.json'), 'workflow_recovery_sha256':sha(root/'workflow_recovery.json') if (root/'workflow_recovery.json').exists() else None, 'workflow_recovery_source_sha256':sha((Path(__file__).resolve().parents[1] / 'recovery/resume_fresh_handover_round.py')), 'report_source_sha256':sha(Path(__file__)), 'validated_sources':protocol['safety_gate']['implementation_sha256']})
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--root',required=True,type=Path);report(p.parse_args().root.resolve())
