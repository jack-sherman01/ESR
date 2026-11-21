"""Freeze, calibrate, audit and report a completed natural campaign, read-only."""
import argparse
from collections import Counter,defaultdict
from concurrent.futures import ProcessPoolExecutor,as_completed
from datetime import datetime,timezone
import csv
import json
import os
from pathlib import Path
import shutil
import time

from esrb.runtime.io import read,save,digest,fingerprint
from esrb.runtime.natural import exclusive,validate_lineage

REPO=Path(__file__).resolve().parents[3]
DEFAULT_CAMPAIGN=REPO/'.esr-local/natural-campaigns/natural-seeds1to5-r3-v50-20261118'


def inventory(campaign):
    manifest=read(campaign/'manifest.json');items=[];chains=[]
    for entry in manifest['chains']:
        root=Path(entry['root']);state=validate_lineage(root);protocol=read(root/'protocol.json')
        for field in ['sources','upstream_sources']:
            for name,expected in protocol.get(field,{}).items():
                if digest(root/'frozen_sources'/name)!=expected:raise ValueError('frozen source mismatch '+name)
        chains.append(dict(entry,state_sha256=digest(root/'state.json'),protocol_sha256=digest(root/'protocol.json'),
                           artifact_manifest_sha256=digest(root/'artifacts.json'),events=state['events'],
                           context_revision=protocol.get('context_revision')))
        for e in state['events']:
            if e['program'] is None:continue
            for case in e['cases']:
                output=Path(case['output']);result=read(output/'result.json')
                items.append({'id':f"{entry['task']}/chain_{entry['chain']:02d}/v{e['version']:02d}/seed_{case['seed']}",
                  'task':entry['task'],'chain':entry['chain'],'version':e['version'],'seed':case['seed'],
                  'execution':str(output),'result_sha256':case['result_sha256'],
                  'program_sha256':e['program_sha256'],'initial_state_id':case['initial_state_id'],
                  'model_sha256':digest(output/'model.xml'),'task_qualified':case['task_qualified'],
                  'episode_source_sha256':protocol['sources']['esrb/backends/aspire/episodes/natural.py'],
                  'capture_source_sha256':protocol['sources']['esrb/backends/aspire/capture/raw.py'],
                  'task_accepted':e['accepted'],'parent_version':(e['parent_before'] or {}).get('version')})
    return items,chains


def write_csv(path,rows):
    if not rows:return
    with path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


def flatten(result):
    return {k:result.get(k) for k in ['id','task','chain','version','seed','parent_version','task_accepted',
             'task_qualified','program_ok','native_task_success','audit_status','coverage_complete','control_steps']} | {
             'violation_kinds':';'.join(sorted({e['kind'] for e in result['events'] if e['kind']!='unconfirmed_drop'})),
             'unknown':';'.join(result['unknown']),
             'physics_steps':result.get('metrics',{}).get('physics_steps',0)}


def freeze(root,campaign,profile):
    root.mkdir(parents=True,exist_ok=False)
    items,chains=inventory(campaign)
    save(root/'inputs.json',{'campaign':str(campaign.resolve()),'episodes':items,'chains':chains})
    shutil.copyfile(profile,root/'rules.json')
    sources={}
    for name in ['esrb/detection/natural_audit.py','esrb/tools/analysis/natural_audit.py']:
        dest=root/'frozen_sources'/name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(REPO/name,dest)
        sources[name]=digest(dest)
    import platform,importlib.metadata
    save(root/'protocol.json',{'schema':'esrb.natural_audit/1','created_at_utc':datetime.now(timezone.utc).isoformat(),
         'analysis_python':platform.python_version(),
         'analysis_packages':{p:importlib.metadata.version(p) for p in ['numpy','mujoco','scipy']},
         'rules_sha256':digest(root/'rules.json'),'inputs_sha256':digest(root/'inputs.json'),'sources':sources,
         'label_timing':'retrospective; no feedback to original agents','new_model_calls':0,'new_physics_episodes':0,
         'scope':'only existing development seeds 1-5; no hidden tests or overall safety certification',
         'calibration':'archived positive/negative and deliberate corrupted computational copies; source originals immutable'})
    (root/'episodes').mkdir();return items,chains


def verify_freeze(root):
    p=read(root/'protocol.json')
    if digest(root/'rules.json')!=p['rules_sha256'] or digest(root/'inputs.json')!=p['inputs_sha256']:raise ValueError('audit protocol drift')
    for name,sha in p['sources'].items():
        if digest(REPO/name)!=sha or digest(root/'frozen_sources'/name)!=sha:raise ValueError('audit source drift '+name)
    return p


def result_path(root,item):
    return root/'episodes'/item['task']/f"chain_{item['chain']:02d}"/f"v{item['version']:02d}_seed_{item['seed']}.json"


def calibrate(root):
    """Full source episodes selected before batch statistics; not an oracle label fit."""
    from esrb.detection.natural_audit import audit_episode
    inputs=read(root/'inputs.json');rules=read(root/'rules.json');selected=[]
    # Fixed engineering sample: first program (often task failure), and last
    # qualified program in each task. Task success is never presumed safety.
    for task in sorted({x['task'] for x in inputs['episodes']}):
        candidates=[i for i in inputs['episodes'] if i['task']==task and i['seed']==1]
        first=next(i for i in candidates if i['chain']==1 and i['version']==1)
        qualified=[i for i in candidates if i['task_accepted']]
        last=max(qualified,key=lambda i:(i['version'],-i['chain']))
        for purpose,item in [('initial_task_attempt',first),('native_success_example',last)]:
            result=audit_episode(item,rules)
            out=root/'calibration'/f"{task}-{purpose}.json";out.parent.mkdir(exist_ok=True)
            save(out,result)
            selected.append({'purpose':purpose,'id':item['id'],'audit_status':result['audit_status'],
                             'unknown':result['unknown'],'events':[e['kind'] for e in result['events']],
                             'file':str(out.relative_to(root))})
    save(root/'calibration.json',{'schema':'esrb.natural_audit_calibration/1','samples':selected,
         'note':'Native task-success examples calibrate expected contacts, not assumed safety negatives. '
                'Synthetic temporal controls and corruption fixtures are in regression tests.',
         'pass':all(not any('mismatch' in s or 'Error:' in s for s in r['unknown']) for r in selected)})
    return selected


def execute(root,workers):
    verify_freeze(root);inputs=read(root/'inputs.json');rules=read(root/'rules.json')
    if not read(root/'calibration.json')['pass']:raise ValueError('engineering calibration not passed')
    from esrb.detection.natural_audit import audit_episode
    started=time.monotonic();done=0
    with exclusive(root),ProcessPoolExecutor(max_workers=workers) as pool:
        pending={}
        for item in inputs['episodes']:
            p=result_path(root,item)
            if p.exists():
                old=read(p)
                if old['result_sha256']!=item['result_sha256']:raise ValueError('audit resume identity mismatch')
                done+=1;continue
            pending[pool.submit(audit_episode,item,rules)]=item
        for future in as_completed(pending):
            item=pending[future];result=future.result();dest=result_path(root,item)
            dest.parent.mkdir(parents=True,exist_ok=True);save(dest,result);done+=1
            if done%50==0 or done==len(inputs['episodes']):
                progress={'completed':done,'total':len(inputs['episodes']),
                          'elapsed_s':time.monotonic()-started,'updated_at_utc':datetime.now(timezone.utc).isoformat()}
                save(root/'progress.json',progress);print(json.dumps(progress),flush=True)
    return report(root)


def transition(old,new):
    if old=='PASS' and new=='VIOLATION':return 'NEW_VIOLATION'
    if old=='VIOLATION' and new=='PASS':return 'RESOLVED'
    if old==new=='VIOLATION':return 'PERSISTENT_VIOLATION'
    if old==new=='PASS':return 'BOTH_PASS_ASSESSED'
    return 'UNKNOWN'


def lineage_intervals(chain,results):
    """Runs along actual accepted parents, separated by UNKNOWN and by seed/rule."""
    accepted=[e for e in chain['events'] if e['accepted']]
    rows=[];task,index=chain['task'],chain['chain']
    for seed in [1,2,3,4,5]:
        values=[results[(task,index,e['version'],seed)] for e in accepted]
        kinds=sorted(set().union(*(v['predicates'] for v in values))) if values else []
        for kind in kinds:
            codes=[v['predicates'].get(kind,'UNKNOWN') for v in values]
            pos=0;previous_end=None
            while pos<len(codes):
                if codes[pos]!='VIOLATION':pos+=1;continue
                start=pos
                while pos<len(codes) and codes[pos]=='VIOLATION':pos+=1
                end=pos
                gap=codes[previous_end:start] if previous_end is not None else []
                rows.append({'task':task,'chain':index,'seed':seed,'predicate':kind,
                    'start_version':accepted[start]['version'],'last_violating_parent_version':accepted[end-1]['version'],
                    'accepted_parent_states':end-start,'resolution_version':accepted[end]['version'] if end<len(codes) and codes[end]=='PASS' else None,
                    'end_status':codes[end] if end<len(codes) else 'RIGHT_CENSORED_AT_FINAL_PARENT',
                    'recurrence_after_known_pass':bool(previous_end is not None and gap and all(c=='PASS' for c in gap)),
                    'onset_parent_was_pass':bool(start>0 and codes[start-1]=='PASS'),
                    'present_in_first_task_parent':start==0})
                previous_end=end
    return rows


def report(root):
    protocol=verify_freeze(root);inputs=read(root/'inputs.json');results={}
    for item in inputs['episodes']:
        p=result_path(root,item)
        if not p.exists():raise ValueError('missing audit '+item['id'])
        result=read(p);results[(item['task'],item['chain'],item['version'],item['seed'])]=result
    case_rows=[];updates=[];candidate_rows=[];chain_rows=[];violations=[];intervals=[]
    for chain in inputs['chains']:
        task,index=chain['task'],chain['chain'];chain_updates=[]
        for e in chain['events']:
            children=[results[(task,index,e['version'],s)] for s in [1,2,3,4,5]]
            counts=Counter(c['audit_status'] for c in children)
            candidate_rows.append({'task':task,'chain':index,'version':e['version'],
              'parent_version':(e['parent_before'] or {}).get('version'),'task_accepted':e['accepted'],
              'task_success_cases':sum(c['task_qualified'] for c in children),
              'violation_cases':counts['VIOLATION'],'unknown_cases':counts['UNKNOWN'],
              'assessed_pass_cases':counts['PASS']})
            for child in children:
                for event in child['events']:
                    if event['kind']!='unconfirmed_drop':violations.append(dict(episode=child['id'],event=event))
            if not e['parent_before']:continue
            parent=e['parent_before'];kind_counts=Counter();pair_states=[];new_kind=set();resolved=set();persistent=set();unknown_kinds=set()
            for child in children:
                old=results[(task,index,parent['version'],child['seed'])]
                paired=(old['initial_state_id']==child['initial_state_id'] and old['model_sha256']==child['model_sha256']
                        and old['program_sha256']==parent['program_sha256'])
                label=transition(old['audit_status'],child['audit_status']) if paired else 'INITIAL_STATE_MISMATCH'
                per={}
                for kind in sorted(set(old['predicates'])|set(child['predicates'])):
                    tag=transition(old['predicates'].get(kind,'UNKNOWN'),child['predicates'].get(kind,'UNKNOWN')) if paired else 'UNKNOWN'
                    per[kind]=tag
                    if tag=='NEW_VIOLATION':new_kind.add(kind)
                    elif tag=='RESOLVED':resolved.add(kind)
                    elif tag=='PERSISTENT_VIOLATION':persistent.add(kind)
                    elif tag=='UNKNOWN':unknown_kinds.add(kind)
                kind_counts.update(per.values());pair_states.append(label)
                case_rows.append({'task':task,'chain':index,'version':e['version'],'parent_version':parent['version'],
                    'seed':child['seed'],'task_accepted':e['accepted'],'paired_identity':paired,
                    'parent_audit':old['audit_status'],'child_audit':child['audit_status'],'overall_transition':label,
                    'new_kinds':';'.join(k for k,v in per.items() if v=='NEW_VIOLATION'),
                    'resolved_kinds':';'.join(k for k,v in per.items() if v=='RESOLVED'),
                    'persistent_kinds':';'.join(k for k,v in per.items() if v=='PERSISTENT_VIOLATION'),
                    'unknown_kinds':';'.join(k for k,v in per.items() if v=='UNKNOWN'),
                    'parent_peak_forbidden_n':old.get('metrics',{}).get('contact_interfaces',{}).get('forbidden_contact',{}).get('force_magnitude_sum',{}).get('peak_n',0),
                    'child_peak_forbidden_n':child.get('metrics',{}).get('contact_interfaces',{}).get('forbidden_contact',{}).get('force_magnitude_sum',{}).get('peak_n',0)})
            row={'task':task,'chain':index,'version':e['version'],'parent_version':parent['version'],
                 'task_accepted':e['accepted'],'new_violation':bool(new_kind),'new_kinds':';'.join(sorted(new_kind)),
                 'resolved_kinds':';'.join(sorted(resolved)),'persistent_kinds':';'.join(sorted(persistent)),
                 'unknown_kinds':';'.join(sorted(unknown_kinds)),
                 'strict_parent_pass_to_child_violation_cases':pair_states.count('NEW_VIOLATION'),
                 'unknown_cases':sum(s in ['UNKNOWN','INITIAL_STATE_MISMATCH'] for s in pair_states)}
            updates.append(row);chain_updates.append(row)
        own=[c for c in candidate_rows if c['task']==task and c['chain']==index]
        own_intervals=lineage_intervals(chain,results);intervals+=own_intervals
        introduced=[u for u in chain_updates if u['new_violation']];accepted_new=[u for u in introduced if u['task_accepted']]
        active=(chain['events'][-1]['parent_after'] or {}).get('version')
        active_rows=[r for r in own if r['version']==active]
        earlier_danger=any(r['task_accepted'] and r['violation_cases'] for r in own if r['version']!=active)
        first=next((e['version'] for e in chain['events'] if e['accepted']),None)
        chain_rows.append({'task':task,'chain':index,'first_task_parent':first,'final_task_parent':active,
           'task_accepted':sum(e['accepted'] for e in chain['events']),'parent_proposals':len(chain_updates),
           'new_violation_updates':len(introduced),'accepted_new_violation_updates':len(accepted_new),
           'first_new_violation_version':introduced[0]['version'] if introduced else None,
           'first_new_after_task_parent':introduced[0]['version']-first if introduced else None,
           'maximum_violating_accepted_parent_states':max((r['accepted_parent_states'] for r in own_intervals),default=0),
           'resolved_seed_rule_intervals':sum(r['resolution_version'] is not None for r in own_intervals),
           'recurrent_seed_rule_intervals':sum(r['recurrence_after_known_pass'] for r in own_intervals),
           'final_parent_violation_cases':active_rows[0]['violation_cases'] if active_rows else None,
           'earlier_accepted_danger_hidden_by_final_pass':bool(active_rows and not active_rows[0]['violation_cases'] and not active_rows[0]['unknown_cases'] and earlier_danger)})
    flat=[flatten(r) for r in results.values()]
    for name,rows in [('episodes.csv',flat),('candidates.csv',candidate_rows),('paired_cases.csv',case_rows),
                      ('updates.csv',updates),('chains.csv',chain_rows),('lineage_intervals.csv',intervals)]:write_csv(root/name,rows)
    with (root/'event_witnesses.jsonl').open('w') as f:
        for row in violations:f.write(json.dumps(row,ensure_ascii=False)+'\n')
    scene_rows=[]
    for task in sorted({r['task'] for r in flat}):
        subset=[r for r in flat if r['task']==task];cs=[r for r in candidate_rows if r['task']==task];us=[r for r in updates if r['task']==task]
        counts=Counter(r['audit_status'] for r in subset)
        scene_rows.append({'task':task,'episodes':len(subset),'task_success_episodes':sum(r['task_qualified'] for r in subset),
          'assessed_pass':counts['PASS'],'violation':counts['VIOLATION'],'unknown':counts['UNKNOWN'],
          'task_success_and_violation':sum(r['task_qualified'] and r['audit_status']=='VIOLATION' for r in subset),
          'unsafe_candidates':sum(r['violation_cases']>0 for r in cs),'parent_proposals':len(us),
          'new_violation_updates':sum(r['new_violation'] for r in us),
          'accepted_new_violation_updates':sum(r['new_violation'] and r['task_accepted'] for r in us)})
    summary={'status':'COMPLETE','rules_sha256':protocol['rules_sha256'],'episodes':len(flat),'candidates':len(candidate_rows),
       'parent_proposals':len(updates),'paired_cases':len(case_rows),
       'physics_steps':sum(r.get('metrics',{}).get('physics_steps',0) for r in results.values()),
       'episode_status':dict(Counter(r['audit_status'] for r in flat)),
       'coverage_complete_episodes':sum(r['coverage_complete'] for r in results.values()),
       'new_violation_updates':sum(r['new_violation'] for r in updates),
       'accepted_new_violation_updates':sum(r['new_violation'] and r['task_accepted'] for r in updates),
       'lineage_intervals':{'count':len(intervals),'unit':'seed × predicate; coarse and component-specific predicates overlap',
            'resolved':sum(r['resolution_version'] is not None for r in intervals),
            'recurrent_after_known_pass':sum(r['recurrence_after_known_pass'] for r in intervals)},
       'scenes':scene_rows,'chains':chain_rows,'new_physics_episodes':0,'agent_feedback_modified':False,
       'limitations':['retrospective development seeds only','PASS means only assessed discrete predicates',
        'no validated overload or torque/jam verdicts','no overall or population safety guarantee',
        'rules calibrated after generation; event rates conditional on this frozen post-hoc profile']}
    save(root/'summary.json',summary)
    lines=['# 五场景自然更新：离线安全审计','',
           '只读审计已有开发轨迹；没有新增仿真、隐藏初态或反馈回写。PASS只覆盖本次已实现离散规则，不能称为整体安全。','',
           f"共 {len(flat)} 回合、{len(candidate_rows)} 候选、{len(updates)} 个实际任务父子提案、{len(case_rows)} 个同初态配对。",'',
           '|场景|回合|离散规则PASS|违规|UNKNOWN|任务成功且违规|新增违规更新|其中被接纳|',
           '|---|---:|---:|---:|---:|---:|---:|---:|']
    for s in scene_rows:lines.append('|'+ '|'.join(str(s[k]) for k in ['task','episodes','assessed_pass','violation','unknown','task_success_and_violation','new_violation_updates','accepted_new_violation_updates'])+'|')
    lines+=['','新增违规按具体规则逐项配对，父子同时有其他违规时，也可能新增另一类违规；整体父PASS→子违规另外保留在updates.csv。',
      '连续力、横向载荷与原始传感器读数只报告测量；未验证损坏/过载阈值、力矩和卡滞资格保持UNKNOWN。',
      '正常目标接触、几何落座释放、双方合法抓持与擦拭接触分别处理。任务失败不自动产生安全违规。',
      '完整证据见 episodes/；首次事件附原始chunk、接触行、部件、物理步、时间、坐标系与力。',
      '父子身份、UNKNOWN及各类持续/消失见 paired_cases.csv；任务接纳谱系见chains.csv与inputs.json。',
      '规则与分析源代码在protocol.json和frozen_sources/绑定；完整性失败不填成零事件。','']
    (root/'REPORT.md').write_text('\n'.join(lines))
    write_csv(root/'scenes.csv',scene_rows)
    return summary


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['freeze','calibrate','run','report'])
    p.add_argument('--root',type=Path,required=True);p.add_argument('--campaign',type=Path,default=DEFAULT_CAMPAIGN)
    p.add_argument('--profile',type=Path,default=REPO/'configs/evaluation/natural_offline_v1.json')
    p.add_argument('--workers',type=int,default=12);a=p.parse_args()
    if a.action=='freeze':value=freeze(a.root,a.campaign,a.profile);print('frozen episodes',len(value[0]))
    elif a.action=='calibrate':print(json.dumps(calibrate(a.root),ensure_ascii=False))
    elif a.action=='run':print(json.dumps(execute(a.root,a.workers),ensure_ascii=False))
    else:print(json.dumps(report(a.root),ensure_ascii=False))


if __name__=='__main__':main()
