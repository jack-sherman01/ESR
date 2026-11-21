"""Derived tables, evidence spot checks and figures for a frozen natural audit."""
import argparse
from collections import Counter,defaultdict
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
from esrb.runtime.io import read,save,digest
from .natural_audit import result_path,write_csv,verify_freeze


def review(root):
    protocol=verify_freeze(root);summary=read(root/'summary.json');inputs=read(root/'inputs.json')
    results={}
    for item in inputs['episodes']:
        r=read(result_path(root,item));results[(r['task'],r['chain'],r['version'],r['seed'])]=r
    changes=[];witnesses=[];covered=set();unchanged=[];panels=[];accepted_cases=[]
    for chain in inputs['chains']:
        if digest(Path(chain['root'])/'state.json')!=chain['state_sha256']:
            raise ValueError('original task lineage changed during audit')
        unchanged.append({'task':chain['task'],'chain':chain['chain'],'state_sha256':chain['state_sha256']})
        for event in chain['events']:
            if not event['parent_before']:continue
            pairs=[(results[(chain['task'],chain['chain'],event['parent_before']['version'],s)],
                    results[(chain['task'],chain['chain'],event['version'],s)]) for s in range(1,6)]
            clean=all(p['audit_status']=='PASS' for p,c in pairs)
            child_bad=any(c['audit_status']=='VIOLATION' for p,c in pairs)
            child_unknown=any(c['audit_status']=='UNKNOWN' for p,c in pairs)
            panels.append({'task':chain['task'],'chain':chain['chain'],'version':event['version'],
              'parent_version':event['parent_before']['version'],'task_accepted':event['accepted'],
              'all_five_parent_pass':clean,'parent_violation_cases':sum(p['audit_status']=='VIOLATION' for p,c in pairs),
              'child_violation_cases':sum(c['audit_status']=='VIOLATION' for p,c in pairs),
              'strict_panel_new_violation':clean and child_bad,
              'strict_panel_outcome_known':clean and (child_bad or not child_unknown),
              'scope':'only frozen assessed discrete predicates on development seeds 1-5'})
            if event['accepted']:
                for old,new in pairs:
                    newpred=[k for k,v in new['predicates'].items() if v=='VIOLATION' and old['predicates'].get(k)=='PASS']
                    if not newpred:continue
                    for witness in new['events']:
                        pair=witness.get('groups',[])
                        specific='contact:'+'|'.join(sorted(pair)) if pair else None
                        if witness['kind'] not in newpred and specific not in newpred:continue
                        contact=witness.get('contact',witness)
                        if 'contact_row' in contact:
                            pp=Path(new['execution'])/'physics'/contact['chunk']
                            with np.load(pp,allow_pickle=False) as raw:
                                cc=raw['contacts'][contact['contact_row']]
                                assert float(cc[15])==contact['normal_force_n']
                                assert cc[:2].astype(int).tolist()==contact['geom_ids']
                        accepted_cases.append({'task':chain['task'],'chain':chain['chain'],
                            'parent_version':event['parent_before']['version'],'version':event['version'],
                            'seed':new['seed'],'parent_status':old['audit_status'],'child_status':new['audit_status'],
                            'parent_all_five_pass':clean,'both_programs_task_accepted':True,
                            'new_predicates':newpred,'witness':witness,
                            'parent_result_sha256':old['result_sha256'],'child_result_sha256':new['result_sha256']})
            for seed in [1,2,3,4,5]:
                old=results[(chain['task'],chain['chain'],event['parent_before']['version'],seed)]
                new=results[(chain['task'],chain['chain'],event['version'],seed)]
                comparable=old['initial_state_id']==new['initial_state_id'] and old['model_sha256']==new['model_sha256']
                valid=comparable and old['coverage_complete'] and new['coverage_complete']
                fields={
                   'peak_n':('force_magnitude_sum','N'),'peak_100ms_mean_n':('force_magnitude_sum','N'),
                   'active_s':('force_magnitude_sum','s'),'max_continuous_active_s':('force_magnitude_sum','s')}
                for interface in ['forbidden_contact','nut_square_peg','tool_table']:
                    if interface=='nut_square_peg' and chain['task']!='nut-assembly':continue
                    if interface=='tool_table' and chain['task']!='wipe':continue
                    for channel in ['force_magnitude_sum','world_xy_magnitude_sum']:
                        for field,(_,unit) in fields.items():
                            values=[]
                            for r in [old,new]:
                                entry=r.get('metrics',{}).get('contact_interfaces',{}).get(interface)
                                values.append(entry[channel].get(field) if entry else 0. if r['coverage_complete'] else None)
                            v0,v1=values
                            changes.append({'task':chain['task'],'chain':chain['chain'],'version':event['version'],
                                'parent_version':event['parent_before']['version'],'seed':seed,'task_accepted':event['accepted'],
                                'interface':interface,'channel':channel,'measurement':field,'unit':unit,
                                'parent':v0,'child':v1,'delta_child_minus_parent':v1-v0 if valid and v0 is not None and v1 is not None else None,
                                'status':'MEASURED' if valid and v0 is not None and v1 is not None else 'UNKNOWN',
                                'risk_interpretation':'continuous change; no validated load limit or safety-margin verdict'})
    # Check one raw contact witness per task×rule. This supplements, not replaces,
    # the full reader's all-chunk SHA/shape/time/finite-identity checks.
    for item in inputs['episodes']:
        result=results[(item['task'],item['chain'],item['version'],item['seed'])]
        for event in result['events']:
            witness=event.get('contact',event)
            if 'contact_row' not in witness:continue
            key=(item['task'],event['kind'])
            if key in covered:continue
            p=Path(item['execution'])/'physics'/witness['chunk']
            with np.load(p,allow_pickle=False) as a:
                c=a['contacts'][witness['contact_row']]
                assert c[:2].astype(int).tolist()==witness['geom_ids']
                assert float(c[15])==witness['normal_force_n']
                np.testing.assert_array_equal(c[3:6],witness['point_world_m'])
                np.testing.assert_allclose(c[6:15].reshape(3,3).T@c[15:18],witness['force_world_on_geom2_n'],rtol=0,atol=0)
            covered.add(key);witnesses.append({'episode':item['id'],'kind':event['kind'],'status':'PASS','witness':witness})
    write_csv(root/'continuous_changes.csv',changes)
    write_csv(root/'panel_updates.csv',panels)
    save(root/'accepted_regression_witnesses.json',accepted_cases)
    save(root/'witness_spotchecks.json',witnesses)
    known={k:sum(v['predicates'].get(k)=='VIOLATION' for v in results.values()) for k in sorted(set().union(*(r['predicates'] for r in results.values())))}
    unknown=Counter(reason for r in results.values() for reason in set(r['unknown']))
    audit_errors=[(r['id'],s) for r in results.values() for s in r['unknown'] if 'Error:' in s]
    assert not audit_errors,audit_errors[:10]
    assert len(results)==3750 and summary['parent_proposals']==446 and summary['paired_cases']==2230
    assert summary['physics_steps']==70_249_625
    receipt={'status':'PASS','checked_at_utc':datetime.now(timezone.utc).isoformat(),
        'episodes':len(results),'physics_steps':summary['physics_steps'],'rules_sha256':protocol['rules_sha256'],
        'original_lineages_unchanged':unchanged,'raw_witness_checks':len(witnesses),
        'predicate_violation_episode_counts':known,'unknown_reasons':dict(unknown),
        'implementation_errors':audit_errors,'source_physics_bytes_read':sum(r.get('source_physics_bytes',0) for r in results.values()),
        'new_physics_episodes':0,'new_model_calls':0,
        'panel_comparison':{'all_five_parent_pass_updates':sum(p['all_five_parent_pass'] for p in panels),
            'eligible_with_known_child_outcome':sum(p['strict_panel_outcome_known'] for p in panels),
            'strict_panel_new_violation_updates':sum(p['strict_panel_new_violation'] for p in panels),
            'accepted_strict_panel_new_violation_updates':sum(p['strict_panel_new_violation'] and p['task_accepted'] for p in panels)},
        'scope':'development initial states, retrospective discrete predicates; no overall safety claim'}
    save(root/'verification.json',receipt)
    return receipt


def figures(root):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap,BoundaryNorm
    inputs=read(root/'inputs.json');matrix=[];tasks=[];qualified=[]
    for c in inputs['chains']:
        row=[];taskrow=[]
        for e in c['events']:
            case=[read(result_path(root,i)) for i in inputs['episodes'] if i['task']==c['task'] and i['chain']==c['chain'] and i['version']==e['version']]
            n=sum(r['audit_status']=='VIOLATION' for r in case)
            row.append(6 if not n and any(r['audit_status']=='UNKNOWN' for r in case) else n)
            taskrow.append(sum(r['task_qualified'] for r in case))
        matrix.append(row);qualified.append(taskrow);tasks.append(f"{c['task']} / {c['chain']}")
    fig,axes=plt.subplots(2,1,figsize=(14,9),sharex=True,constrained_layout=True)
    first=axes[0].imshow(qualified,vmin=0,vmax=5,cmap='Blues',aspect='auto')
    axes[0].set_title('Native task successes across five development seeds')
    colors=['#e6f4e8','#fee6ce','#fdae6b','#f16913','#cb181d','#67000d','#aaaaaa']
    cmap=ListedColormap(colors);second=axes[1].imshow(matrix,cmap=cmap,norm=BoundaryNorm(np.arange(-.5,7,.5)[::2],cmap.N),aspect='auto')
    axes[1].set_title('Episodes with observed violations of scoped discrete rules (grey: unknown, no observed violation)')
    for ax in axes:
        ax.set_yticks(np.arange(15));ax.set_yticklabels(tasks,fontsize=8)
        ax.set_xticks(np.arange(0,50,5));ax.set_xticklabels(np.arange(1,51,5))
        for row,c in enumerate(inputs['chains']):
            v=c.get('context_revision',{}).get('effective_from_version')
            if v:ax.plot(v-1,row,marker='|',color='black',markersize=12)
    axes[1].set_xlabel('Candidate version (black mark: model-context revision; adjacent versions are not independent replicates)')
    fig.colorbar(first,ax=axes[0],ticks=range(6),label='Task success cases / 5')
    fig.colorbar(second,ax=axes[1],ticks=range(7),label='Violation cases / 5; 6 = unknown')
    fig.savefig(root/'development_matrix.png',dpi=180);plt.close(fig)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    print(review(a.root));figures(a.root)


if __name__=='__main__':main()
