"""Export already recorded Stack gate decisions; does not rerun or reassess them."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
import csv
import json
from pathlib import Path


def export(root):
    read=lambda p:json.loads(p.read_text())
    protocol=read(root/'protocol.json')
    if not (protocol.get('safety_gate') or {}).get('enabled'):
        raise ValueError('historical ungated records cannot be relabeled as gated')
    rows=[]
    for folder in sorted(root.glob('round_*/version_*')):
        if not (folder/'gate_decision.json').exists():
            continue
        decision=read(folder/'gate_decision.json')
        result=read(folder/'execution/result.json')
        proposal=read(folder/'proposal.json')
        rows.append({'round':result['round'],'seed':decision['seed'],'version':decision['version'],
                     'proposal_kind':proposal.get('kind','candidate'),
                     'program_ok':result.get('program_ok'),'native_task_success':result.get('native_task_success'),
                     'safety_status':decision['safety_status'],'decision':decision['decision'],
                     'reason':decision['reason'],'parent_before':(decision['parent_before'] or {}).get('version'),
                     'parent_after':(decision['parent_after'] or {}).get('version'),
                     'retained_parent':decision['retained_parent'],
                     'observed_safety_regression':decision['observed_safety_regression'],
                     'online_stopped':(result.get('safety') or {}).get('online_stopped'),
                     'physics_steps':(result.get('safety') or {}).get('physics_steps')})
    fields=['round','seed','version','proposal_kind','program_ok','native_task_success','safety_status','decision','reason',
            'parent_before','parent_after','retained_parent','observed_safety_regression','online_stopped','physics_steps']
    with (root/'gate_episodes.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(rows)
    actual=[r for r in rows if r['proposal_kind']!='controlled_acceptance']
    controlled=[r for r in rows if r['proposal_kind']=='controlled_acceptance']
    report={'recorded_decisions':len(rows),'promotions':sum(r['decision']=='PROMOTE' for r in rows),
            'safety_rejections':sum(r['reason']=='observed_safety_violation' for r in rows),
            'controlled_acceptance_cases':len(controlled),
            'controlled_safety_regressions':sum(r['observed_safety_regression'] for r in controlled),
            'agent_candidate_cases':len(actual),
            'agent_observed_safety_regressions':sum(r['observed_safety_regression'] for r in actual),
            'scope':'export of recorded same-seed gate decisions; not an independent safety reassessment'}
    (root/'gate_report.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('root',type=Path)
    export(parser.parse_args().root.resolve())
