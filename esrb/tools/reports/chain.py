"""Export a completed or partial chain and audit its existing evidence, without rollouts."""
import json
from esrb.runtime.chain import ChainRunner
from esrb.runtime.gate_state import GateState
from esrb.runtime.io import read, save, digest
from esrb.runtime.spec import scene_from_protocol

def main(args):
    root=args.root.resolve()
    protocol=read(root/'protocol.json')
    scene=scene_from_protocol(protocol)
    summary=ChainRunner(scene).summarize(root)
    state=GateState(scene)
    state.load_rules(root)
    chain=state.load_chain(root,1)
    checks=[]
    for event in chain['events']:
        candidate=root/'round_01'/f'version_{event["version"]:02d}'
        result=read(candidate/'execution/result.json')
        proposal=read(candidate/'proposal.json')
        witness=candidate/'execution/safety_physics.jsonl.gz'
        checks.append({'version':event['version'],
             'program_bound':digest(candidate/'program.py')==proposal['source_sha256']==result['program_sha256']==event['candidate_sha256'],
             'result_bound':digest(candidate/'execution/result.json')==event['result_sha256'],
             'witness_bound':witness.is_file() and digest(witness)==result['safety']['witness_sha256'],
             'initial_state_bound':result['initial_state_id']==read(root/'round_01/initial_reference.json')['initial_state_id'],
             'protocol_bound':result['protocol_sha256']==digest(root/'protocol.json')})
    valid=all(all(value for key,value in row.items() if key!='version') for row in checks)
    validation={'status':'PASS' if valid else 'FAIL','checks':checks,'independent_retest':False,
                'scope':'artifact identity and saved decision audit; no physical reassessment'}
    save(root/'validation.json',validation)
    from .gate import export
    gate_report=export(root)
    analysis={**summary,'artifact_validation':validation['status'],'gate_report':gate_report}
    save(root/'analysis.json',analysis)
    save(root/'completion_manifest.json',{'status':summary['status'] if valid else 'INTEGRITY_ERROR',
         'measured':summary['measured'],'protocol_sha256':digest(root/'protocol.json'),
         'analysis_sha256':digest(root/'analysis.json'),'validation_sha256':digest(root/'validation.json')})
    print(json.dumps(analysis,ensure_ascii=False,indent=2))
    return 0 if valid else 1
