"""Atomic, per-round policy admission history for the PickPlaceCan runner."""
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path

from scripts.gates.pickup_can_gate_rules import decide, validate_rules
from scripts.feedback.pickup_can_task_feedback import archive_inputs, validate_prompt


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def fingerprint(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),allow_nan=False).encode()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def save(path,value):
    path=Path(path); temp=path.with_name(path.name+f'.{os.getpid()}.tmp')
    temp.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temp.replace(path)


@contextmanager
def round_lock(folder):
    with (folder/'.gate.lock').open('a') as lock:
        try:
            fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError('another process owns this round; candidates must run sequentially') from exc
        try:
            yield
        finally:
            fcntl.flock(lock,fcntl.LOCK_UN)


def load_rules(root):
    protocol=read(root/'protocol.json')
    gate=protocol.get('safety_gate') or {}
    if gate.get('enabled') is not True:
        raise ValueError('legacy batch has no safety gate; create a new gated batch, preserving historical records')
    if digest(root/'safety_rules.json')!=gate['rules_sha256']:
        raise ValueError('frozen safety rules changed')
    repo=Path(__file__).resolve().parents[2]
    if not gate.get('implementation_sha256') or any(digest(repo/name)!=value for name,value in gate['implementation_sha256'].items()):
        raise ValueError('frozen gate implementation changed; create a new batch')
    rules=read(root/'safety_rules.json');validate_rules(rules)
    return rules


def load_chain(root,number):
    folder=root/f'round_{number:02d}'
    path=folder/'gate_chain.json'
    chain=read(path) if path.exists() else {'schema':'esrb.pickup_can_gate_chain/1','events':[],'active':None}
    active,previous=None,None
    for index,event in enumerate(chain['events'],1):
        content={k:v for k,v in event.items() if k!='event_sha256'}
        if (event['version']!=index or event['previous_event_sha256']!=previous
                or event['event_sha256']!=fingerprint(content) or event['parent_before']!=active):
            raise ValueError('broken gate decision history')
        active,previous=event['parent_after'],event['event_sha256']
    if active!=chain['active']:
        raise ValueError('active parent differs from decision history')
    if active is not None:
        for field,sha_field in [('program_path','program_sha256'),('evidence_path','evidence_sha256'),('witness_path','witness_sha256')]:
            if digest(root/active[field])!=active[sha_field]:
                raise ValueError('frozen parent artifact changed')
    return chain


def preflight(root,number,version):
    rules=load_rules(root)
    if not 1<=version<=read(root/'protocol.json')['versions_per_round']:
        raise ValueError('candidate exceeds the frozen per-round budget')
    chain=load_chain(root,number)
    if version!=len(chain['events'])+1:
        raise ValueError('candidate version does not follow the last gate decision')
    folder=root/f'round_{number:02d}'/f'version_{version:02d}'
    proposal=read(folder/'proposal.json')
    active=chain['active']
    expected={'version':active['version'],'program_sha256':active['program_sha256']} if active else None
    if proposal.get('gate_base')!=expected or 'gate_base' not in proposal:
        raise ValueError('candidate must bind to the current active parent, not the last rejected candidate')
    validate_prompt((folder/'candidate_prompt.txt').read_text())
    if proposal.get('candidate_prompt_sha256')!=digest(folder/'candidate_prompt.txt') or proposal.get('task_feedback_sha256')!=digest(folder/'task_feedback.json'):
        raise ValueError('candidate task input changed')
    if proposal.get('source_sha256')!=digest(folder/'program.py'):
        raise ValueError('candidate program differs from its proposal')
    return rules,chain


def prepare(root,number,version,program,rationale,kind='candidate',generation_base=None):
    load_rules(root)
    if generation_base is None:
        raise ValueError('explicit task generation base required')
    validate_prompt(rationale)
    round_folder=root/f'round_{number:02d}'
    with round_lock(round_folder):
        chain=load_chain(root,number)
        if version!=len(chain['events'])+1:
            raise ValueError('prepare the next version after the recorded gate decision')
        active=chain['active']
        source=program.read_bytes()
        folder=round_folder/f'version_{version:02d}';folder.mkdir(exist_ok=False)
        (folder/'program.py').write_bytes(source)
        archive_inputs(folder,generation_base.read_text(),round_folder/f'version_{version-1:02d}' if version>1 else None)
        base={'version':active['version'],'program_sha256':active['program_sha256']} if active else None
        save(folder/'proposal.json',{'round':number,'version':version,'gate_base':base,
             'source_sha256':digest(folder/'program.py'),'rationale':rationale,'kind':kind,
             'candidate_prompt_sha256':digest(folder/'candidate_prompt.txt'),'task_feedback_sha256':digest(folder/'task_feedback.json'),
             'parent_program':active['program_path'] if active else None})
        save(folder/'execution_settings.json',{'renderer':'egl','gpu_device_id':(number-1)%4,
             'camera_sampling_hz':5,'fresh_explicit_observation':True})
        return folder


def commit(root,number,version):
    # Caller holds the round lock across execution and admission.
    protocol=read(root/'protocol.json');gate=protocol['safety_gate']
    folder=root/f'round_{number:02d}'/f'version_{version:02d}'
    chain=load_chain(root,number);active=chain['active']
    if version!=len(chain['events'])+1:
        raise ValueError('gate decision already committed or out of order')
    result=read(folder/'execution/result.json');proposal=read(folder/'proposal.json')
    expected={'version':active['version'],'program_sha256':active['program_sha256']} if active else None
    safety=result.get('safety') or {}
    reference=root/f'round_{number:02d}'/'initial_reference.json'
    repo=Path(__file__).resolve().parents[2]
    code_bound=all(digest(repo/name)==value for name,value in gate['implementation_sha256'].items())
    binding=(code_bound and digest(folder/'candidate_prompt.txt')==proposal.get('candidate_prompt_sha256')
             and digest(folder/'task_feedback.json')==proposal.get('task_feedback_sha256') and proposal.get('gate_base')==expected and 'gate_base' in proposal
             and digest(folder/'program.py')==proposal.get('source_sha256')==result.get('program_sha256')
             and proposal.get('round')==number and proposal.get('version')==version
             and result.get('round')==number and result.get('version')==version
             and result.get('seed')==protocol['seeds'][number-1]
             and result.get('protocol_sha256')==digest(root/'protocol.json')
             and digest(root/'safety_rules.json')==gate['rules_sha256']==safety.get('rules_sha256')
             and reference.exists() and result.get('initial_state_id')==read(reference)['initial_state_id'])
    witness=folder/'execution/safety_physics.jsonl.gz'
    binding=bool(binding and witness.exists() and digest(witness)==safety.get('witness_sha256'))
    if active is not None:
        binding &= (active['initial_state_id']==result.get('initial_state_id') and active['rules_sha256']==gate['rules_sha256'])
    decision=decide(result,active=active,binding_ok=binding,
                    expected_settle_steps=read(root/'safety_rules.json')['settle_control_steps'])
    after=active
    if decision['accepted']:
        parent=root/f'round_{number:02d}'/'parents'/f'version_{version:02d}'
        parent.mkdir(parents=True,exist_ok=False)
        (parent/'program.py').write_bytes((folder/'program.py').read_bytes())
        (parent/'evidence.json').write_bytes((folder/'execution/result.json').read_bytes())
        after={'version':version,'program_sha256':digest(parent/'program.py'),
               'program_path':str((parent/'program.py').relative_to(root)),
               'evidence_path':str((parent/'evidence.json').relative_to(root)),
               'evidence_sha256':digest(parent/'evidence.json'),
               'witness_path':str(witness.relative_to(root)),'witness_sha256':digest(witness),
               'rules_sha256':gate['rules_sha256'],'initial_state_id':result['initial_state_id'],
               'seed':result['seed'],'qualification':'PASS_WITHIN_FROZEN_RULE_SCOPE'}
    decision.update(version=version,seed=protocol['seeds'][number-1],parent_after=after,
                    candidate_sha256=proposal['source_sha256'],result_sha256=digest(folder/'execution/result.json'),
                    rules_sha256=gate['rules_sha256'],
                    previous_event_sha256=chain['events'][-1]['event_sha256'] if chain['events'] else None)
    decision['event_sha256']=fingerprint(decision)
    chain['events'].append(decision);chain['active']=after
    save(root/f'round_{number:02d}'/'gate_chain.json',chain)
    save(folder/'gate_decision.json',decision)
    # Convenience view only; gate_chain.json is the authoritative state.
    save(root/f'round_{number:02d}'/'active_parent.json',after)
    return decision
