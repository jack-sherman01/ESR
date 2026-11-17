"""Report-only paired trials; no labels enter selection, memory or public feedback."""
from copy import deepcopy
from contextlib import contextmanager
import random
import sys

import numpy as np
from dataclasses import replace

from esrb.agent.evolution_loop import EvolutionLoop
from esrb.utils.hashing import hash_dict
from .protocol import reference_cases, evaluation_cases


def _collect_reference(instance, parent, candidate, generation, frozen, *, profile,
                      receipt, parent_config, candidate_config, encode, record_event):
    """Use a separate task and fresh policy copies after the gate decision is fixed."""
    panels = {'parent': [], 'candidate': [], 'development': [], 'protocol_digest': frozen['digest']}
    task = type(instance.task)()
    isolated = EvolutionLoop(replace(instance,task=task),parent)
    original = (hash_dict(profile.snapshot(parent)), hash_dict(profile.snapshot(candidate)))
    try:
        for case in evaluation_cases(frozen,task,generation,'development'):
            seed_trial(case['seed'])
            if candidate_config is not None:
                task.configure(**candidate_config)
            measured=isolated._run_rollout(deepcopy(candidate),case['seed'],generation,
                                          reset_options={'disturbance':case['disturbance']})
            raw=encode(measured,case,profile.policy_digest(receipt['candidate_components']))
            panels['development'].append({'raw_result':raw,'evidence_id':hash_dict(raw)})
            record_event(measured,'development_candidate',case,task=task)
        for case in reference_cases(frozen,task,generation):
            for role, policy, config in (('parent',parent,parent_config),('candidate',candidate,candidate_config)):
                seed_trial(case['seed'])
                if config is not None:
                    task.configure(**config)
                replica = deepcopy(policy)
                result = isolated._run_rollout(replica,case['seed'],generation,
                                               reset_options={'disturbance':case['disturbance']})
                digest = profile.policy_digest(receipt[role+'_components'])
                raw = encode(result,case,digest)
                panels[role].append({'raw_result':raw,'evidence_id':hash_dict(raw)})
                record_event(result,'reference_'+role,case,task=task)
    finally:
        if hasattr(task,'close'):
            task.close()
    if original != (hash_dict(profile.snapshot(parent)),hash_dict(profile.snapshot(candidate))):
        raise ValueError('reference evaluation changed live policy state')
    return panels


@contextmanager
def isolated_random_state():
    """Serial runner isolation: report-only trials must not consume optimizer RNG."""
    python_state=random.getstate()
    numpy_state=np.random.get_state()
    torch=sys.modules.get('torch')
    torch_state=torch.random.get_rng_state() if torch is not None else None
    cuda_state=(torch.cuda.get_rng_state_all() if torch is not None and torch.cuda.is_initialized() else None)
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        if torch is not None:
            torch.random.set_rng_state(torch_state)
            if cuda_state is not None:
                torch.cuda.set_rng_state_all(cuda_state)


def seed_trial(seed):
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch=sys.modules.get('torch')
    if torch is not None:
        torch.default_generator.manual_seed(seed % (2**63))
        if torch.cuda.is_initialized():
            torch.cuda.manual_seed_all(seed % (2**63))


def collect_reference(*args,**kwargs):
    """Collect separate DEV/REFERENCE trials without changing global random streams."""
    with isolated_random_state():
        return _collect_reference(*args,**kwargs)
