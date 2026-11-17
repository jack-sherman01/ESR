"""Versioned, pre-execution evaluation settings and hidden reset schedules."""
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from esrb.detection.gate_cases import gate_case
from esrb.utils.hashing import hash_dict, hash_artifact


class EvaluationProtocol(BaseModel):
    """Reference trials are additional evaluations, never part of gate feedback.

    Risk intervals are exact binomial intervals. Four one-sided bounds share
    alpha within each parent/child comparison; no across-generation guarantee
    is claimed. Thresholds below define reporting and do not change the gate.
    """
    model_config = ConfigDict(extra='forbid', strict=True, allow_inf_nan=False)
    schema_version: Literal['esrb.evaluation/2'] = 'esrb.evaluation/2'
    purpose: Literal['engineering', 'research'] = 'engineering'
    development_samples: int = Field(default=4, ge=1)
    development_seed: int = Field(default=826000, ge=0)
    reference_samples: int = Field(default=64, ge=1)
    reference_seed: int = Field(default=827000, ge=0)
    max_reference_rollouts: int = Field(default=8192, ge=2)
    max_development_rollouts: int = Field(default=1024, ge=1)
    alpha: float = Field(default=0.05, gt=0, lt=1)
    epsilon: float = Field(default=0.09, ge=0, le=1)
    regression_tolerance: float = Field(default=0.02, ge=0, le=1)
    min_task_gain: float = Field(default=0.01, ge=0)
    min_reference_samples: int = Field(default=32, ge=1)
    bounds: dict[str, dict[str, list[float]]] = Field(default_factory=dict)

    @model_validator(mode='after')
    def valid_bounds(self):
        if self.development_seed == self.reference_seed:
            raise ValueError('development and reference seeds must be distinct')
        for fields in self.bounds.values():
            for name, interval in fields.items():
                if len(interval) != 2 or not interval[0] <= interval[1]:
                    raise ValueError(f'invalid reference bounds: {name}')
        return self


def load_protocol(path):
    import yaml
    # Reuse its duplicate-key loader via a small local equivalent, not its schema.
    class Loader(yaml.SafeLoader):
        def construct_mapping(self, node, deep=False):
            value = super().construct_mapping(node, deep=deep)
            if len(value) != len(node.value):
                raise ValueError('duplicate evaluation YAML key')
            return value
    return EvaluationProtocol.model_validate(yaml.load(Path(path).read_text(), Loader=Loader))


def freeze_evaluation(protocol, *, task, oracle, scenario, run_id, generations,
                      gate_budget, parent_budget, run_seed, gate_sampling=None, gate_alpha=0.1):
    """Bind actual physical domain, rules, budgets and source files before proposals."""
    # A shared profile may include other selected scenarios; only this one is resolved.
    fields = protocol.bounds.get(scenario, {})
    if fields.keys() - task.reset_bounds.keys():
        raise ValueError('unknown reference coordinate')
    bounds = {}
    for name, (low, high) in task.reset_bounds.items():
        baseline=gate_sampling['bounds'][name] if gate_sampling is not None else (low,high)
        left, right = fields.get(name, baseline)
        if not low <= left <= right <= high:
            raise ValueError(f'reference bounds outside task domain: {name}')
        bounds[name] = [float(left), float(right)]
    if gate_sampling is not None and bounds != gate_sampling['bounds']:
        raise ValueError('reference distribution must match independent gate-risk distribution')
    root = Path(__file__).resolve().parents[2]
    files = sorted((root/'esrb').rglob('*.py')) + sorted((root/'esrb/tasks').rglob('*.xml'))
    files.append(root/'experiments/run_benchmark.py')
    settings = {
        'protocol': protocol.model_dump(mode='json'), 'run_id': run_id, 'scenario': scenario,
        'generations': generations, 'oracle': oracle.protocol(),
        'task_settings': getattr(task, 'engineering_settings', {}),
        'reference_distribution': {'kind': 'product_uniform', 'bounds': bounds},
        'gate_sampling': gate_sampling,
        'gate_selector': {'method':'legacy_wilson_upper', 'alpha':gate_alpha,
                          'epsilon':protocol.epsilon,
                          'sample_scope':'independent_risk' if gate_sampling else 'mixed_gate_panel_not_iid'},
        'budgets': {'candidate_per_generation': gate_budget, 'parent_per_generation': parent_budget,
                    'reference_per_policy_per_generation': protocol.reference_samples,
                    'development_per_generation': protocol.development_samples},
        'splits': {'development': {'seed':protocol.development_seed, 'purpose':'agent_task_feedback',
                                   'namespace':'esrb.development/2'},
                   'gate': {'seed': run_seed, 'purpose': 'update_selection'},
                   'reference': {'seed': protocol.reference_seed, 'purpose': 'report_only',
                                 'namespace': 'esrb.reference/2'}},
        'statistical_unit': 'candidate_update', 'replication_unit': 'independent_run',
        'confidence_scope': 'per_update_four_one_sided_bounds_bonferroni',
        'label_source': 'heldout_rollouts_same_frozen_simulation_oracle',
        'sources': {str(p.relative_to(root)): hash_artifact(p) for p in files},
    }
    return {'settings': settings, 'digest': hash_dict(settings)}


def validate_frozen(frozen):
    if set(frozen) != {'settings', 'digest'} or hash_dict(frozen['settings']) != frozen['digest']:
        raise ValueError('frozen evaluation protocol digest mismatch')
    return EvaluationProtocol.model_validate(frozen['settings']['protocol'])


def evaluation_cases(frozen, task, generation, split):
    protocol = validate_frozen(frozen)
    settings = frozen['settings']
    if not 0 <= generation < settings['generations']:
        raise ValueError('reference generation outside frozen plan')
    if split not in {'reference','development'}:
        raise ValueError('unsupported evaluation split')
    result = []
    count = getattr(protocol,split+'_samples')
    for index in range(count):
        seed = int(hash_dict({'namespace': f'esrb.{split}/2', 'seed': getattr(protocol,split+'_seed'),
                            'run': settings['run_id'], 'scenario': settings['scenario'],
                            'generation': generation, 'index': index})[:32], 16)
        rng = np.random.default_rng(seed)
        point = {name: float(rng.uniform(*interval)) for name, interval in
                 sorted(settings['reference_distribution']['bounds'].items())}
        result.append(gate_case(task, seed, split, point))
    return result


def reference_cases(frozen, task, generation):
    return evaluation_cases(frozen, task, generation, 'reference')
