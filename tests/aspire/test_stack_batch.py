from esrb.runtime.batch import BatchRunner
from esrb.runtime.chain import ChainRunner
from experiments.aspire.scenes import get_scene
from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root
from esrb.runtime.submissions import program_identity

"""Batch bookkeeping must not turn missing measurements into success."""
import importlib.util
import json
from pathlib import Path

MODULE = BatchRunner(get_scene('stack'))


def test_random_round_panel_is_distinct_and_requires_all_five_versions(tmp_path):
    root=tmp_path/'batch'
    MODULE.init(root)
    protocol=json.loads((root/'protocol.json').read_text())
    assert len(set(protocol['seeds']))==10
    assert all(0 < seed < 2**31 for seed in protocol['seeds'])
    assert protocol['versions_per_round']==5 and protocol['stop_on_success'] is False
    assert not protocol['independent_retest']
    assert protocol['safety_gate']['enabled']
    MODULE.summarize(root)
    summary=json.loads((root/'summary.json').read_text())
    assert summary['status']=='IN_PROGRESS' and summary['measured_rollouts']==0


def test_success_does_not_hide_missing_versions_or_an_infrastructure_error(tmp_path):
    root=tmp_path/'batch'
    MODULE.init(root)
    protocol=json.loads((root/'protocol.json').read_text())
    for number,seed in enumerate(protocol['seeds'],1):
        for version in range(1,6):
            folder=root/f'round_{number:02d}'/f'version_{version:02d}'/'execution'
            folder.mkdir(parents=True)
            save(folder/'result.json',dict(round=number,version=version,seed=seed,status='MEASURED',
                        native_task_success=(version==1),initial_state_id=f'initial-{number}'))
    MODULE.summarize(root)
    summary=read(root/'summary.json')
    # Fifty task measurements without gate decisions cannot complete a gated chain.
    assert summary['status']=='IN_PROGRESS' and summary['measured_rollouts']==50
    assert summary['promotions']==0
    assert summary['native_success_rollouts']==10
    assert all(r['distinct_actual_initial_states']==1 for r in summary['rounds'])
    # Retain compatibility with the historical, explicitly ungated archive.
    del protocol['safety_gate']
    save(root/'protocol.json',protocol)
    MODULE.summarize(root)
    assert read(root/'summary.json')['status']=='COMPLETE'
    save(root/'round_10/version_05/execution/result.json',dict(round=10,version=5,seed=protocol['seeds'][9],status='INFRASTRUCTURE_ERROR'))
    MODULE.summarize(root)
    summary=read(root/'summary.json')
    assert summary['status']=='IN_PROGRESS' and summary['measured_rollouts']==49
    assert summary['infrastructure_errors']==1


def test_small_engineering_panel_does_not_claim_fifty_requested_rollouts(tmp_path):
    root=tmp_path/'acceptance'
    MODULE.init(root,[974038773],versions=3)
    MODULE.summarize(root)
    summary=read(root/'summary.json')
    assert summary['requested_rollouts']==3 and summary['status']=='IN_PROGRESS'
