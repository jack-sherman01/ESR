import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))


@pytest.mark.parametrize('name,task,port', [('handover', 'two_arm_handover', 8130), ('wipe', 'spill_wipe', 8131)])
def test_fresh_fifty_version_contract(tmp_path, name, task, port):
    import importlib
    module = importlib.import_module('scripts.runners.run_' + name + '_long_chain')
    root, capsule = tmp_path / 'run', tmp_path / 'worker'
    module.init(root, capsule, seed=321455)
    module.state.load_rules(root)
    protocol = module.read(root / 'protocol.json')
    assert protocol['task'] == task
    assert protocol['versions_per_round'] == 50
    assert protocol['ik_service_url'].endswith(':' + str(port))
    assert protocol['seeds'] == [321455]
    assert module.read(root / 'worker_contract.json')['initial_program'] is None
    assert not list(capsule.glob('*.py'))


def test_wipe_feedback_only_adds_task_marker_counts(tmp_path):
    from scripts.runners import run_wipe_long_chain as module
    folder = tmp_path / 'execution'
    folder.mkdir()
    module.save(folder / 'result.json', {'last_telemetry': {'wiped_markers': 3, 'num_markers': 100, 'force': 999}, 'safety': {'status': 'VIOLATION'}})
    data = module.feedback(tmp_path)
    assert data['task_progress'] == {'cleaned_markers': 3, 'total_markers': 100}
    assert 'force' not in str(data) and 'safety' not in str(data)
