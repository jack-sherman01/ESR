from esrb.runtime.batch import BatchRunner
from esrb.runtime.chain import ChainRunner
from experiments.aspire.scenes import get_scene
from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root
from esrb.runtime.submissions import program_identity

import sys
from pathlib import Path

chain = ChainRunner(get_scene('stack'))


def test_cosmetic_edits_do_not_create_new_program():
    baseline = '"description"\ndef move():\n    "doc"\n    return 3\nmove()\n'
    cosmetic = '# changed\n"new"\ndef move():\n    "new doc"\n    return 3  # same\nmove()\n'
    assert program_identity(baseline) == program_identity(cosmetic)
    assert program_identity(baseline) != program_identity(baseline.replace('return 3', 'return 4'))


def test_fifty_versions_have_one_frozen_seed_and_no_initial_program(tmp_path):
    root, capsule = tmp_path / 'long', tmp_path / 'worker'
    chain.init(root, capsule, seed=1834521)
    protocol = read(root / 'protocol.json')
    assert protocol['seeds'] == [1834521]
    assert protocol['task'] == 'cube_stack'
    assert protocol['versions_per_round'] == 50
    assert protocol['independent_retest'] is False
    assert protocol['ik_service_url'] == 'http://127.0.0.1:8128'
    chain.state.load_rules(root)
    assert len(list(root.glob('round_*'))) == 1
    assert read(root / 'worker_contract.json')['initial_program'] is None
    assert not list(capsule.glob('*.py'))
    import pytest
    with pytest.raises(ValueError, match='between 1 and 5'):
        chain.runner.init(tmp_path / 'legacy', [1], versions=50)
