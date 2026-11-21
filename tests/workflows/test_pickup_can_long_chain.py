from esrb.runtime.batch import BatchRunner
from esrb.runtime.chain import ChainRunner
from experiments.aspire.scenes import get_scene
from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root
from esrb.runtime.submissions import program_identity

import sys
from pathlib import Path

chain = ChainRunner(get_scene('pickplacecan'))


def test_comment_and_docstring_changes_cannot_consume_new_execution():
    baseline = '"description"\ndef move():\n    "doc"\n    return 3\nmove()\n'
    cosmetic = '# changed\n"new description"\ndef move():\n    "new doc"\n    return 3  # same motion\nmove()\n'
    assert program_identity(baseline) == program_identity(cosmetic)
    assert program_identity(baseline) != program_identity(baseline.replace('return 3', 'return 4'))


def test_fifty_version_protocol_is_frozen_without_changing_legacy_limit(tmp_path):
    root, capsule = tmp_path / 'long', tmp_path / 'worker'
    chain.init(root, capsule, seed=769153983)
    protocol = read(root / 'protocol.json')
    assert protocol['seeds'] == [769153983]
    assert protocol['versions_per_round'] == 50
    assert protocol['independent_retest'] is False
    chain.state.load_rules(root)
    assert len(list(root.glob('round_*'))) == 1
    assert read(root / 'worker_contract.json')['initial_program'] is None
    api = project_root() / 'integrations/aspire/pickup_can_skill_api.txt'
    assert (capsule / 'API.txt').read_bytes() == api.read_bytes()
    assert protocol['safety_gate']['implementation_sha256']['integrations/aspire/pickup_can_skill_api.txt'] == digest(api)
    import pytest
    with pytest.raises(ValueError, match='between 1 and 5'):
        chain.runner.init(tmp_path / 'legacy', [1], versions=50)
