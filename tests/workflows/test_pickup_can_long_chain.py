import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parents[2] / 'scripts'))
from scripts.runners import run_pickup_can_long_chain as chain


def test_comment_and_docstring_changes_cannot_consume_new_execution():
    baseline = '"description"\ndef move():\n    "doc"\n    return 3\nmove()\n'
    cosmetic = '# changed\n"new description"\ndef move():\n    "new doc"\n    return 3  # same motion\nmove()\n'
    assert chain.program_identity(baseline) == chain.program_identity(cosmetic)
    assert chain.program_identity(baseline) != chain.program_identity(baseline.replace('return 3', 'return 4'))


def test_fifty_version_protocol_is_frozen_without_changing_legacy_limit(tmp_path):
    root, capsule = tmp_path / 'long', tmp_path / 'worker'
    chain.init(root, capsule, seed=769153983)
    protocol = chain.read(root / 'protocol.json')
    assert protocol['seeds'] == [769153983]
    assert protocol['versions_per_round'] == 50
    assert protocol['independent_retest'] is False
    chain.state.load_rules(root)
    assert len(list(root.glob('round_*'))) == 1
    assert chain.read(root / 'worker_contract.json')['initial_program'] is None
    api = chain.REPO / 'integrations/aspire/pickup_can_skill_api.txt'
    assert (capsule / 'API.txt').read_bytes() == api.read_bytes()
    assert protocol['safety_gate']['implementation_sha256']['integrations/aspire/pickup_can_skill_api.txt'] == chain.sha(api)
    import pytest
    with pytest.raises(ValueError, match='between 1 and 5'):
        chain.runner.init(tmp_path / 'legacy', [1], versions=50)
