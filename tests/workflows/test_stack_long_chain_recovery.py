import sys
from pathlib import Path

import pytest

from esrb.runtime.interruption import archive_incomplete_execution


def test_interrupted_launch_is_preserved_and_cannot_be_overwritten(tmp_path):
    candidate = tmp_path / 'round_01/version_16'
    execution = candidate / 'execution'
    execution.mkdir(parents=True)
    (execution / 'telemetry.jsonl').write_bytes(b'partial evidence')
    (candidate / 'program.py').write_text('immutable program')
    archive = archive_incomplete_execution(tmp_path, 16)
    assert (archive / 'execution/telemetry.jsonl').read_bytes() == b'partial evidence'
    assert (candidate / 'program.py').read_text() == 'immutable program'
    with pytest.raises(FileExistsError):
        archive_incomplete_execution(tmp_path, 16)


@pytest.mark.parametrize('name', ['execution/result.json', 'gate_decision.json'])
def test_completed_result_or_decision_is_never_archived(tmp_path, name):
    candidate = tmp_path / 'round_01/version_16'
    (candidate / 'execution').mkdir(parents=True)
    (candidate / name).write_text('{}')
    with pytest.raises(ValueError, match='refuse'):
        archive_incomplete_execution(tmp_path, 16)
    assert (candidate / name).exists()
    assert not (tmp_path / 'interrupted_attempts').exists()
