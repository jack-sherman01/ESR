"""Moved entry points must work outside the repository working directory."""
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('entry', [
    'runners/run_stack_long_chain.py',
    'runners/run_pickup_can_long_chain.py',
    'runners/run_handover_long_chain.py',
    'runners/run_nut_long_chain.py',
    'runners/run_wipe_long_chain.py',
    'runners/run_long_chain_job.py',
    'reports/report_single_seed_long_chain.py',
])
def test_direct_entry_help_from_another_directory(tmp_path, entry):
    result = subprocess.run(
        [sys.executable, '-B', str(ROOT / 'scripts' / entry), '--help'],
        cwd=tmp_path, capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 0, result.stderr
    assert 'usage:' in result.stdout.lower()


@pytest.mark.parametrize('name', ['setup_esr.sh', 'setup_aspire.sh'])
def test_moved_setup_help(tmp_path, name):
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/setup' / name), '--help'],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout


def test_moved_aspire_setup_resolves_repository_root(tmp_path):
    result = subprocess.run(
        ['bash', str(ROOT / 'scripts/setup/setup_aspire.sh'), 'path'],
        cwd=tmp_path, capture_output=True, text=True, timeout=10,
    )
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == ROOT / '.esr-local/aspire/checkout'
