import json
from pathlib import Path
import subprocess
import sys


def test_supervisor_records_real_child_failure_code(tmp_path):
    entry = tmp_path / 'child.py'
    entry.write_text('raise SystemExit(3)\n')
    supervisor = Path(__file__).parents[2] / 'scripts/runners/run_long_chain_job.py'
    result = subprocess.run([sys.executable, '-B', str(supervisor), '--entry', str(entry), '--root', str(tmp_path)])
    assert result.returncode == 3
    assert json.loads((tmp_path / 'driver_exit.json').read_text())['returncode'] == 3
