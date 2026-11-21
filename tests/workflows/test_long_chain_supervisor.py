import json
from pathlib import Path
import subprocess
import sys


def test_supervisor_records_real_child_failure_code(tmp_path):
    entry = tmp_path / 'child.py'
    entry.write_text('raise SystemExit(3)\n')
    from esrb.runtime.supervisor import supervise
    code = supervise([sys.executable, '-B', str(entry)], tmp_path)
    assert code == 3
    assert json.loads((tmp_path / 'driver_exit.json').read_text())['returncode'] == 3
