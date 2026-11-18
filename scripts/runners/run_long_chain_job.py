"""Detached trial coordinator supervisor retaining the actual child exit code."""

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse
from datetime import datetime, timezone
import json
import subprocess
import sys
from pathlib import Path

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--entry', required=True, type=Path)
    parser.add_argument('--root', required=True, type=Path)
    args = parser.parse_args()
    root = args.root.resolve()
    with (root / 'driver.log').open('a') as log:
        child = subprocess.run([sys.executable, '-B', str(args.entry.resolve()), 'drive', '--root', str(root)], stdout=log, stderr=subprocess.STDOUT)
    (root / 'driver_exit.json').write_text(json.dumps({'returncode': child.returncode, 'finished_at_utc': datetime.now(timezone.utc).isoformat()}, indent=2) + '\n')
    sys.exit(child.returncode)
