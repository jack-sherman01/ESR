"""Explicit provider launch commands; each runs in its existing isolated environment."""
import subprocess
import sys
from esrb.runtime.paths import project_root

SERVICES = {
    'pyroki': ('pyroki', 'aspire_pyroki_service.py'),
    'sam3': ('sam3', 'aspire_sam3_service.py'),
    'graspnet': ('graspnet', 'aspire_graspnet_cpu_service.py'),
    'supervisor': (None, 'aspire_service_supervisor.py'),
    'prepare-sam3': ('sam3', 'prepare_sam3.py'),
    'verify-aspire': (None, 'verify_aspire.py'),
}


def main(args):
    root = project_root()
    environment, name = SERVICES[args.name]
    python = root / f'.esr-local/aspire/checkout/.venv-{environment}/bin/python' if environment else sys.executable
    arguments = args.arguments[1:] if args.arguments[:1] == ['--'] else args.arguments
    return subprocess.call([str(python), '-B', str(root / 'integrations/aspire' / name), *arguments], cwd=root)
