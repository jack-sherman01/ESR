"""Source-checkout resources used by hosts and isolated simulator processes."""
import os
from pathlib import Path


def project_root():
    root = Path(os.environ.get('ESR_PROJECT_ROOT', Path(__file__).resolve().parents[2])).resolve()
    if not (root / 'pyproject.toml').is_file() or not (root / 'configs').is_dir():
        raise ValueError('ESR_PROJECT_ROOT must name an ESR source checkout')
    return root


def robosuite_python():
    return project_root() / '.esr-local/aspire/checkout/.venv-robosuite/bin/python'
