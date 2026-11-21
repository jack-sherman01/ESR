"""Freeze the execution source closure before measuring the first candidate."""
from pathlib import Path
from .io import digest
from .paths import project_root


def implementation_paths(scene):
    root = project_root()
    paths = {'esrb/cli/commands.json', 'pyproject.toml', 'esrb/__init__.py', 'esrb/__main__.py', 'experiments/__init__.py', scene.api_path,
             'integrations/aspire/aspire_pyroki_service.py'}
    for directory in ['esrb/runtime', 'esrb/backends', 'esrb/detection', 'esrb/cli', 'experiments/aspire']:
        paths.update(str(path.relative_to(root)) for path in (root / directory).rglob('*.py'))
    return sorted(paths)


def freeze_sources(root, scene):
    repo = project_root()
    identities = {}
    for name in implementation_paths(scene):
        source = repo / name
        destination = root / 'frozen_implementation' / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(source.read_bytes())
        identities[name] = digest(destination)
    return identities


def sources_match(root, identities):
    repo = project_root()
    return bool(identities) and all(
        (repo / name).is_file() and digest(repo / name) == value
        and (root / 'frozen_implementation' / name).is_file()
        and digest(root / 'frozen_implementation' / name) == value
        for name, value in identities.items())
