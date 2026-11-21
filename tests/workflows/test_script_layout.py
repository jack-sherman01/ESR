"""The single command surface remains lightweight and works away from the checkout."""
from contextlib import redirect_stdout
import io
import os
from pathlib import Path
import subprocess
import sys

import pytest

from esrb.cli.main import build_parser, command_registry, TASKS

ROOT = Path(__file__).resolve().parents[2]


def command_paths():
    result = [['--help']]
    result.extend(['tools','service',name,'--help'] for name in ['pyroki','sam3','graspnet','supervisor','prepare-sam3','verify-aspire'])
    for key in command_registry():
        parts = key.split('.')
        result.append((['benchmark', 'study', parts[1]] if parts[0] == 'study' else parts) + ['--help'])
    for task in TASKS:
        for mode in ['chain', 'batch']:
            actions = ['init','drive','episode','summarize'] if mode == 'chain' else ['init','prepare','episode','wave','drive','summarize']
            result.extend(['aspire', task, mode, action, '--help'] for action in actions)
    return result


@pytest.mark.parametrize('arguments', command_paths(), ids=lambda args:' '.join(args))
def test_all_registered_commands_have_help(arguments):
    parser = build_parser()
    stream = io.StringIO()
    with redirect_stdout(stream), pytest.raises(SystemExit) as result:
        parser.parse_args(arguments)
    assert result.value.code == 0
    assert 'usage:' in stream.getvalue()


def test_module_help_is_lightweight_from_another_directory(tmp_path):
    env = {**os.environ, 'PYTHONPATH': str(ROOT), 'PYTHONDONTWRITEBYTECODE':'1'}
    script = "import sys; from esrb.cli.main import main; main(['aspire','pickplacecan','chain','init','--help'])"
    # SystemExit is intercepted only after argparse has built and rendered every option.
    script = "import sys\nfrom esrb.cli.main import main\ntry: main(['aspire','pickplacecan','chain','init','--help'])\nexcept SystemExit as e: assert e.code == 0\nassert not ({'numpy','mujoco','torch','cap_harness'} & sys.modules.keys())"
    result = subprocess.run([sys.executable, '-B', '-c', script], cwd=tmp_path, env=env, capture_output=True, text=True, timeout=30)
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize('name', ['setup_esr.sh','setup_aspire.sh'])
def test_setup_help_from_another_directory(tmp_path, name):
    result = subprocess.run(['bash', str(ROOT/'scripts'/name), '--help'], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert 'Usage:' in result.stdout


def test_setup_resolves_source_checkout(tmp_path):
    result = subprocess.run(['bash', str(ROOT/'scripts/setup_aspire.sh'), 'path'], cwd=tmp_path, capture_output=True, text=True, timeout=10)
    assert result.returncode == 0, result.stderr
    assert Path(result.stdout.strip()) == ROOT/'.esr-local/aspire/checkout'


def test_old_python_entrypoints_and_compatibility_trees_are_absent():
    assert not list((ROOT/'scripts').rglob('*.py'))
    assert not list((ROOT/'experiments').glob('run_*.py'))
    assert not (ROOT/'esrb/workflows').exists()


def test_episode_uses_frozen_interpreter_and_resolves_caller_paths(tmp_path, monkeypatch):
    from experiments.aspire.scenes import get_scene
    from esrb.runtime.batch import BatchRunner
    from esrb.cli.main import main
    runner=BatchRunner(get_scene('stack'))
    root=tmp_path/'run'
    runner.init(root,[42],1,python=tmp_path/'simulator/bin/python',gpu_device_id=2)
    source=tmp_path/'skill.py';source.write_text('result = 1\n')
    runner.state.prepare(root,1,1,source,'Initial task skill.')
    calls=[]
    def execute(command, **kwargs):
        calls.append((command,kwargs))
        return 17
    monkeypatch.setattr(subprocess,'call',execute)
    monkeypatch.chdir(tmp_path)
    assert main(['aspire','stack','batch','episode','--root','run','--round','1','--version','1'])==17
    command,options=calls[0]
    assert command[:5]==[str(tmp_path/'simulator/bin/python'),'-B','-m','esrb','--execution-worker']
    assert command[command.index('--root')+1]==str(root)
    assert options['cwd']==ROOT
    assert options['env']['MUJOCO_GL']=='egl'
    assert options['env']['MUJOCO_EGL_DEVICE_ID']=='2'


def test_old_scene_aliases_are_not_registered():
    with pytest.raises(SystemExit) as result:
        build_parser().parse_args(['aspire','pickup_can','chain','init'])
    assert result.value.code==2


def test_core_does_not_import_experiment_or_script_implementations():
    import ast
    for path in (ROOT/'esrb').rglob('*.py'):
        if path.is_relative_to(ROOT/'esrb/cli'):
            continue
        for node in ast.walk(ast.parse(path.read_text())):
            names = [node.module] if isinstance(node, ast.ImportFrom) and node.module else [alias.name for alias in node.names] if isinstance(node, ast.Import) else []
            assert not any(name.split('.')[0] in {'experiments','scripts'} for name in names), path
