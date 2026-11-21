"""The composition layer: lightweight parsing followed by lazy dispatch."""
import argparse
from importlib import import_module
import json
from pathlib import Path
import subprocess
import sys
from . import options

TASKS = ['stack', 'pickplacecan', 'two-arm-handover', 'nut-assembly', 'wipe']


def command_registry():
    return json.loads(Path(__file__).with_name('commands.json').read_text())


def build_parser():
    parser = argparse.ArgumentParser(prog='esrb', description='ESRB experiments and engineering tools')
    parser.add_argument('--execution-worker', action='store_true', help=argparse.SUPPRESS, dest='_execution_worker')
    groups = parser.add_subparsers(dest='_group', required=True)
    benchmark = groups.add_parser('benchmark', help='In-house benchmarks and research studies')
    benchmark_commands = benchmark.add_subparsers(dest='_benchmark', required=True)
    study = benchmark_commands.add_parser('study').add_subparsers(dest='_study', required=True)
    tools = groups.add_parser('tools', help='Reports, analysis, diagnostics, rendering and explicit services')
    tool_groups = tools.add_subparsers(dest='_category', required=True)
    categories = {}
    for key, command in command_registry().items():
        if key.startswith('benchmark.'):
            leaf = benchmark_commands.add_parser(key.split('.')[-1])
        elif key.startswith('study.'):
            leaf = study.add_parser(key.split('.')[-1])
        else:
            _, category, name = key.split('.')
            if category not in categories:
                categories[category] = tool_groups.add_parser(category).add_subparsers(dest='_tool', required=True)
            leaf = categories[category].add_parser(name)
        getattr(options, command['builder'])(leaf)
        route = ['benchmark', 'study', key.split('.')[-1]] if key.startswith('study.') else key.split('.')
        leaf.set_defaults(_module=command['module'], _route=route, _leaf=leaf, _runtime=command.get('runtime'))
    # Providers retain their own interpreter and argument parser. Help never imports their dependencies.
    service = tool_groups.add_parser('service', help='Explicitly run a provider in its own environment')
    providers = service.add_subparsers(dest='name', required=True)
    for name in ['pyroki', 'sam3', 'graspnet', 'supervisor', 'prepare-sam3', 'verify-aspire']:
        provider = providers.add_parser(name, description='Pass provider arguments after --; this command starts it explicitly.')
        provider.add_argument('arguments', nargs=argparse.REMAINDER)
        provider.set_defaults(_module='esrb.tools.services')
    aspire = groups.add_parser('aspire', help='ASPIRE skill experiments; prepare services and workers explicitly')
    tasks = aspire.add_subparsers(dest='task', required=True)
    from experiments.aspire.natural_campaign import add_parser as add_campaign_parser
    add_campaign_parser(tasks)
    for task in TASKS:
        modes = tasks.add_parser(task).add_subparsers(dest='_mode', required=True)
        from experiments.aspire.natural import add_parser as add_natural_parser
        add_natural_parser(modes, task)
        for mode in ['chain', 'batch']:
            actions = modes.add_parser(mode).add_subparsers(dest='action', required=True)
            names = ['init', 'drive', 'episode', 'summarize'] if mode == 'chain' else ['init', 'prepare', 'episode', 'wave', 'drive', 'summarize']
            for action in names:
                leaf = actions.add_parser(action)
                leaf.set_defaults(_module=f'experiments.aspire.{mode}', _route=['aspire', task, mode, action],
                                  _leaf=leaf, _native_from_protocol=action == 'episode')
                leaf.add_argument('--root', required=True, type=Path)
                if action == 'init':
                    leaf.add_argument('--python', type=Path, help='Simulator interpreter (frozen into the protocol)')
                    leaf.add_argument('--ik-service-url', help='Explicitly provisioned PyRoki service')
                    leaf.add_argument('--gpu-device-id', type=int)
                    if mode == 'chain':
                        leaf.add_argument('--capsule', type=Path, required=True)
                        leaf.add_argument('--seed', type=int)
                    else:
                        leaf.add_argument('--seeds', type=int, nargs='+')
                        leaf.add_argument('--versions-per-round', type=int, choices=range(1, 6), default=5)
                if action in ['prepare', 'episode', 'wave']:
                    leaf.add_argument('--version', type=int, choices=range(1, 51 if mode == 'chain' else 6), required=True)
                if action in ['prepare', 'episode'] and mode == 'batch':
                    leaf.add_argument('--round', type=int, choices=range(1, 11), required=True)
                if action == 'prepare':
                    leaf.add_argument('--program', type=Path, required=True)
                    leaf.add_argument('--generation-base', type=Path)
                    leaf.add_argument('--rationale', default='feedback-driven candidate')
                    leaf.add_argument('--kind', choices=['candidate', 'controlled_acceptance'], default='candidate')
                if action in ['wave', 'drive'] and mode == 'batch':
                    leaf.add_argument('--rounds', type=int, nargs='+')
                    leaf.add_argument('--workers', type=int, default=10)
                if action == 'drive':
                    leaf.add_argument('--worker', action='store_true', help=argparse.SUPPRESS)
    return parser


def serialized_arguments(parser, namespace):
    """Rebuild leaf arguments, resolving paths before entering the simulator cwd."""
    result = []
    for action in parser._actions:
        if action.dest == 'help' or action.dest.startswith('_'):
            continue
        value = getattr(namespace, action.dest)
        if value is None:
            continue
        if isinstance(action, argparse._StoreTrueAction):
            if value:
                result.append(action.option_strings[0])
            continue
        if isinstance(action, argparse._StoreFalseAction):
            if not value:
                result.append(action.option_strings[0])
            continue
        if action.option_strings:
            result.append(action.option_strings[0])
        values = value if isinstance(value, list) else [value]
        result.extend(str(item.resolve()) if isinstance(item, Path) else str(item) for item in values)
    return result


def main(argv=None):
    parser = build_parser()
    namespace = parser.parse_args(argv)
    module = namespace._module
    args = argparse.Namespace(**{key: value for key, value in vars(namespace).items() if not key.startswith('_')})
    try:
        target = None
        environment = None
        if getattr(namespace, '_native_from_protocol', False):
            from esrb.runtime.io import read
            from esrb.runtime.spec import scene_from_protocol
            from esrb.runtime.batch import child_environment
            protocol = read(args.root / 'protocol.json')
            scene_from_protocol(protocol)
            target = protocol['runtime']['python']
            folder = args.root / f'round_{getattr(args, "round", 1):02d}' / f'version_{args.version:02d}'
            environment = child_environment(read(folder / 'execution_settings.json'))
        elif getattr(namespace, '_runtime', None):
            from esrb.runtime.paths import project_root
            target = project_root() / f'.esr-local/aspire/checkout/.venv-{namespace._runtime}/bin/python'
        if target and not namespace._execution_worker:
            from esrb.runtime.paths import project_root
            return subprocess.call([str(target), '-B', '-m', 'esrb', '--execution-worker',
                                    *namespace._route, *serialized_arguments(namespace._leaf, namespace)],
                                   cwd=project_root(), env=environment)
        if module == 'esrb.tools.diagnostics.robosuite':
            from experiments.aspire.scenes import get_scene
            args.scene = get_scene(args.task)
        result = import_module(module).main(args)
        return result if isinstance(result, int) else 0
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
