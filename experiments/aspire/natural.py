"""Command composition for independent task-only natural chains."""
import json
from pathlib import Path


def main(args):
    if args.action == 'episode':
        from esrb.backends.aspire.episodes.natural import execute
        return execute(args.request.resolve())
    from esrb.runtime.natural import initialize, drive, report, Scheduler
    from .scenes import get_scene
    if args.action == 'init':
        initialize(args.root, get_scene(args.task), model=args.model, budget=args.candidates,
                   python=args.python, ik_url=args.ik_service_url, gpu=args.gpu_device_id,
                   seeds=args.dev_seeds, chain=args.chain_index)
    elif args.action in {'drive', 'resume'}:
        scheduler = Scheduler(args.workers, args.gpus)
        try:
            drive(args.root, scheduler)
        finally:
            scheduler.pool.shutdown(wait=True)
    else:
        print(json.dumps(report(args.root), ensure_ascii=False, indent=2))
    return 0


def add_parser(modes, task):
    actions = modes.add_parser('natural', help='Task-only multi-seed skill evolution, raw capture').add_subparsers(dest='action', required=True)
    for action in ['init', 'drive', 'resume', 'status', 'report', 'episode']:
        leaf = actions.add_parser(action)
        leaf.set_defaults(_module='experiments.aspire.natural', _route=['aspire', task, 'natural', action], _leaf=leaf)
        if action == 'episode':
            leaf.add_argument('--request', type=Path, required=True)
            continue
        leaf.add_argument('--root', type=Path, required=True)
        if action == 'init':
            leaf.add_argument('--dev-seeds', type=int, nargs='+', default=[1,2,3,4,5])
            leaf.add_argument('--candidates', type=int, default=50)
            leaf.add_argument('--model', default='gpt-6.1-sol')
            leaf.add_argument('--python', type=Path)
            leaf.add_argument('--ik-service-url', default='http://127.0.0.1:8140')
            leaf.add_argument('--gpu-device-id', type=int, default=3)
            leaf.add_argument('--chain-index', type=int, default=1)
        if action in {'drive', 'resume'}:
            leaf.add_argument('--workers', type=int, default=8)
            leaf.add_argument('--gpus', type=int, nargs='+', default=[3,2])
