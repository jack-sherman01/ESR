"""Compose scene-specific experiments with common parallel batch execution."""
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import sys
from esrb.runtime.batch import BatchRunner
from esrb.runtime.io import read
from esrb.runtime.paths import project_root
from esrb.runtime.supervisor import supervise
from esrb.runtime.worker import drive_round
from .scenes import get_scene


def main(args):
    scene = get_scene(args.task)
    runner = BatchRunner(scene)
    root = args.root.resolve()
    if args.action == 'init':
        settings = {name: getattr(args, name) for name in ['python', 'ik_service_url', 'gpu_device_id'] if getattr(args, name) is not None}
        runner.init(root, args.seeds, args.versions_per_round, **settings)
    elif args.action == 'prepare':
        print(runner.state.prepare(root, args.round, args.version, args.program.resolve(), args.rationale,
                                   args.kind, generation_base=args.generation_base.resolve() if args.generation_base else None))
    elif args.action == 'episode':
        return runner.episode(root, args.round, args.version)
    elif args.action == 'wave':
        rounds = args.rounds or list(range(1, len(read(root / 'protocol.json')['seeds']) + 1))
        runner.wave(root, args.version, args.workers, rounds)
    elif args.action == 'drive':
        rounds = args.rounds or list(range(1, len(read(root / 'protocol.json')['seeds']) + 1))
        if not args.worker:
            return supervise([sys.executable, '-B', '-m', 'esrb', 'aspire', scene.key, 'batch', 'drive', '--root', str(root),
                              '--rounds', *map(str, rounds), '--workers', str(args.workers), '--worker'], root, cwd=project_root())
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            for future in as_completed([pool.submit(drive_round, scene, root, number) for number in rounds]):
                print(json.dumps({'completed_round': future.result()}), flush=True)
    else:
        runner.summarize(root)
    return 0
