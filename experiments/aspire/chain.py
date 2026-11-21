"""Compose a scene specification with the shared fifty-version runtime."""
import sys
from esrb.runtime.chain import ChainRunner
from esrb.runtime.paths import project_root
from esrb.runtime.supervisor import supervise
from .scenes import get_scene


def main(args):
    scene = get_scene(args.task)
    runner = ChainRunner(scene)
    root = args.root.resolve()
    if args.action == 'init':
        settings = {name: getattr(args, name) for name in ['python', 'ik_service_url', 'gpu_device_id'] if getattr(args, name) is not None}
        runner.init(root, args.capsule.resolve(), args.seed, **settings)
    elif args.action == 'episode':
        return runner.runner.episode(root, 1, args.version)
    elif args.action == 'drive':
        if args.worker:
            return runner.drive(root)
        return supervise([sys.executable, '-B', '-m', 'esrb', 'aspire', scene.key, 'chain', 'drive',
                          '--root', str(root), '--worker'], root, cwd=project_root())
    else:
        import json
        print(json.dumps(runner.summarize(root), indent=2))
    return 0
