"""Independent resets and a few held-joint ticks, with native scene evidence."""
from importlib import import_module
import json
from pathlib import Path
from esrb.runtime.io import read, save, digest
from esrb.runtime.paths import project_root


def main(args):
    import numpy as np
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.contracts import ArmCommand, RobotAction
    from esrb.backends.aspire.evidence import native_state, identity, geom_group
    scene = args.scene
    registry = None
    if scene.prefix == 'pickup_can':
        from esrb.backends.aspire.pickup_can_registry import install_registry
        install_registry()
        from cap_harness.robosuite.registry import RobosuiteTaskRegistry
        registry = RobosuiteTaskRegistry()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    rules = read(project_root() / scene.rules_path)
    safety = import_module(scene.safety_module)
    classify = geom_group if scene.prefix == 'nut' else safety.classify_geometry
    capture_module = import_module(f'esrb.backends.aspire.capture.{scene.prefix}')
    capture_type = next(value for name, value in vars(capture_module).items() if name.endswith('SafetyCapture'))
    identities, reports = [], []
    for reset in range(2):
        folder = out / f'reset_{reset+1:02d}'
        folder.mkdir()
        adapter = RobosuiteAdapter(registry=registry, camera_height=128, camera_width=128)
        capture = None
        try:
            observation = adapter.reset(f'{scene.task}:0', args.seed)
            env = adapter.native_env
            model = env.sim.model
            xml = model.get_xml()
            (folder / 'model.xml').write_text(xml)
            state = native_state(env.sim)
            np.save(folder / 'initial_state.npy', state)
            identities.append(identity({'model': digest(folder / 'model.xml'), 'state': state.tolist()}))
            groups = []
            for index in range(model.ngeom):
                body, ancestors = int(model.geom_bodyid[index]), []
                while body:
                    ancestors.append(model.body_id2name(body))
                    body = int(model.body_parentid[body])
                groups.append(classify(model.geom_id2name(index), ancestors))
            save(folder / 'geometry_groups.json', groups)
            capture = capture_type(env, groups, rules, folder)
            for tick in range(args.steps):
                robot = adapter.get_robot_state()
                if scene.prefix == 'handover':
                    capture.accumulator.command(float(env.sim.data.time), robot.gripper_positions, tick)
                arms = {arm: ArmCommand(mode='joint_position', target=joints,
                        gripper_position=None if scene.prefix == 'wipe' else robot.gripper_positions[arm],
                        embodiment='robosuite') for arm, joints in robot.joint_positions.items()}
                result = adapter.step(RobotAction(arms=arms))
                if not result.ok:
                    raise ValueError(f'held-joint step failed: {result.error}')
            report = capture.finish()
            report['camera_names'] = list(observation.cameras)
            reports.append(report)
            save(folder / 'capture_summary.json', report)
        finally:
            if capture is not None:
                capture.finish()
            adapter.close()
    complete = all(report['coverage_complete'] for report in reports)
    summary = {'kind': 'structural_capture_smoke', 'task': scene.key, 'seed': args.seed,
               'same_actual_initial_state': len(set(identities)) == 1, 'initial_state_ids': identities,
               'captures': reports, 'task_success_evaluated': False, 'agent_called': False,
               'status': 'PASS' if complete and len(set(identities)) == 1 else 'FAIL'}
    save(out / 'summary.json', summary)
    print(json.dumps(summary, indent=2))
    return 0 if summary['status'] == 'PASS' else 1
