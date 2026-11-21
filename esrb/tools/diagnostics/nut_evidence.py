"""Execute one generated nut-assembly skill with native ASPIRE and dense evidence.

Host-only instrumentation wraps the upstream adapter; generated policies retain
the unchanged public API and cannot inspect the native simulator or these files.
"""
from __future__ import annotations


import argparse
import gzip
import hashlib
import importlib
import json
from pathlib import Path
import sys
import traceback
from types import SimpleNamespace

import numpy as np
import mujoco

from esrb.backends.aspire.evidence import (contact_rows, digest, geometry_inventory, identity,
                                 native_state, read, save)


class Capture:
    def __init__(self, adapter, root, protocol):
        self.adapter, self.root, self.protocol = adapter, root, protocol
        self.env, self.sim = adapter.native_env, adapter.native_env.sim
        self.inventory = geometry_inventory(self.env)
        self.initial = native_state(self.sim)
        self.pre, self.post = [], []
        self.actions, self.frames = [], []
        self.control_index, self.phase = 0, "program"
        self.stream = gzip.open(root / "physics.jsonl.gz", "wt", compresslevel=1)
        save(root / "geometry.json", self.inventory)
        (root / "model.xml").write_text(self.sim.model.get_xml())
        mujoco.mj_saveModel(self.sim.model._model, str(root / "model.mjb"))
        np.save(root / "initial_state.npy", self.initial)
        reference = Path(protocol["smoke_reference"])
        self.reset_matches = (np.array_equal(np.load(reference / "initial_state.npy"), self.initial)
                              and digest(reference / "model.xml") == digest(root / "model.xml"))
        if not self.reset_matches:
            raise ValueError("INITIAL_STATE_MISMATCH against frozen seed-1 smoke")
        self.initial_id = identity({"model": digest(root / "model.xml"), "state": self.initial.tolist()})
        self.nut = self.env.obj_body_id["SquareNut"]
        self.peg = self.sim.model.body_name2id("peg1")
        self.table_z = float(self.sim.data.get_site_xpos("table_top")[2])
        self.finger_qpos = int(self.env.robots[0]._ref_gripper_joint_pos_indexes[0])
        self.grasped = self.lifted = False
        self.drop = False
        self.original_step = self.sim.step
        self.sim.step = self.physical_step

    def task_frame(self, contacts):
        data = self.sim.data
        nut, peg = data.body_xpos[self.nut], data.body_xpos[self.peg]
        nut_rotation = data.body_xmat[self.nut].reshape(3, 3)
        peg_rotation = data.body_xmat[self.peg].reshape(3, 3)
        rel = peg_rotation.T @ (nut - peg)
        rotation = peg_rotation.T @ nut_rotation
        yaw = float(np.arctan2(rotation[1, 0], rotation[0, 0]))
        tilt = float(np.arccos(np.clip(rotation[2, 2], -1., 1.)))
        extent = self.protocol["peg_halfwidth_m"] * (abs(np.cos(yaw)) + abs(np.sin(yaw)))
        fits = bool(max(abs(rel[0]), abs(rel[1])) + extent <= self.protocol["nut_hole_halfwidth_m"])
        bottom = float(nut[2] - self.protocol["nut_halfthickness_m"])
        seated = bool(fits and tilt <= self.protocol["task_max_tilt_rad"]
                      and abs(bottom - self.table_z) <= self.protocol["task_seat_tolerance_m"])
        touching_gripper = any({c["group1"], c["group2"]} == {"gripper", "square_nut"}
                               and c["wrench_contact_raw"][0] > 1e-6 for c in contacts)
        gripper = float(np.clip(data.qpos[self.finger_qpos] / .04, 0., 1.))
        eef = data.get_site_xpos(self.env.robots[0].gripper.important_sites["grip_site"])
        released = bool(not touching_gripper and gripper >= .8 and np.linalg.norm(eef - nut) >= .06)
        unexpected = [c["pair"] for c in contacts if c["wrench_contact_raw"][0] > 1e-6
                      and ((c["group1"] in {"gripper", "robot_link"} and c["group2"] in {"square_peg", "round_peg", "table"})
                           or (c["group2"] in {"gripper", "robot_link"} and c["group1"] in {"square_peg", "round_peg", "table"}))]
        self.grasped |= touching_gripper
        self.lifted |= bool(self.grasped and bottom - self.table_z >= .04)
        impact = any("square_nut" in {c["group1"], c["group2"]}
                     and ({c["group1"], c["group2"]} & {"table", "floor"})
                     and c["wrench_contact_raw"][0] > 1e-6 for c in contacts)
        self.drop |= bool(self.lifted and not touching_gripper and impact and not seated)
        stage = ("verify" if seated and released else "release" if seated else "insert" if fits and bottom < peg[2] + .1
                 else "align" if self.lifted and np.linalg.norm(rel[:2]) < .08
                 else "transport" if self.lifted else "grasp" if touching_gripper else "approach")
        return {"time_s": float(data.time), "stage": stage, "nut_position_world_m": nut.tolist(),
                "nut_relative_peg_m": rel.tolist(), "relative_yaw_rad": yaw, "tilt_rad": tilt,
                "nut_bottom_world_m": bottom, "table_top_world_m": self.table_z,
                "geometric_fit": fits, "seated": seated, "released": released,
                "independent_task_frame": seated and released, "gripper_nut_contact": touching_gripper,
                "unexpected_contacts": unexpected, "drop_observed": self.drop}

    def physical_step(self, *args, **kwargs):
        before = native_state(self.sim)
        t0 = float(self.sim.data.time)
        self.original_step(*args, **kwargs)
        contacts = contact_rows(self.sim.model._model, self.sim.data._data, self.inventory)
        frame = self.task_frame(contacts)
        frame["time_s"] = t0  # native kinematics/contact solve refer to the interval start
        sample = {"physics_index": len(self.pre), "control_index": self.control_index,
                  "phase": self.phase, "time_start_s": t0, "time_end_s": float(self.sim.data.time),
                  "contacts": contacts, "task_frame": frame}
        self.stream.write(json.dumps(sample, allow_nan=False) + "\n")
        self.pre.append(before)
        self.post.append(native_state(self.sim))
        self.frames.append(frame)

    def finish(self):
        self.sim.step = self.original_step
        self.stream.close()
        size = len(self.initial)
        np.savez_compressed(self.root / "physics_states.npz", pre=np.asarray(self.pre).reshape(-1, size),
                            post=np.asarray(self.post).reshape(-1, size))
        save(self.root / "actions.json", self.actions)
        tail_count = int(self.protocol["verification_steps"] / self.adapter.control_frequency / self.sim.model.opt.timestep)
        tail = self.frames[-tail_count:]
        task = bool(len(tail) == tail_count and all(f["independent_task_frame"] for f in tail))
        if tail:
            positions = np.asarray([f["nut_position_world_m"] for f in tail])
            drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
            task &= drift <= self.protocol["task_stability_drift_m"]
        else:
            drift = None
        violations = sum(bool(f["unexpected_contacts"]) for f in self.frames)
        return {"initial_state_id": self.initial_id, "initial_state_matches_smoke": self.reset_matches,
                "control_steps": self.control_index, "physics_steps": len(self.pre),
                "physics_timestep_s": float(self.sim.model.opt.timestep), "independent_task_success": task,
                "native_task_success": bool(self.env._check_success()), "grasp_contact_observed": self.grasped,
                "lift_observed": self.lifted, "drop_observed": self.drop,
                "unexpected_contact_frames": violations, "verification_drift_m": drift,
                "safety_status": "VIOLATION" if violations or self.drop else "PASS",
                "safety_scope": "contact identity and observed drop; load limits not enabled",
                "stage_counts": dict(__import__('collections').Counter(f["stage"] for f in self.frames)),
                "last_task_frame": self.frames[-1] if self.frames else None}


def execute(args):
    from cap_harness.contracts import ArmCommand, RobotAction
    from cap_harness.robosuite.adapter import RobosuiteAdapter
    from cap_harness.runtime import ProgramExecutor
    run_module = importlib.import_module("cap_harness.run")
    adapter_module = importlib.import_module("cap_harness.robosuite.adapter")
    protocol = read(args.protocol)
    args.out.mkdir(parents=True, exist_ok=False)
    save(args.out / "protocol.json", protocol)
    (args.out / "capture_source.py").write_bytes(Path(__file__).read_bytes())
    holder = {}

    class Instrumented(RobosuiteAdapter):
        def _after_native_reset(self, env):
            super()._after_native_reset(env)
            holder["adapter"] = self
            holder["capture"] = Capture(self, args.out, protocol)

        def native_step(self, action):
            capture = holder.get("capture")
            if capture is not None:
                capture.actions.append({"index": capture.control_index, "phase": capture.phase,
                                        "time_s": float(self.native_env.sim.data.time),
                                        "targets": {arm: {"joints_rad": cmd.target.tolist(), "gripper": cmd.gripper_position}
                                                    for arm, cmd in action.arms.items()}})
            result = super().native_step(action)
            if capture is not None and result.ok:
                capture.control_index += 1
            return result

        def close(self):
            if "capture" in holder and "summary" not in holder:
                holder["summary"] = holder["capture"].finish()
            super().close()

    class ObservedExecutor(ProgramExecutor):
        def execute_program(self, code):
            result = super().execute_program(code)
            holder["program_execution"] = result
            adapter, capture = holder["adapter"], holder["capture"]
            capture.phase = "post_program_verification"
            for _ in range(min(protocol["verification_steps"], protocol["max_control_steps"] - capture.control_index)):
                state = adapter.get_robot_state()
                step = adapter.step(RobotAction(arms={"primary": ArmCommand(
                    mode="joint_position", target=state.joint_positions["primary"],
                    gripper_position=adapter._gripper_targets["primary"], embodiment="robosuite")}))
                if not step.ok or step.terminated or step.truncated:
                    break
            return result

    adapter_module.RobosuiteAdapter = Instrumented
    run_module.ProgramExecutor = ObservedExecutor
    try:
        outcome = run_module.run_program(benchmark="robosuite", suite="nut_assembly", task_id=0,
            seed=protocol["seed"], program_path=args.program.resolve(), output_root=args.out / "native",
            max_steps=protocol["max_control_steps"], camera_width=128, camera_height=128, capture_videos=True)
        summary = holder["summary"]
        summary.update(status="MEASURED", program_ok=outcome.program_ok, native_run=str(outcome.run_dir),
                       program_sha256=digest(args.program), protocol_sha256=digest(args.protocol))
        save(args.out / "result.json", summary)
        print(json.dumps(summary, indent=2), flush=True)
    except Exception:
        save(args.out / "result.json", {"status": "INFRASTRUCTURE_ERROR", "error": traceback.format_exc()})
        traceback.print_exc()
        return 1
    finally:
        save(args.out / "artifacts.json", {str(p.relative_to(args.out)): digest(p) for p in sorted(args.out.rglob('*'))
                                           if p.is_file() and p.name != "artifacts.json"})
    return 0


def audit(root):
    receipt = read(root / "artifacts.json")
    errors = [f"artifact missing/changed: {name}" for name, sha in receipt.items()
              if not (root / name).is_file() or digest(root / name) != sha]
    if errors:
        return {"status": "INVALID_ARTIFACT_BINDING", "errors": errors}
    summary = read(root / "result.json")
    if summary["status"] != "MEASURED":
        return {"status": "UNKNOWN", "errors": [summary.get("error", "unmeasured")]}
    model = mujoco.MjModel.from_binary_path(str(root / "model.mjb"))
    data = mujoco.MjData(model)
    inventory = read(root / "geometry.json")
    protocol_path = root / "protocol.json"
    if not protocol_path.exists():
        protocol_path = root.parent.parent / "protocol.json"
    if digest(protocol_path) != summary['protocol_sha256']:
        return {'status': 'INVALID_PROTOCOL_BINDING', 'errors': ['frozen rules changed']}
    protocol = read(protocol_path)
    class DataView:
        body_xpos = data.xpos
        body_xmat = data.xmat
        qpos = data.qpos
        @property
        def time(self):
            return float(data.time)
        def get_site_xpos(self, name):
            return data.site_xpos[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, name)]
    evaluator = Capture.__new__(Capture)
    evaluator.sim = SimpleNamespace(data=DataView())
    evaluator.nut = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'SquareNut_main')
    evaluator.peg = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'peg1')
    finger = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, 'gripper0_finger_joint1')
    finger_qpos = model.jnt_qposadr[finger]
    evaluator.finger_qpos = int(finger_qpos)
    evaluator.adapter = SimpleNamespace(get_robot_state=lambda: SimpleNamespace(
        gripper_positions={'primary': float(np.clip(data.qpos[finger_qpos]/.04, 0, 1))}))
    evaluator.env = SimpleNamespace(robots=[SimpleNamespace(gripper=SimpleNamespace(
        important_sites={'grip_site': 'gripper0_grip_site'}))])
    evaluator.protocol = protocol
    evaluator.grasped = evaluator.lifted = evaluator.drop = False
    recomputed_frames = []
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    with np.load(root / "physics_states.npz") as states:
        pre, post = states['pre'], states['post']
    expected = summary["control_steps"] * round(.05 / model.opt.timestep)
    if len(pre) != expected or post.shape != pre.shape or len(pre) != summary["physics_steps"]:
        return {"status": "UNKNOWN_MISSING_SAMPLES", "expected": expected, "recorded": len(pre)}
    initial = np.load(root / 'initial_state.npy')
    if (not np.all(np.isfinite(pre)) or not np.all(np.isfinite(post))
            or not np.allclose(pre[:, 0], initial[0] + np.arange(expected)*model.opt.timestep, rtol=0, atol=1e-10)):
        return {'status': 'UNKNOWN_INVALID_STATE_OR_TIME_GRID'}
    max_state, max_force, count, samples = 0., 0., 0, 0
    with gzip.open(root / "physics.jsonl.gz", "rt") as stream:
        for index, line in enumerate(stream):
            row = json.loads(line)
            if index >= len(pre) or row["physics_index"] != index or row["control_index"] != index // 25:
                errors.append("missing/nonuniform sample index")
                break
            if abs(row["time_start_s"] - pre[index, 0]) > 1e-10:
                errors.append("physical timestamp mismatch")
                break
            mujoco.mj_setState(model, data, pre[index], spec)
            mujoco.mj_step(model, data)
            reconstructed = np.empty(len(pre[index]))
            mujoco.mj_getState(model, data, reconstructed, spec)
            max_state = max(max_state, float(np.max(np.abs(reconstructed - post[index]))))
            contacts = contact_rows(model, data, inventory)
            if len(contacts) != len(row['contacts']):
                errors.append('contact count mismatch')
                break
            if [(a['geom1'], a['geom2'], a['pair']) for a in contacts] != [
                    (b['geom1'], b['geom2'], b['pair']) for b in row['contacts']]:
                errors.append('contact identity mismatch')
            if contacts:
                for key in ('wrench_contact_raw', 'position_world_m', 'frame_world_to_contact', 'force_world_n', 'distance_m'):
                    delta = float(np.max(np.abs(np.asarray([a[key] for a in contacts]) -
                                                np.asarray([b[key] for b in row['contacts']]))))
                    if key == 'wrench_contact_raw':
                        max_force = max(max_force, delta)
                    if not np.isfinite(delta) or delta > 1e-8:
                        errors.append(f'contact measurement mismatch: {key}')
                count += len(contacts)
            evaluator.table_z = float(evaluator.sim.data.get_site_xpos('table_top')[2])
            frame = evaluator.task_frame(contacts)
            frame['time_s'] = row['time_start_s']
            for key, value in frame.items():
                recorded = row['task_frame'][key]
                if isinstance(value, (bool, str, list)) and key not in ('nut_position_world_m', 'nut_relative_peg_m'):
                    if value != recorded:
                        errors.append(f'task event mismatch: {key}')
                elif not np.allclose(value, recorded, rtol=0, atol=1e-8):
                    errors.append(f'task measurement mismatch: {key}')
            recomputed_frames.append(frame)
            samples += 1
            if errors:
                break
    if samples != expected or max_state > 1e-9 or max_force > 1e-8:
        errors.append('coverage or reconstruction error')
    tail_count = round(protocol['verification_steps'] / 20 / model.opt.timestep)
    tail = recomputed_frames[-tail_count:]
    independent = len(tail) == tail_count and all(f['independent_task_frame'] for f in tail)
    if tail:
        positions = np.asarray([f['nut_position_world_m'] for f in tail])
        drift = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
        independent &= drift <= protocol['task_stability_drift_m']
    violation_frames = sum(bool(f['unexpected_contacts']) for f in recomputed_frames)
    if (independent != summary['independent_task_success'] or evaluator.drop != summary['drop_observed']
            or violation_frames != summary['unexpected_contact_frames']):
        errors.append('task/safety summary mismatch')
    return {"status": "PASS" if not errors else "INVALID_EVIDENCE", "errors": errors,
            "physics_steps_recomputed": samples, "contacts_recomputed": count,
            "max_state_error": max_state, "max_wrench_error": max_force,
            "independent_task_success_recomputed": bool(independent),
            "drop_recomputed": evaluator.drop, "unexpected_contact_frames_recomputed": violation_frames}


def verify_retest(development, retest):
    """Verify a fresh physical execution against a fully reconstructed source.

    Exact equality of the entire integration-state and contact streams lets the
    reference's force reconstruction cover the duplicate trace without rerunning
    identical native calculations. Different traces require their own audit.
    """
    reference = read(development.parent / f'{development.name}.audit.json')
    if reference.get('status') != 'PASS':
        return {'status': 'UNKNOWN_REFERENCE_NOT_AUDITED'}
    receipt = read(retest / 'artifacts.json')
    for name, sha in receipt.items():
        if not (retest / name).is_file() or digest(retest / name) != sha:
            return {'status': 'INVALID_ARTIFACT_BINDING', 'file': name}
    a, b = read(development / 'result.json'), read(retest / 'result.json')
    if (a['program_sha256'] != b['program_sha256'] or a['protocol_sha256'] != b['protocol_sha256']
            or a['initial_state_id'] != b['initial_state_id']):
        return {'status': 'INITIAL_STATE_OR_VERSION_MISMATCH'}
    for name in ('model.xml', 'initial_state.npy', 'geometry.json', 'actions.json'):
        if digest(development / name) != digest(retest / name):
            return {'status': 'REPLAY_MISMATCH', 'file': name}
    with np.load(development / 'physics_states.npz') as left, np.load(retest / 'physics_states.npz') as right:
        if any(not np.array_equal(left[key], right[key]) for key in ('pre', 'post')):
            return {'status': 'REPLAY_MISMATCH', 'file': 'physics_states.npz'}
    def stream_sha(path):
        sha = hashlib.sha256()
        with gzip.open(path, 'rb') as stream:
            while block := stream.read(1024 * 1024):
                sha.update(block)
        return sha.hexdigest()
    left_sha = stream_sha(development / 'physics.jsonl.gz')
    right_sha = stream_sha(retest / 'physics.jsonl.gz')
    if left_sha != right_sha:
        return {'status': 'REPLAY_MISMATCH', 'file': 'physics.jsonl.gz', 'needs_separate_native_audit': True}
    for key in ('control_steps', 'physics_steps', 'native_task_success', 'independent_task_success',
                'safety_status', 'unexpected_contact_frames', 'grasp_contact_observed', 'lift_observed', 'drop_observed'):
        if a[key] != b[key]:
            return {'status': 'INVALID_SUMMARY', 'key': key}
    return {'status': 'PASS', 'method': 'fresh physical rollout; exact full-state and contact-stream equality to independently reconstructed development evidence',
            'reference_audit': str(development.parent / f'{development.name}.audit.json'),
            'reference_audit_sha256': digest(development.parent / f'{development.name}.audit.json'),
            'contact_stream_sha256': left_sha, 'physics_steps_verified_by_equivalence': b['physics_steps'],
            'contacts_verified_by_equivalence': reference['contacts_recomputed'], 'max_state_delta': 0.0}


def main(args=None):
    from esrb.cli.options import options_tools_diagnose_nut_evidence
    parser = argparse.ArgumentParser()
    options_tools_diagnose_nut_evidence(parser)
    args = parser.parse_args() if args is None else args
    if args.verify_retest:
        result = verify_retest(*args.verify_retest)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    if args.audit:
        result = audit(args.audit)
        print(json.dumps(result, indent=2))
        return 0 if result['status'] == 'PASS' else 1
    return execute(args)
