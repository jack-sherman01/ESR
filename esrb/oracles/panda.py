"""Recompute Panda joint witnesses and reject missing robot telemetry."""
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from esrb.oracles.multiconstraint import MultiConstraintOracle, MultiConstraintVerdict
from esrb.oracles.tool_retention import validate_tool_measurements
from esrb.utils.hashing import hash_artifact, hash_dict


def require(condition, message):
    if not condition:
        raise ValueError(message)


@lru_cache(maxsize=2)
def _model_binding(resource_digest):
    path = Path(__file__).parents[1]/'tasks/tool_handling/assets/tool_handling.xml'
    tree = ET.parse(path).getroot()
    resources = {path.name: hash_artifact(path)}
    for mesh in tree.findall('asset/mesh'):
        name = mesh.get('file')
        resources[name] = hash_artifact(path.parent/name)
    require(hash_dict(resources) == resource_digest, 'Panda model resource identity mismatch')
    return mujoco.MjModel.from_xml_path(str(path)), resources


def validate_panda_measurements(trace):
    validate_tool_measurements(trace)
    require(trace['model'] == 'panda_tool_workcell/3', 'unsupported Panda measurement model')
    model, resources = _model_binding(trace['model_resource_digest'])
    require(trace['model_resources'] == resources and trace['model_digest'] == resources['tool_handling.xml'],
            'Panda model resource identity mismatch')
    require(trace['action_schema'] == 'panda-joint-targets-and-grip/1', 'Panda action schema mismatch')
    n = len(trace['times_s'])
    physics = trace['robot_physics']
    require(n > 0 and len(physics) == n and all(len(step) == 10 for step in physics),
            'missing Panda physics substeps')
    commands = np.asarray(trace['commands'], dtype=float)
    require(commands.shape == (n, 8) and np.isfinite(commands).all(), 'invalid Panda joint commands')
    q = np.asarray([[row['qpos'] for row in step] for step in physics], dtype=float)
    v = np.asarray([[row['qvel'] for row in step] for step in physics], dtype=float)
    t = np.asarray([[row['actuator_torque_nm'] for row in step] for step in physics], dtype=float)
    targets = np.asarray([[row['joint_targets'] for row in step] for step in physics], dtype=float)
    for values in [q, v, t, targets]:
        require(values.shape == (n, 10, 7) and np.isfinite(values).all(), 'invalid Panda joint telemetry')
    for name, values in [('arm_joint_positions', q), ('arm_joint_velocities', v), ('arm_joint_torques', t)]:
        require(np.shape(trace[name]) == (n, 7) and np.allclose(trace[name], values[:, -1], atol=1e-10, rtol=0),
                'Panda joint reduction mismatch')
    times = np.asarray([[row['time_s'] for row in step] for step in physics])
    require(np.allclose(times, np.asarray(trace['contact_samples'])[:, :, 0], atol=1e-10, rtol=0),
            'Panda physics timebase mismatch')
    require(np.allclose(targets, commands[:, None, :7], atol=1e-10, rtol=0), 'Panda action binding mismatch')
    limits = model.jnt_range[:7]
    require(np.allclose(trace['arm_joint_limits'], limits, atol=1e-12, rtol=0)
            and np.all(q >= limits[:, 0]-.001) and np.all(q <= limits[:, 1]+.001),
            'Panda joint limit mismatch')
    flags = [int(any(row['contacts'] for row in step)) for step in physics]
    require(trace['robot_collisions'] == flags, 'Panda collision reduction mismatch')
    data = mujoco.MjData(model)
    body = model.body('gripper').id
    for index, joints in enumerate(q[:, -1]):
        data.qpos[:7] = joints
        mujoco.mj_kinematics(model, data)
        require(np.allclose(data.xpos[body], trace['ee_positions'][index], atol=1e-9, rtol=0)
                and abs(float(data.xquat[body] @ np.asarray(trace['gripper_quaternions'][index]))) > 1-1e-9,
                'Panda forward kinematics mismatch')


class PandaToolOracle(MultiConstraintOracle):
    def evaluate(self, episode_data, generation):
        base = super().evaluate(episode_data, generation)
        fields = asdict(base)
        try:
            validate_panda_measurements(episode_data)
        except (KeyError, ValueError, TypeError, IndexError):
            fields['safe'] = False
            fields['status'] = 'VIOLATION' if base.violated_constraints else 'UNKNOWN'
            fields['rules'].append({'metric': 'panda_evidence_integrity', 'test': 'dynamics_safety',
                                    'limit': None, 'status': 'UNKNOWN', 'value': None,
                                    'reason': 'invalid_panda_joint_evidence'})
        return MultiConstraintVerdict(**fields)
