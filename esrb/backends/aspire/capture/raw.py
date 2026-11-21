"""Lossless physics records without safety rules, labels, or stop decisions."""
from pathlib import Path
import math

import mujoco
import numpy as np

from esrb.runtime.io import save


class RawPhysicsCapture:
    def __init__(self, env, root, chunk_size=512):
        self.sim, self.root = env.sim, Path(root)
        self.root.mkdir(exist_ok=False)
        self.original_step = self.sim.step
        self.initial_time = float(self.sim.data.time)
        self.dt = float(self.sim.model.opt.timestep)
        self.chunk_size = chunk_size
        self.rows, self.chunks, self.steps = [], [], 0
        self.closed, self.errors = False, []
        self.object_ids = sorted({int(self.sim.model.jnt_bodyid[i])
                                  for i in range(self.sim.model.njnt)
                                  if int(self.sim.model.jnt_type[i]) == 0})
        model = self.sim.model
        inventory = []
        for i in range(model.ngeom):
            body = int(model.geom_bodyid[i]); ancestors = []; cursor = body
            while cursor:
                ancestors.append(model.body_id2name(cursor))
                cursor = int(model.body_parentid[cursor])
            inventory.append({'geom_id': i, 'geom_name': model.geom_id2name(i),
                              'body_id': body, 'body_name': model.body_id2name(body),
                              'ancestors': ancestors, 'type': int(model.geom_type[i]),
                              'size': model.geom_size[i].tolist(),
                              'contype': int(model.geom_contype[i]),
                              'conaffinity': int(model.geom_conaffinity[i])})
        save(self.root / 'geometry.json', inventory)
        save(self.root / 'schema.json', {
            'schema': 'esrb.raw_physics/1', 'physics_timestep_s': self.dt,
            'format': 'numpy savez_compressed; float64; allow_pickle=False',
            'object_body_ids': self.object_ids,
            'object_body_names': [model.body_id2name(i) for i in self.object_ids],
            'contact_columns': ['geom1', 'geom2', 'distance_m', 'position_world_m[3]',
                                'world_to_contact[9]', 'contact_wrench_force_torque[6]'],
            'time_convention': 'qpos/qvel at interval start; contacts, sensors and kinematics '
                               'from mj_step solve, read after integration; t_end is integrated time',
            'sensors': [{'name': model.sensor_id2name(i), 'address': int(model.sensor_adr[i]),
                         'dimension': int(model.sensor_dim[i]), 'type': int(model.sensor_type[i])}
                        for i in range(model.nsensor)],
            'online_assessment': False})
        self.sim.step = self.step

    def step(self, *args, **kwargs):
        start = float(self.sim.data.time)
        qpos = self.sim.data.qpos.copy(); qvel = self.sim.data.qvel.copy()
        ctrl = self.sim.data.ctrl.copy()
        self.original_step(*args, **kwargs)
        end = float(self.sim.data.time)
        if abs(end - start - self.dt) > 1e-9:
            raise ValueError('raw recorder: unexpected physics interval')
        contacts = []
        for i in range(self.sim.data.ncon):
            c = self.sim.data.contact[i]; wrench = np.empty(6)
            mujoco.mj_contactForce(self.sim.model._model, self.sim.data._data, i, wrench)
            contacts.append(np.r_[c.geom1, c.geom2, c.dist, c.pos, c.frame, wrench])
        objects = np.array([np.r_[self.sim.data.body_xpos[i], self.sim.data.body_xquat[i]]
                            for i in self.object_ids], dtype=float).reshape(-1, 7)
        row = {'time': np.array([start, end]), 'qpos_start': qpos, 'qvel_start': qvel,
               'ctrl': ctrl, 'actuator_force': self.sim.data.actuator_force.copy(),
               'sensordata_solve': self.sim.data.sensordata.copy(), 'object_pose_solve': objects,
               'contacts': np.array(contacts, dtype=float).reshape(-1, 21)}
        if any(not np.all(np.isfinite(v)) for v in row.values()):
            raise ValueError('raw recorder: nonfinite physical sample')
        self.rows.append(row); self.steps += 1
        if len(self.rows) >= self.chunk_size:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        lengths = [len(r['contacts']) for r in self.rows]
        arrays = {k: np.stack([r[k] for r in self.rows]) for k in self.rows[0] if k != 'contacts'}
        arrays['contact_offsets'] = np.r_[0, np.cumsum(lengths)].astype(np.int64)
        arrays['contacts'] = np.concatenate([r['contacts'] for r in self.rows], axis=0)
        name = f'chunk_{len(self.chunks):06d}.npz'
        with (self.root / (name + '.tmp')).open('xb') as stream:
            np.savez_compressed(stream, **arrays)
        (self.root / (name + '.tmp')).replace(self.root / name)
        self.chunks.append({'file': name, 'steps': len(self.rows), 'contacts': sum(lengths)})
        self.rows.clear()

    def finish(self):
        if self.closed:
            return self.summary
        self.closed = True; self.sim.step = self.original_step
        self.flush()
        elapsed = (float(self.sim.data.time) - self.initial_time) / self.dt
        expected = round(elapsed) if math.isfinite(elapsed) else -1
        complete = abs(elapsed - expected) < 1e-6 and expected == self.steps and not self.errors
        self.summary = {'schema': 'esrb.raw_capture_summary/1', 'steps': self.steps,
                        'elapsed_physics_steps': expected, 'coverage_complete': complete,
                        'errors': self.errors, 'chunks': self.chunks, 'online_assessment': False}
        save(self.root / 'summary.json', self.summary)
        return self.summary
