"""Host-only MuJoCo physics-step witness for the NutAssemblySquare gate."""
from __future__ import annotations
import gzip
import json
import math

import mujoco
import numpy as np

from scripts.gates.nut_gate_rules import NutSafety


class SafetyStop(RuntimeError):
    pass


class NutSafetyCapture:
    def __init__(self, env, groups, rules, root):
        self.env, self.sim, self.groups, self.rules = env, env.sim, groups, rules
        self.original_step = self.sim.step
        self.initial_time = float(self.sim.data.time)
        self.accumulator = NutSafety(rules, float(self.sim.data.get_site_xpos('table_top')[2]), self.initial_time)
        self.records_written, self.stopped, self.closed = 0, False, False
        self.stream = gzip.open(root/'safety_physics.jsonl.gz', 'wt', compresslevel=1)
        self.event_stream = (root/'safety_events.jsonl').open('w')
        required = {'square_nut','square_peg','gripper','robot_link','table','floor'}
        if not required <= set(groups):
            raise ValueError(f'missing geometry identities: {required-set(groups)}')
        self.sim.step = self.step

    def step(self, *args, **kwargs):
        if self.stopped:
            raise SafetyStop('ESRB safety monitor already stopped this episode')
        start = float(self.sim.data.time)
        self.original_step(*args, **kwargs)
        end = float(self.sim.data.time)
        try:
            # Each pair keeps its peak normal contact and the original wrench
            # from that same contact. This preserves the zero-load predicate
            # without writing every expected support contact individually.
            pairs = {}
            for i in range(self.sim.data.ncon):
                contact = self.sim.data.contact[i]
                first, second = int(contact.geom1), int(contact.geom2)
                wrench = np.zeros(6)
                mujoco.mj_contactForce(self.sim.model._model,self.sim.data._data,i,wrench)
                if not np.all(np.isfinite(wrench)):
                    raise ValueError('nonfinite MuJoCo contact wrench')
                key = (first,second)
                if key not in pairs or wrench[0] > pairs[key]['normal_force_n']:
                    pairs[key] = {'geom1':first,'geom2':second,'groups':[self.groups[first],self.groups[second]],
                                  'normal_force_n':float(wrench[0]),'wrench_contact_raw':wrench.tolist(),
                                  'position_world_m':contact.pos.tolist(),
                                  'frame_world_to_contact':np.asarray(contact.frame).reshape(3,3).tolist()}
            contacts = list(pairs.values())
            nut_id = self.env.obj_body_id['SquareNut']
            peg_id = self.sim.model.body_name2id('peg1')
            nut = self.sim.data.body_xpos[nut_id].copy()
            peg = self.sim.data.body_xpos[peg_id].copy()
            nr = self.sim.data.body_xmat[nut_id].reshape(3,3)
            pr = self.sim.data.body_xmat[peg_id].reshape(3,3)
            relative = pr.T @ (nut-peg)
            rotation = pr.T @ nr
            yaw = float(np.arctan2(rotation[1,0],rotation[0,0]))
            tilt = float(np.arccos(np.clip(rotation[2,2],-1.,1.)))
            extent = self.rules['peg_halfwidth_m']*(abs(math.cos(yaw))+abs(math.sin(yaw)))
            fits = max(abs(relative[0]),abs(relative[1]))+extent <= self.rules['nut_hole_halfwidth_m']
            bottom = float(nut[2]-self.rules['nut_halfthickness_m'])
            seated = bool(fits and tilt <= self.rules['max_tilt_rad'] and abs(bottom-self.accumulator.table_z) <= self.rules['seat_tolerance_m'])
            previous = len(self.accumulator.events)
            event = self.accumulator.observe(start,end,nut.tolist(),contacts,seated=seated)
            held = any(set(c['groups']) == {'square_nut','gripper'} and c['normal_force_n'] > self.rules['positive_contact_epsilon_n'] for c in contacts)
            stage = ('release_or_verify' if seated and not held else 'insert' if fits and relative[2] < .1 else 'align' if self.accumulator.lifted and np.linalg.norm(relative[:2]) < .08 else 'transport' if self.accumulator.lifted else 'grasp_contact' if held else 'approach')
            peg_contacts = [c for c in contacts if set(c['groups']) == {'square_nut','square_peg'}]
            lateral = [float(np.linalg.norm((np.asarray(c['frame_world_to_contact']).T @ np.asarray(c['wrench_contact_raw'][:3]))[:2])) for c in peg_contacts]
            row = {'physics_index':self.accumulator.steps-1,'time_start_s':start,'time_end_s':end,
                   'nut_xyz_m':nut.tolist(),'peg_xyz_m':peg.tolist(),'nut_relative_peg_m':relative.tolist(),
                   'relative_yaw_rad':yaw,'tilt_rad':tilt,'nut_bottom_world_m':bottom,'geometric_seating':seated,'stage':stage,
                   'nut_peg_lateral_magnitude_sum_n':sum(lateral),
                   'contact_pair_peaks':[[c['geom1'],c['geom2'],c['normal_force_n']] for c in contacts],
                   'nut_peg_contacts':peg_contacts}
            self.stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n')
            self.records_written += 1
            for item in self.accumulator.events[previous:]:
                self.event_stream.write(json.dumps(item,allow_nan=False)+'\n')
            if event is not None:
                self.stopped = True
                self.stream.flush(); self.event_stream.flush()
        except Exception as exc:
            self.accumulator.errors.append(f'{type(exc).__name__}: {exc}')
            self.stopped = True
            raise SafetyStop('ESRB safety evidence unavailable') from exc
        if self.stopped:
            raise SafetyStop(f'ESRB safety violation: {event["kind"]}')

    def finish(self):
        if self.closed:
            return self.summary
        self.closed = True
        self.sim.step = self.original_step
        elapsed = (float(self.sim.data.time)-self.initial_time)/self.rules['physics_timestep_s']
        if not math.isfinite(elapsed) or abs(elapsed-round(elapsed)) > 1e-6:
            self.accumulator.errors.append('physical elapsed time is not on the frozen time grid')
            expected = -1
        else:
            expected = round(elapsed)
        try:
            self.stream.close(); self.event_stream.close()
        except Exception as exc:
            self.accumulator.errors.append(f'cannot finalize safety witness: {exc}')
        self.summary = self.accumulator.report(elapsed_physics_steps=expected,records_written=self.records_written)
        self.summary['online_stopped'] = self.stopped
        self.summary['time_convention'] = 'contact and kinematics at physical interval start; MuJoCo mj_step solve'
        return self.summary
