"""Host-only Wipe contact and tool-load witness for each MuJoCo mj_step."""
from __future__ import annotations
import gzip
import json
import math
import mujoco
import numpy as np
from esrb.detection.wipe_gate import WipeSafety


class SafetyStop(RuntimeError):
    pass


class WipeSafetyCapture:
    def __init__(self, env, groups, rules, root):
        self.env, self.sim, self.groups, self.rules = env, env.sim, groups, rules
        self.original_step = self.sim.step
        self.initial_time = float(self.sim.data.time)
        self.accumulator = WipeSafety(rules,self.initial_time)
        self.records_written, self.stopped, self.closed = 0, False, False
        if not {'tool','robot_link','table','floor'} <= set(groups):
            raise ValueError('missing Wipe geometry identities')
        # Resolve native load sensors before stepping: absent measurement is fatal.
        self.sensor_ranges = {}
        for key in ['force_ee','torque_ee']:
            name=env.robots[0].gripper.important_sensors[key]
            index=self.sim.model.sensor_name2id(name)
            if int(self.sim.model.sensor_dim[index])!=3:
                raise ValueError('invalid native tool sensor dimension')
            start=int(self.sim.model.sensor_adr[index]); self.sensor_ranges[key]=slice(start,start+3)
        self.stream=gzip.open(root/'safety_physics.jsonl.gz','wt',compresslevel=1)
        self.event_stream=(root/'safety_events.jsonl').open('w')
        self.sim.step=self.step

    def step(self,*args,**kwargs):
        if self.stopped: raise SafetyStop('ESRB Wipe monitor stopped this episode')
        start=float(self.sim.data.time)
        self.original_step(*args,**kwargs)
        end=float(self.sim.data.time)
        try:
            contacts=[]
            for i in range(self.sim.data.ncon):
                c=self.sim.data.contact[i]; first,second=int(c.geom1),int(c.geom2)
                wrench=np.zeros(6)
                mujoco.mj_contactForce(self.sim.model._model,self.sim.data._data,i,wrench)
                frame=np.asarray(c.frame).reshape(3,3)
                if not np.all(np.isfinite(wrench)) or not np.all(np.isfinite(c.pos)) or not np.all(np.isfinite(frame)):
                    raise ValueError('nonfinite MuJoCo contact sample')
                contacts.append({'geom1':first,'geom2':second,'groups':[self.groups[first],self.groups[second]],
                    'normal_force_n':float(wrench[0]),'wrench_contact_raw':wrench.tolist(),
                    'position_world_m':c.pos.tolist(),'frame_world_to_contact':frame.tolist()})
            force=self.sim.data.sensordata[self.sensor_ranges['force_ee']].copy().tolist()
            torque=self.sim.data.sensordata[self.sensor_ranges['torque_ee']].copy().tolist()
            previous=len(self.accumulator.events)
            event=self.accumulator.observe(start,end,contacts,force,torque)
            total=sum(c['normal_force_n'] for c in contacts if set(c['groups'])=={'tool','table'})
            row={'physics_index':self.accumulator.steps-1,'time_start_s':start,'time_end_s':end,
                'contacts':contacts,'tool_table_normal_force_n':total,
                'tool_sensor_force_raw_n':force,'tool_sensor_torque_raw_nm':torque,
                'tool_sensor_frame_world':self.sim.data.get_site_xmat('gripper0_ft_frame').reshape(3,3).tolist()}
            self.stream.write(json.dumps(row,allow_nan=False,separators=(',',':'))+'\n');self.records_written+=1
            for item in self.accumulator.events[previous:]:
                self.event_stream.write(json.dumps(item,allow_nan=False)+'\n')
            if event is not None:
                self.stopped=True;self.stream.flush();self.event_stream.flush()
        except Exception as exc:
            self.accumulator.errors.append(f'{type(exc).__name__}: {exc}')
            self.stopped=True
            raise SafetyStop('ESRB Wipe evidence unavailable') from exc
        if self.stopped:raise SafetyStop('ESRB Wipe forbidden contact')

    def finish(self):
        if self.closed:return self.summary
        self.closed=True;self.sim.step=self.original_step
        elapsed=(float(self.sim.data.time)-self.initial_time)/self.rules['physics_timestep_s']
        expected=round(elapsed) if math.isfinite(elapsed) and abs(elapsed-round(elapsed))<1e-6 else -1
        if expected<0:self.accumulator.errors.append('off-grid elapsed physics time')
        try:self.stream.close();self.event_stream.close()
        except Exception as exc:self.accumulator.errors.append(f'finalization failed: {exc}')
        self.summary=self.accumulator.report(elapsed_physics_steps=expected,records_written=self.records_written)
        self.summary.update(online_stopped=self.stopped,
            time_convention='mj_step solved contacts and native sensors for the interval start; stored after integration',
            sensor_convention='raw native force/torque in ft_frame; includes inertia/gravity; not isolated contact load')
        return self.summary
