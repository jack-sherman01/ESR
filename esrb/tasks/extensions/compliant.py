"""Continuous assembly with actual socket displacement and contact admittance."""
import mujoco
from esrb.tasks.extensions.evidence import reset_snapshot
import numpy as np

from esrb.tasks.base import TaskObservation
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
from esrb.agent.assembly_policy import AssemblyPolicy


class CompliantAssemblyTask(LongHorizonAssemblyTask):
    name='compliant_assembly'
    reset_reference={**LongHorizonAssemblyTask.reset_reference,'socket_dx_m':.002,'socket_dy_m':.001}
    reset_bounds={**LongHorizonAssemblyTask.reset_bounds,'socket_dx_m':(-.003,.003),'socket_dy_m':(-.002,.002)}
    boundary_fields=('socket_dx_m','socket_dy_m','tool_yaw_rad','friction')
    STAGE_BUDGETS=(100,150,220,120,350,120)
    MAX_STEPS=1100

    def __init__(self):
        super().__init__()
        self._socket_geoms=[self._model.geom(name).id for name in
                            ('work_surface','socket_left','socket_right','socket_front','socket_back')]
        self._socket_positions=self._model.geom_pos[self._socket_geoms].copy()

    def reset(self,seed=0,*,disturbance=None):
        # Super reset initializes all phases; moving only fixed geometry does not reset the part.
        super().reset(seed,disturbance=disturbance)
        point=self._log['reset_readback']
        delta=np.array([point['socket_dx_m'],point['socket_dy_m'],0.])
        self._model.geom_pos[self._socket_geoms]=self._socket_positions+delta
        mujoco.mj_forward(self._model,self._data)
        self._log.update(extension_model=self.name+'/1',socket_xy_m=[.10+delta[0],delta[1]],
                         sustained_forces=[],jamming_durations=[])
        self._jam_ticks=0
        self._log['applied_reset_state']=reset_snapshot(self._model,self._data)
        return self._observe()

    def _observe(self):
        obs=super()._observe()
        force=self._log['contact_forces'][-1] if self._log['contact_forces'] else [0.,0.,0.]
        return TaskObservation(obs.state,{**obs.info,'contact_force_world_n':force,
                    'socket_estimate_xy_m':self._log.get('socket_xy_m',[.10,0.])})

    def step(self,action):
        obs,reward,done,_=super().step(action)
        forces=np.linalg.norm(self._log['contact_forces'],axis=1)
        self._log['sustained_forces'].append(float(np.mean(forces[-10:])) if self._log['stages'][-10:]==['insert']*10 else 0.)
        depths=self._log['insertion_depth_m']
        jam=(self._log['stages'][-1]=='insert' and len(depths)>10 and forces[-1]>1.
             and depths[-1]-depths[-10]<.0001)
        self._jam_ticks=self._jam_ticks+1 if jam else 0
        self._log['jamming_durations'].append(self._jam_ticks*.02)
        return obs,reward,done,dict(self._log)


class CompliantAssemblyPolicy(AssemblyPolicy):
    name='compliant_assembly'

    def __init__(self):
        super().__init__()
        self.admittance_gain=.00012
        self.insertion_step_m=.00035
        self.force_target_n=6.

    def act(self,observation):
        action=super().act(observation)
        info=observation.info
        if info['stage']=='insert':
            force=np.asarray(info['contact_force_world_n'])
            previous=np.asarray(info['actuator_targets'])
            # External lateral reaction moves the position target away from the loaded rim.
            correction=self.admittance_gain*force[:2]
            if self.admittance_gain>0:
                correction+=.2*(np.array(info['socket_estimate_xy_m'])-previous[:2])
            action[:2]=previous[:2]+np.clip(correction,-.0001,.0001)
            action[2]=previous[2]-(self.insertion_step_m if info['insertion_depth_m']<.028 else 0.)
            if np.linalg.norm(force)>self.force_target_n:
                action[2]=previous[2]+self.insertion_step_m
        return action

    def state_dict(self):
        return {**super().state_dict(),**{key:getattr(self,key) for key in
                ('admittance_gain','insertion_step_m','force_target_n')}}
