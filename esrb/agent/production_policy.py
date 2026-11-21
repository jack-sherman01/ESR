"""State feedback only: real Panda joint targets for multi-part workcells."""
import numpy as np

from esrb.agent.base import BaseAgent
from esrb.tasks.robots.panda import joint_target,yaw_rotation


class ProductionPolicy(BaseAgent):
    name='panda_production'
    def __init__(self):
        self.max_speed=.003
        self.grasp_closure=.04
        self.transport_closure=.04
        self.transport_yaw_bias=0.
        self.seat_wait_steps=25.
        self.velocity_compensation=1.
        self.grasp_wait_steps=25.
        self.fixture_lag_s=0.
        self._target_history=[]
        self._history_part=None

    def act(self,observation):
        i=observation.info
        phase=i['phase']; pick=np.array(i['pick_grip']); target=np.array(i['place_grip'])
        velocity=np.array(i['part_velocities'][i['active_part']])
        if self._history_part!=i['active_part'] or self._target_history and i['time_s']<=self._target_history[-1][0]:
            self._target_history=[]; self._history_part=i['active_part']
        self._target_history.append((i['time_s'],np.array(i['destination_position']),np.array(i['destination_rotation'])))
        self._target_history=[row for row in self._target_history if row[0]>=i['time_s']-max(10.,self.fixture_lag_s+1.)]
        place_rotation=np.array(i['place_rotation'])
        if self.fixture_lag_s>0 and phase in {'transfer','lower','release','retreat','verify_object','verify'}:
            older=[row for row in self._target_history if row[0]<=i['time_s']-self.fixture_lag_s]
            _,old_position,old_rotation=older[-1] if older else self._target_history[0]
            delta_rotation=old_rotation@np.array(i['destination_rotation']).T
            target=old_position+delta_rotation@(target-np.array(i['destination_position']))
            place_rotation=delta_rotation@place_rotation
        if phase in {'approach','descend','close'}:
            target=pick+self.velocity_compensation*.04*velocity
            angle=i['pick_yaw']; closure=0. if phase in {'approach','descend'} else self.grasp_closure
            if phase=='approach': target[2]=.36
            if phase=='close' and self.grasp_wait_steps<25 and i['grasp_ticks']>=self.grasp_wait_steps: target[2]=.36
        else:
            angle=i['place_yaw']+self.transport_yaw_bias
            closure=self.transport_closure
            if phase=='lift': target[:2]=i['lift_xy']
            if phase in {'lift','transfer','retreat','verify_object','verify'}: target[2]=.36
            if phase in {'release','retreat','verify_object','verify'}: closure=0.
            if phase=='lower' and self.seat_wait_steps<25 and i['seat_ticks']>=self.seat_wait_steps: closure=0.
        previous=np.array(i['target_grip_position'])
        delta=target-previous
        length=np.linalg.norm(delta)
        if length>self.max_speed: target=previous+delta/length*self.max_speed
        rotation=yaw_rotation(angle) if phase in {'approach','descend','close'} else yaw_rotation(self.transport_yaw_bias) @ place_rotation
        return joint_target(i,target,rotation,closure)

    def update(self,episode_data):
        pass

    def state_dict(self):
        return {key:float(getattr(self,key)) for key in ('max_speed','grasp_closure','transport_closure',
            'transport_yaw_bias','seat_wait_steps','velocity_compensation','grasp_wait_steps','fixture_lag_s')}

    def load_state_dict(self,state):
        if set(state)!=set(self.state_dict()): raise ValueError('incompatible Panda production policy schema')
        for key,value in state.items(): setattr(self,key,float(value))
        self._target_history=[]; self._history_part=None
