"""Measured state and contact feedback, converted to seven real joint targets."""
import numpy as np
from esrb.agent.base import BaseAgent
from esrb.tasks.robots.panda import joint_target,yaw_rotation

class PandaContactPolicy(BaseAgent):
    name='panda_contact'
    def __init__(self):
        self.max_speed=.003
        self.push_step_m=.0007
        self.insertion_step_m=.0005
        self.lateral_bias_m=.003
        self.force_target_n=8.
        self.admittance_gain=.00004
        self._guide_engaged=False
        self._last_time=-1.
    def act(self,observation):
        i=observation.info;s=i['stage'];part=np.array(i['part_position']);previous=np.array(i['target_grip_position'])
        if i['time_s']<=self._last_time:self._guide_engaged=False
        self._last_time=i['time_s']
        rotation=np.eye(3);closure=.04
        if i['scenario']=='constrained_push':
            target=np.r_[part[0]-.067,part[1],.26]
            if s=='lower':target[2]=.042
            elif s=='push':
                target=previous.copy();error=i['goal'][0]-part[0]
                target[0]+=np.clip(error,-self.push_step_m,self.push_step_m);target[1]=.10;target[2]=.042
            elif s in {'retract','verify'}:target=np.r_[i['retreat_xy'],.27]
            speed=self.max_speed
        else:
            if s in {'approach','descend','close'}:
                target=np.array(i['pick_grip']);rotation=yaw_rotation(i['part_yaw_rad'])
                if s=='approach':target[2]=.27
                closure=.04 if s=='close' else 0.
            else:
                target=np.array(i['place_grip']);rotation=np.array(i['place_rotation'])
                if s=='lift':target=np.r_[previous[:2],.27];rotation=np.array(i['grip_rotation'])
                if s=='transfer':target[2]=.27
                if s in {'retreat','verify'}:target=np.r_[i['retreat_xy'],.27]
                if s in {'release','retreat','verify'}:closure=0.
                if s=='release':
                    target=np.array(i['release_grip']);rotation=np.array(i['release_rotation'])
                if s=='insert':
                    force=np.array(i['contact_force_world_n'])
                    if i['scenario']=='force_alignment':
                        if np.linalg.norm(force[:2])>.15:self._guide_engaged=True
                        if not self._guide_engaged:
                            target[0]+=self.lateral_bias_m*np.clip((i['goal'][1]-part[1])/.175,0.,1.)
                    target+=np.clip(self.admittance_gain*force,-.001,.001)
                    if np.linalg.norm(force)>self.force_target_n:
                        axis=1 if i['scenario']=='force_alignment' else 0
                        target[axis]=previous[axis]-.0002
            speed=self.insertion_step_m if s in {'lower','insert'} else self.max_speed
        delta=target-previous;length=np.linalg.norm(delta)
        if length>speed:target=previous+speed*delta/length
        return joint_target(i,target,rotation,closure)
    def update(self,episode_data):pass
    def state_dict(self):return {k:float(getattr(self,k)) for k in ('max_speed','push_step_m','insertion_step_m','lateral_bias_m','force_target_n','admittance_gain')}
    def load_state_dict(self,state):
        if set(state)!=set(self.state_dict()):raise ValueError('incompatible Panda contact policy schema')
        for k,v in state.items():setattr(self,k,float(v))
        self._guide_engaged=False;self._last_time=-1.
