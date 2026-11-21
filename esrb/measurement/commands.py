"""Signed insertion commands reconstructed from applied controls and model FK.

Controller stage is context, never evidence of forward motion on its own.
This module never updates the live simulator or trusts a logged intent flag.
"""
import numpy as np
import mujoco


def stage_axis(scene,phase):
    stage=phase.split(':')[-1]
    if scene=='constrained_push' and stage=='push':return np.array([1.,0.,0.])
    if scene=='force_alignment' and stage=='insert':return np.array([0.,1.,0.])
    if scene=='compliant_assembly' and stage=='insert':return np.array([1.,0.,0.])
    if scene in ('long_horizon_assembly','moving_assembly','nut-assembly') and stage=='lower':return np.array([0.,0.,-1.])
    return None


class CommandProjection:
    def __init__(self,model,scene,roles,controller=None):
        self.model=model;self.scene=scene;self.roles=roles;self.controller=controller
        self.scratch=None
        if scene!='nut-assembly' and not roles.get('analytic_fixture'):
            self.scratch=mujoco.MjData(model)
            self.arm=np.array([model.joint(f'joint{i}').qposadr[0] for i in range(1,8)])
            self.site=model.site('grip_origin').id

    def project(self,phase,qpos,ctrl,action,mocap_pos=None,mocap_quat=None):
        axis=stage_axis(self.scene,phase)
        if axis is None or self.roles.get('analytic_fixture'):return 0.
        if self.scene=='nut-assembly':
            if self.controller!='OSC_POSE':raise ValueError('unverified native insertion command controller')
            action=np.asarray(action,dtype=float)
            if action.shape!=(7,) or not np.isfinite(action).all():raise ValueError('invalid native insertion action')
            # The frozen native OSC_POSE config has +/-0.05 m translation scale.
            return float(np.dot(.05*np.clip(action[:3],-1.,1.),axis))
        if len(ctrl)<7:raise ValueError('applied arm targets unavailable for command FK')
        scratch=self.scratch;model=self.model
        scratch.qpos[:]=qpos
        if mocap_pos is not None:scratch.mocap_pos[:]=mocap_pos
        if mocap_quat is not None:scratch.mocap_quat[:]=mocap_quat
        mujoco.mj_kinematics(model,scratch)
        before=scratch.site_xpos[self.site].copy()
        if self.scene in ('long_horizon_assembly','moving_assembly'):
            name=phase.split(':')[0]
            body=model.body(name).id
            parent=self.roles.get('goal_bodies',{}).get(str(body))
            if parent is not None:axis=scratch.xmat[int(parent)].reshape(3,3)@axis
        # PandaRobot.apply clips requested targets. Use its actual actuator
        # controls from the first physical sample, rather than unclipped actions.
        scratch.qpos[self.arm]=np.asarray(ctrl)[:7]
        mujoco.mj_kinematics(model,scratch)
        return float(np.dot(scratch.site_xpos[self.site]-before,axis))
