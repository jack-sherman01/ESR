"""Free-body constrained pushing and contact-guided planar alignment.

No object pose is assigned after reset. Only a two-axis pusher is actuated;
channel/funnel contact and friction determine the part's final pose.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from esrb.tasks.extensions.evidence import reset_snapshot
import numpy as np

from esrb.agent.base import BaseAgent
from esrb.tasks.base import BaseTask,TaskObservation,TaskResult
from esrb.tasks.reset_domain import validate_disturbance
from esrb.tasks.tool_handling.tool_handling import yaw
from esrb.utils.hashing import hash_dict

DT=.02


class ConstrainedPushTask(BaseTask):
    name='constrained_push'
    MODEL_XML=Path(__file__).parent/'assets/contact_channel.xml'
    MAX_STEPS=650
    reset_reference={'part_mass_kg':.08,'friction':.45,'part_x_m':-.10,'part_y_m':0.,'part_yaw_rad':0.}
    reset_bounds={'part_mass_kg':(.06,.10),'friction':(.35,.55),'part_x_m':(-.102,-.098),
                  'part_y_m':(-.004,.004),'part_yaw_rad':(-.04,.04)}
    boundary_fields=('friction','part_y_m','part_yaw_rad')
    STAGES=('approach','push','verify','done')
    ALIGN=False

    def __init__(self):
        tree=ET.fromstring(self.MODEL_XML.read_text())
        if self.ALIGN:
            tree.find(".//geom[@name='pusher_pad']").set('size','.008 .004 .012')
            # A physical tapered lead-in and narrow final rails rotate/centre the part.
            for name,sign in (('rail_left',1),('rail_right',-1)):
                geom=tree.find(f".//geom[@name='{name}']")
                geom.set('pos',f'.09 {sign*.023} .025')
                geom.set('size','.08 .01 .025')
                ET.SubElement(tree.find('worldbody'),'geom',name=name+'_funnel',type='box',
                    pos=f'-.025 {sign*.033} .025',size='.04 .008 .025',euler=f'0 0 {-sign*.25}',
                    rgba='.3 .5 .7 1')
        xml=ET.tostring(tree,encoding='unicode')
        self.model_digest=hash_dict({'xml':xml})
        self._model=mujoco.MjModel.from_xml_string(xml)
        self._data=mujoco.MjData(self._model)
        self._part=self._model.body('part').id
        self._geom=self._model.geom('part_geom').id
        self._inertia=self._model.body_inertia[self._part].copy()
        self._renderer=None

    @classmethod
    def disturbance_for_seed(cls,seed):
        rng=np.random.default_rng(seed)
        return {key:float(rng.uniform(*bounds)) for key,bounds in cls.reset_bounds.items()}

    @property
    def action_space(self):
        return {'shape':(2,),'low':np.array([-.2,-.08]),'high':np.array([.17,.08])}

    @property
    def observation_space(self):
        return {'shape':(8,),'dtype':np.float64}

    def reset(self,seed=0,*,disturbance=None):
        point=validate_disturbance(self.reset_bounds,self.disturbance_for_seed(seed) if disturbance is None else disturbance)
        self._model.body_mass[self._part]=point['part_mass_kg']
        self._model.body_inertia[self._part]=self._inertia*point['part_mass_kg']/.08
        self._model.geom_friction[:,0]=point['friction']
        mujoco.mj_setConst(self._model,self._data)
        mujoco.mj_resetData(self._model,self._data)
        angle=point['part_yaw_rad']
        self._data.qpos[:2]=[point['part_x_m']-.035,point['part_y_m']]
        self._data.qpos[2:]=[point['part_x_m'],point['part_y_m'],.016,np.cos(angle/2),0,0,np.sin(angle/2)]
        self._data.ctrl[:]=self._data.qpos[:2]
        mujoco.mj_forward(self._model,self._data)
        self._steps=self._stage=self._hold=self._stage_ticks=0
        self._done=False
        self._log={key:[] for key in ('ee_positions','tool_positions','tool_quaternions','contact_forces',
            'contact_torques','wall_forces','pusher_forces','forbidden_collisions','boundary_excess','times_s','stages','commands','contact_samples','physics_samples')}
        self._log.update(model=self.name+'/1',model_digest=self.model_digest,reset_readback=point,
            dt_s=DT,reset_count=1,measurement_frame='world',wrench_origin='part_center',
            stage_events=[{'stage':'approach','step':0,'time_s':0.}],stage_records=[],failure_reason=None,success=False)
        self._enter_stage()
        self._log['measurement_schema']={'frame':'world','wrench_origin':'part_center','physics_dt_s':.002,
            'control_dt_s':.02,'force_unit':'N','torque_unit':'N*m','position_unit':'m',
            'forbidden_contacts':['channel_endstop'],'guide_contacts':['rail_left','rail_right','rail_left_funnel','rail_right_funnel']}
        self._log['applied_reset_state']=reset_snapshot(self._model,self._data)
        return self._observe()

    def _state(self):
        return {'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist()}

    def _enter_stage(self):
        self._log['stage_records'].append({'stage':self.STAGES[self._stage],
            'enter_step':self._steps,'enter_time_s':float(self._data.time),'entry_state':self._state(),
            'exit_step':None,'exit_time_s':None,'exit_condition':None,'failure_reason':None,'status':'active'})

    def _exit_stage(self,condition,failed=False):
        self._log['stage_records'][-1].update(exit_step=self._steps,exit_time_s=float(self._data.time),
            exit_state=self._state(),exit_condition=condition,failure_reason=condition if failed else None,
            status='failed' if failed else 'completed')

    def _observe(self):
        position=self._data.xpos[self._part]
        force=max(row[8] for row in self._log['contact_samples'][-1]) if self._log['contact_samples'] else 0.
        return TaskObservation(np.r_[self._data.qpos[:2],position,yaw(self._data.xquat[self._part]),force,self._stage],
            {'stage':self.STAGES[self._stage],'actuator_targets':self._data.ctrl.tolist(),
             'part_position':position.tolist(),'part_yaw_rad':yaw(self._data.xquat[self._part]),
             'force_n':float(force),'alignment':self.ALIGN,'target_x_m':.09})

    def _contact_readback(self):
        force,torque=np.zeros(3),np.zeros(3)
        wall=0.
        applied=0.
        contacts=[]
        for i in range(self._data.ncon):
            c=self._data.contact[i]
            if self._geom not in (c.geom1,c.geom2):
                continue
            other=c.geom2 if c.geom1==self._geom else c.geom1
            name=self._model.geom(other).name
            wrench=np.zeros(6)
            mujoco.mj_contactForce(self._model,self._data,i,wrench)
            sign=1. if c.geom2==self._geom else -1.
            f=sign*c.frame.reshape(3,3).T@wrench[:3]
            t=sign*c.frame.reshape(3,3).T@wrench[3:]
            # The support normal is not an applied manipulation load.
            if name!='floor':
                force+=f
                torque+=t+np.cross(c.pos-self._data.xpos[self._part],f)
            if name=='pusher_pad':
                applied+=float(wrench[0])
            if name.startswith('rail_'):
                wall+=float(wrench[0])
            contacts.append({'part':name,'normal_force_n':float(wrench[0]),'point_world':c.pos.tolist()})
        return force,torque,wall,contacts,applied

    def _boundary_excess(self):
        # Exact box support along world Y, including roll/pitch after an impact.
        rotation=self._data.geom_xmat[self._geom].reshape(3,3)
        extent=float(np.abs(rotation[1])@self._model.geom_size[self._geom])
        return max(0.,abs(float(self._data.geom_xpos[self._geom,1]))+extent-.06)

    def step(self,action):
        if self._done:
            raise RuntimeError('reset is required after contact episode termination')
        action=np.asarray(action,dtype=float)
        if action.shape!=(2,) or not np.isfinite(action).all():
            raise ValueError('contact action must have two finite coordinates')
        self._data.ctrl[:]=np.clip(action,self.action_space['low'],self.action_space['high'])
        samples=[]
        for _ in range(10):
            mujoco.mj_step(self._model,self._data)
            mujoco.mj_forward(self._model,self._data)
            f,t,wall,contacts,applied=self._contact_readback()
            sample={'time_s':float(self._data.time),'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),
                    'force_world_n':f.tolist(),'torque_world_nm':t.tolist(),'wall_force_n':wall,'pusher_force_n':applied,'contacts':contacts,
                    'boundary_excess_m':self._boundary_excess(), 'stage':self.STAGES[self._stage],
                    'ctrl':self._data.ctrl.tolist(),'qacc_warmstart':self._data.qacc_warmstart.tolist()}
            self._log['physics_samples'].append(sample)
            samples.append([self._data.time,*f,*t,wall,applied])
        array=np.array(samples)
        force=array[np.argmax(np.linalg.norm(array[:,1:4],axis=1)),1:4]
        torque=array[np.argmax(np.linalg.norm(array[:,4:7],axis=1)),4:7]
        position=self._data.xpos[self._part]
        q=self._data.xquat[self._part]
        outside=max(row['boundary_excess_m'] for row in self._log['physics_samples'][-10:])
        for key,value in {'ee_positions':[*self._data.qpos[:2],.018], 'tool_positions':position.tolist(),
            'tool_quaternions':q.tolist(),'contact_forces':force.tolist(),'contact_torques':torque.tolist(),
            'wall_forces':float(array[:,7].max()),'pusher_forces':float(array[:,8].max()),
            'forbidden_collisions':int(any(c['part']=='channel_endstop' and c['normal_force_n']>0
                                          for row in self._log['physics_samples'][-10:] for c in row['contacts'])),
            'boundary_excess':float(outside),
            'times_s':float(self._data.time),'stages':self.STAGES[self._stage],
            'commands':self._data.ctrl.tolist(),'contact_samples':samples}.items():
            self._log[key].append(value)
        self._steps+=1
        self._stage_ticks+=1
        pose_ok=(abs(position[0]-.09)<.004 and abs(position[1])<(.004 if self.ALIGN else .02)
                 and (not self.ALIGN or abs(yaw(q))<.08))
        if self._stage==0:
            reached=np.linalg.norm(force)>.05
        elif self._stage==1:
            reached=pose_ok
        else:
            speed=(np.linalg.norm(np.array(self._log['tool_positions'][-1])-self._log['tool_positions'][-2])/DT
                   if len(self._log['tool_positions'])>1 else np.inf)
            self._hold=self._hold+1 if pose_ok and speed<.01 else 0
            reached=self._hold>=20
        if reached:
            self._exit_stage(('part_contact','part_at_target_pose','stable_target_for_0.4s')[self._stage])
            self._stage+=1
            self._stage_ticks=0
            self._log['stage_events'].append({'stage':self.STAGES[self._stage],'step':self._steps,'time_s':float(self._data.time)})
            if self._stage<3:
                self._enter_stage()
        if self._stage==3:
            self._log['success']=True
            self._done=True
        elif self._steps>=self.MAX_STEPS or self._stage_ticks>=550:
            self._done=True
            self._log['failure_reason']=self.STAGES[self._stage]+'_timeout'
            self._exit_stage(self._log['failure_reason'],failed=True)
        return self._observe(),self._stage/3.,self._done,dict(self._log)

    @staticmethod
    def evaluate(trace):
        try:
            position=np.asarray(trace['tool_positions'])[-20:]
            rotation=np.asarray(trace['tool_quaternions'])[-20:]
            alignment=trace['model']=='force_alignment/1'
            pose=np.all(np.abs(position[:,0]-.09)<.004) and np.all(np.abs(position[:,1])<(.004 if alignment else .02))
            angles=not alignment or all(abs(yaw(q))<.08 for q in rotation)
            speed=np.linalg.norm(np.diff(position,axis=0),axis=1)/DT
            success=bool(len(position)==20 and pose and angles and np.all(speed<.01)
                         and trace['stage_events'][-1]['stage']=='done' and trace['failure_reason'] is None)
            score=float(max(0.,min(1.,(position[-1,0]+.10)/.19)))
            return TaskResult(score,success,trace,{'failure_reason':trace['failure_reason']})
        except (KeyError,IndexError,TypeError,ValueError):
            return TaskResult(0.,False,trace,{'failure_reason':'missing_or_invalid_measurement'})

    def render_frame(self):
        if self._renderer is None:
            self._renderer=mujoco.Renderer(self._model,height=480,width=640)
        self._renderer.update_scene(self._data,camera='overview')
        return self._renderer.render()

    def close(self):
        if self._renderer:
            self._renderer.close()


class ForceAlignmentTask(ConstrainedPushTask):
    name='force_alignment'
    ALIGN=True
    reset_bounds={**ConstrainedPushTask.reset_bounds,'part_y_m':(-.008,.008),'part_yaw_rad':(-.12,.12)}
    reset_reference={**ConstrainedPushTask.reset_reference,'part_y_m':.005,'part_yaw_rad':.08}


class ContactPolicy(BaseAgent):
    name='contact_guidance'

    def __init__(self):
        self.push_step_m=.0007
        self.lateral_bias_m=0.
        self.force_target_n=2.
        self.compliance_gain=.00015

    def act(self,observation):
        info=observation.info
        previous=np.array(info['actuator_targets'])
        target_y=self.lateral_bias_m
        if info['alignment']:
            target_y+=info['part_position'][1]+np.clip(info['part_yaw_rad']*.10,-.012,.012)
        step=self.push_step_m
        if info['alignment'] and info['force_n']>self.force_target_n:
            step=-min(self.push_step_m,self.compliance_gain*(info['force_n']-self.force_target_n))
        target_x=.09-.026
        if info['stage']=='verify':
            step=0.
        if info['alignment'] and info['stage']=='push' and abs(info['part_yaw_rad'])>.02:
            target_x=min(target_x,info['part_position'][0]-.026+.0008)
        return np.array([min(target_x,previous[0]+step), previous[1]+np.clip(target_y-previous[1],-.0005,.0005)])

    def update(self,episode_data):
        pass

    def state_dict(self):
        return {key:getattr(self,key) for key in ('push_step_m','lateral_bias_m','force_target_n','compliance_gain')}

    def load_state_dict(self,state):
        for key in self.state_dict():
            setattr(self,key,state[key])
