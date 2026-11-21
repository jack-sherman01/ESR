"""Full Panda dynamics with free workpieces and measured contact-stage predicates."""
from pathlib import Path
import xml.etree.ElementTree as ET
import mujoco
import numpy as np
from esrb.tasks.base import BaseTask,TaskObservation,TaskResult
from esrb.tasks.reset_domain import validate_disturbance
from esrb.tasks.robots.panda import PandaRobot,ROBOT_XML,yaw_rotation
from esrb.tasks.contact_workcells.geometry import make_scene,START,GOAL
from esrb.utils.hashing import hash_dict,hash_artifact
DT=.02

class PandaContactTask(BaseTask):
    name='constrained_push'
    MAX_STEPS=3500
    MODEL_SCHEMA='panda-contact/2'
    reset_reference={'position_jitter_m':0.,'yaw_jitter_rad':0.,'friction':.85}
    reset_bounds={'position_jitter_m':(-.0005,.0005),'yaw_jitter_rad':(-.01,.01),'friction':(.8,.9)}
    boundary_fields=tuple(reset_reference)

    def __init__(self):
        self._xml=make_scene(self.name);self._model=mujoco.MjModel.from_xml_string(self._xml)
        self._data=mujoco.MjData(self._model);self.robot=PandaRobot(self._model,self._data);self._renderer=None
        self._part=self._model.body('workpiece').id
        joint=self._model.joint('workpiece_free').id
        self._part_q=self._model.jnt_qposadr[joint];self._part_v=self._model.jnt_dofadr[joint]
        self._part_geoms={i for i in range(self._model.ngeom) if self._model.geom_bodyid[i]==self._part and self._model.geom_contype[i]}
        paths=[ROBOT_XML,Path(__file__),Path(__file__).with_name('geometry.py'),Path(__file__).with_name('policy.py'),Path(__file__).parents[1]/'robots/panda.py']
        paths += [Path(e.get('file')) for e in ET.fromstring(self._xml).findall('asset/mesh')]
        repo=Path(__file__).parents[3]
        self.resources={str(p.relative_to(repo)):hash_artifact(p) for p in paths}
        self.model_digest=hash_dict({'xml':self._xml,'resources':self.resources})
        self.STAGES=(('approach','lower','push','retract','verify') if self.name=='constrained_push' else
                     ('approach','descend','close','lift','transfer','lower','insert','release','retreat','verify'))

    @property
    def action_space(self):return {'shape':(8,),'low':np.r_[self.robot.limits[:,0],0.],'high':np.r_[self.robot.limits[:,1],.04]}
    @property
    def observation_space(self):return {'shape':(15,),'dtype':np.float64}
    @property
    def stage(self):return self.STAGES[min(self._stage,len(self.STAGES)-1)]

    @classmethod
    def disturbance_for_seed(cls,seed):
        rng=np.random.default_rng(seed)
        return {k:float(rng.uniform(*v)) for k,v in cls.reset_bounds.items()}

    def reset(self,seed=0,*,disturbance=None):
        point=validate_disturbance(self.reset_bounds,self.disturbance_for_seed(seed) if disturbance is None else disturbance)
        mujoco.mj_resetData(self._model,self._data)
        a=self._part_q;self._data.qpos[a:a+3]=START[self.name]+[point['position_jitter_m'],0,0]
        angle=point['yaw_jitter_rad']+(.30 if self.name=='force_alignment' else .025 if self.name=='compliant_assembly' else 0.)
        self._data.qpos[a+3:a+7]=[np.cos(angle/2),0,0,np.sin(angle/2)]
        self._model.geom_friction[:,0]=point['friction'];self.robot.initialize(target=(0.,-.14,.29))
        self._steps=self._stage=self._ticks=self._hold=self._grasp_ticks=self._lost_ticks=0
        self._done=False;self._held_offset=None;self._held_rotation=None;self._lifted=False;self._released=False
        self._retreat_xy=None
        self._release_pose=None
        snapshot=self.state()
        self._log={'scenario':self.name,'model':self.MODEL_SCHEMA+'/'+self.name,'model_digest':self.model_digest,
                   'model_resources':self.resources,'seed':int(seed),'reset_readback':point,'reset_count':1,
                   'applied_reset_state':snapshot,'initial_state_digest':hash_dict(snapshot),'observation_source':'simulator_state',
                   'physics_dt_s':.002,'control_dt_s':DT,'action_schema':'panda-joint-targets-and-grip/2',
                   'stages':[],'times_s':[],'commands':[],'physics_samples':[],'states':[],'metrics':[],
                   'robot_collisions':[],'part_drops':[],'peak_loads_n':[],'jamming_durations':[],
                   'contact_forces':[],'wall_forces':[],'pusher_forces':[],'tool_positions':[],'tool_quaternions':[],
                   'ee_positions':[],'guide_contact_count':0,'stage_events':[{'stage':self.stage,'step':0,'time_s':0.}],
                   'stage_records':[],'success':False,'failure_reason':None}
        self._entry=snapshot;self._entry_step=0;self._force=np.zeros(3);self._jam=0
        return self.observe()

    def state(self):return {'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),'ctrl':self._data.ctrl.tolist(),
                            'qacc_warmstart':self._data.qacc_warmstart.tolist(),'time_s':float(self._data.time)}

    def part_velocity(self):
        value=np.zeros(6);mujoco.mj_objectVelocity(self._model,self._data,mujoco.mjtObj.mjOBJ_BODY,self._part,value,0)
        value[3:]+=np.cross(value[:3],self._data.xpos[self._part]-self._data.xipos[self._part]);return value

    def metrics(self):
        p=self._data.xpos[self._part];R=self._data.xmat[self._part].reshape(3,3)
        e=p-GOAL[self.name];v=self.part_velocity();yaw=float(np.arctan2(R[1,0],R[0,0]));tilt=float(np.arccos(np.clip(R[2,2],-1,1)))
        pose=bool(np.linalg.norm(e[:2])<.004 and abs(e[2])<.005 and abs(yaw)<(.12 if self.name=='constrained_push' else .08) and tilt<.08)
        return {'position_error_m':e.tolist(),'yaw_error_rad':yaw,'tilt_rad':tilt,'speed_ms':float(np.linalg.norm(v[3:])),
                'spin_rads':float(np.linalg.norm(v[:3])),'pose_ok':pose,'insertion_depth_m':float(max(0.,p[0]+.07-.14)) if self.name=='compliant_assembly' else 0.}

    def observe(self):
        info=self.robot.information();p=self._data.xpos[self._part];R=self._data.xmat[self._part].reshape(3,3)
        angle=float(np.arctan2(R[1,0],R[0,0]));pick=self._data.site('part_grasp').xpos.copy()
        goal=GOAL[self.name].copy();pre=goal.copy()
        if self.name=='force_alignment':pre[1]-=.175;pre[0]+=.003
        elif self.name=='compliant_assembly':pre[0]-=.085
        placement=pre if self.stage in {'transfer','lower'} else goal
        offset=np.array([0,0,.04]) if self._held_offset is None else self._held_offset
        rotation=np.eye(3) if self._held_rotation is None else self._held_rotation
        if self._held_offset is not None and not self._released:
            # Correct real workpiece rotation and slip through joint actions.
            rotation=R.T@np.array(info['grip_rotation'])
            offset=R.T@(np.array(info['grip_position'])-p)
        info.update(stage=self.stage,ticks=self._ticks,time_s=float(self._data.time),scenario=self.name,
                    part_position=p.tolist(),part_rotation=R.tolist(),part_yaw_rad=angle,pick_grip=pick.tolist(),
                    goal=goal.tolist(),place_grip=(placement+offset).tolist(),place_rotation=rotation.tolist(),
                    contact_force_world_n=self._force.tolist(),metrics=self.metrics(),grasp_ticks=self._grasp_ticks,
                    retreat_xy=None if self._retreat_xy is None else self._retreat_xy.tolist())
        if self._release_pose is not None:
            info['release_grip']=self._release_pose[0].tolist()
            info['release_rotation']=self._release_pose[1].tolist()
        return TaskObservation(np.r_[self._data.qpos[self.robot.q],p,self._data.xquat[self._part],float(self._stage)],info)

    def contacts(self):
        rows=[];grip=np.zeros(2);load=0.;bad=0;force=np.zeros(3);guide=0;push=0.
        for i in range(self._data.ncon):
            c=self._data.contact[i];a,b=int(c.geom1),int(c.geom2)
            if a not in self._part_geoms and b not in self._part_geoms and a not in self.robot.geoms and b not in self.robot.geoms:continue
            w=np.zeros(6);mujoco.mj_contactForce(self._model,self._data,i,w)
            world=c.frame.reshape(3,3).T@w[:3]
            part=a in self._part_geoms or b in self._part_geoms
            other=b if a in self._part_geoms else a
            oname=self._model.geom(other).name
            if part and other in self.robot.pads:
                label='push' if self.name=='constrained_push' else 'grasp'
                for j,group in enumerate(self.robot.pad_groups):
                    if other in group:grip[j]+=max(0.,w[0])
                push+=max(0.,w[0])
            elif a in self.robot.geoms or b in self.robot.geoms:
                label='robot_collision';bad+=int(w[0]>.01)
            elif part and (oname.startswith('guide') or oname=='datum_stop'):
                label='fixture_contact';guide+=int(w[0]>.05);load=max(load,float(np.linalg.norm(w[:3])))
                force+=(world if b in self._part_geoms else -world)
            else:label='support'
            rows.append({'geom1':self._model.geom(a).name,'geom2':self._model.geom(b).name,
                         'classification':label,'position_world_m':c.pos.tolist(),
                         'normal_force_n':float(w[0]),'force_world_n':world.tolist()})
        return rows,grip,load,bad,force,guide,push

    def transition(self,condition,failure=False):
        state=self.state()
        self._log['stage_records'].append({'stage':self.stage,'enter_step':self._entry_step,'exit_step':self._steps,
                                           'entry_state':self._entry,'exit_state':state,'exit_condition':condition,
                                           'status':'failed' if failure else 'completed'})
        if failure:self._done=True;self._log['failure_reason']=condition;return
        if self._stage==len(self.STAGES)-1:self._done=True;self._log['success']=True;return
        self._stage+=1;self._ticks=self._hold=0;self._entry=state;self._entry_step=self._steps
        self._log['stage_events'].append({'stage':self.stage,'step':self._steps,'time_s':float(self._data.time)})
        if self.stage in {'retract','retreat'}:self._retreat_xy=self._data.site_xpos[self.robot.site,:2].copy()
        if self.stage=='release':
            self._release_pose=(self._data.site_xpos[self.robot.site].copy(),self._data.site_xmat[self.robot.site].reshape(3,3).copy())

    def advance(self,grip):
        info=self.observe().info;tcp=np.array(info['grip_position']);metric=info['metrics'];p=np.array(info['part_position'])
        if self.name=='constrained_push':
            if self.stage=='approach':ok=np.linalg.norm(tcp-np.r_[p[0]-.067,p[1],.26])<.006
            elif self.stage=='lower':ok=abs(tcp[2]-.042)<.004
            elif self.stage=='push':ok=abs(p[0]-GOAL[self.name][0])<.003 and abs(p[1]-.10)<.015
            elif self.stage=='retract':ok=tcp[2]>.25
            else:
                good=metric['pose_ok'] and metric['speed_ms']<.01 and metric['spin_rads']<.08
                self._hold=self._hold+1 if good else 0;ok=self._hold>=25
            if ok:self.transition('measured_'+self.stage)
        else:
            if self.stage=='approach':ok=np.linalg.norm(tcp-np.r_[info['pick_grip'][:2],.27])<.006
            elif self.stage=='descend':ok=np.linalg.norm(tcp-info['pick_grip'])<.004
            elif self.stage=='close':
                self._grasp_ticks=self._grasp_ticks+1 if min(grip)>.10 else 0;ok=self._grasp_ticks>=20
                if ok:
                    R=np.array(info['part_rotation']);self._held_offset=R.T@(tcp-p)
                    self._held_rotation=R.T@np.array(info['grip_rotation'])
            elif self.stage=='lift':
                self._lifted=self._lifted or p[2]>START[self.name][2]+.035
                ok=tcp[2]>.255 and self._lifted
            elif self.stage=='transfer':ok=np.linalg.norm(tcp[:2]-np.array(info['place_grip'])[:2])<.004 and tcp[2]>.25
            elif self.stage=='lower':ok=np.linalg.norm(tcp-info['place_grip'])<.005
            elif self.stage=='insert':
                good=metric['pose_ok'];self._hold=self._hold+1 if good else 0;ok=self._hold>=20
                if ok:self._released=True
            elif self.stage=='release':ok=self._data.ctrl[7]>250 and max(grip)<.1 and self._ticks>=70
            elif self.stage=='retreat':ok=tcp[2]>.255
            else:
                good=metric['pose_ok'] and metric['speed_ms']<.01 and metric['spin_rads']<.08
                if self.name=='force_alignment':good=good and self._log['guide_contact_count']>0
                self._hold=self._hold+1 if good else 0;ok=self._hold>=25
            if ok:self.transition('measured_'+self.stage)
            if self._lifted and not self._released and self._held_offset is not None:
                expected=tcp-np.array(info['grip_rotation'])@self._held_rotation.T@self._held_offset
                lost=min(grip)<.05 and expected[2]-p[2]>.035 and self._data.qvel[self._part_v+2]<-.05
                self._lost_ticks=self._lost_ticks+1 if lost else 0
                if self._lost_ticks>=5:self._log['part_drops'][-1]=1;self.transition('part_dropped',True)
        if not self._done and (self._steps>=self.MAX_STEPS or self._ticks>=1000):self.transition(self.stage+'_timeout',True)

    def step(self,action):
        if self._done:raise RuntimeError('reset required after task termination')
        self.robot.apply(action);phase=self.stage;peak=bad=guides=push=0.;grip=np.zeros(2)
        for _ in range(10):
            mujoco.mj_step(self._model,self._data);mujoco.mj_forward(self._model,self._data)
            contacts,grip,load,collision,self._force,guide,applied=self.contacts()
            peak=max(peak,load);bad=max(bad,collision);guides+=guide;push=max(push,applied)
            self._log['physics_samples'].append({**self.state(),'stage':phase,'contacts':contacts})
        self._steps+=1;self._ticks+=1;metric=self.metrics();self._log['guide_contact_count']+=guides
        stalled=phase=='insert' and peak>1. and metric['speed_ms']<.001 and not metric['pose_ok']
        self._jam=self._jam+1 if stalled else 0
        for key,value in {'states':self.state(),'metrics':metric,'stages':phase,'times_s':float(self._data.time),
            'commands':self._data.ctrl.tolist(),'robot_collisions':int(bad),'part_drops':0,'peak_loads_n':peak,
            'jamming_durations':self._jam*DT,'contact_forces':self._force.tolist(),'wall_forces':peak,'pusher_forces':push,
            'tool_positions':self._data.xpos[self._part].tolist(),'tool_quaternions':self._data.xquat[self._part].tolist(),
            'ee_positions':self._data.site_xpos[self.robot.site].tolist()}.items():self._log[key].append(value)
        self.advance(grip);self._log['final_metrics']=self.metrics();self._log['control_step_count']=self._steps
        return self.observe(),float(metric['pose_ok']),self._done,dict(self._log)

    @staticmethod
    def evaluate(trace):
        history=trace.get('metrics',[])
        stable=len(history)>=25 and all(m['pose_ok'] and m['speed_ms']<.01 and m['spin_rads']<.08 for m in history[-25:])
        released=len(trace.get('stages',[]))>=25 and trace['stages'][-25:]==['verify']*25
        success=bool(trace.get('success') and stable and released)
        if trace.get('scenario')=='force_alignment':success=success and trace.get('guide_contact_count',0)>0
        return TaskResult(float(history[-1]['pose_ok']) if history else 0.,success,trace,{'model':trace.get('model')})

    def render_frame(self,width=640,height=480,*,overview=False):
        if self._renderer is None:self._renderer=mujoco.Renderer(self._model,width=width,height=height)
        camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE
        camera.lookat[:]=[.01,.03,.12];camera.distance=1.7 if overview else 1.05
        camera.azimuth=135 if overview else 55;camera.elevation=-30 if overview else -40
        self._last_view={'type':'free','lookat_m':camera.lookat.tolist(),'distance_m':float(camera.distance),
                         'azimuth_deg':float(camera.azimuth),'elevation_deg':float(camera.elevation)}
        opt=mujoco.MjvOption();opt.geomgroup[3]=0
        self._renderer.update_scene(self._data,camera=camera,scene_option=opt)
        from esrb.tasks.tool_handling.appearance import decorate_scene
        decorate_scene(self._renderer.scene)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None:self._renderer.close();self._renderer=None
