"""Continuous, friction-only Panda assembly and conveyor episodes."""
import hashlib
from pathlib import Path

import mujoco
import numpy as np

from esrb.tasks.base import BaseTask, TaskObservation, TaskResult
from esrb.tasks.reset_domain import validate_disturbance
from esrb.tasks.robots.panda import PandaRobot, ROBOT_XML, yaw_rotation
from esrb.tasks.production.geometry import make_scene, PARTS, GRASP, GRASP_YAW, HALF_HEIGHT, ASSEMBLED_Z
from esrb.utils.hashing import hash_dict, hash_artifact

DT = .02
PHYSICS_DT = .002
PHASES = ('approach','descend','close','lift','transfer','lower','release','retreat','verify_object')
POSITION_TOLERANCE = .004
ANGLE_TOLERANCE = .08
HOLD_TIME = .5


def yaw(q):
    w,x,y,z=q
    return float(np.arctan2(2*(w*z+x*y),1-2*(y*y+z*z)))


def wrap(a):
    return float(np.arctan2(np.sin(a),np.cos(a)))


class ProductionWorkcellTask(BaseTask):
    mode = 'assembly'
    MAX_STEPS = 6000
    reset_reference = {'position_jitter_m':0., 'yaw_jitter_rad':0., 'friction':.9,'speed_scale':1.}
    reset_bounds = {'position_jitter_m':(-.001,.001),'yaw_jitter_rad':(-.01,.01),
                    'friction':(.85,.95),'speed_scale':(.95,1.05)}
    boundary_fields = tuple(reset_reference)
    MODEL_SCHEMA = 'panda-production/2'

    def __init__(self):
        xml,self._initial_positions,self._destinations=make_scene(self.mode)
        self._xml=xml
        self._model=mujoco.MjModel.from_xml_string(xml)
        self._data=mujoco.MjData(self._model)
        self.robot=PandaRobot(self._model,self._data)
        self._renderer=None
        self._part_ids={name:self._model.body(name).id for name in PARTS}
        self._body_parts={value:key for key,value in self._part_ids.items()}
        self._part_q={name:self._model.jnt_qposadr[self._model.joint(name+'_free').id] for name in PARTS}
        self._part_v={name:self._model.jnt_dofadr[self._model.joint(name+'_free').id] for name in PARTS}
        self._belt=self._model.geom('conveyor_surface').id if self.mode!='assembly' else None
        self._resources=self.resources()
        self.model_digest=hash_dict({'xml':xml,'resources':self._resources})

    def resources(self):
        paths=[ROBOT_XML,Path(__file__),Path(__file__).with_name('geometry.py'),
               Path(__file__).parents[1]/'robots/panda.py']
        import xml.etree.ElementTree as ET
        paths += [Path(e.get('file')) for e in ET.fromstring(self._xml).findall('asset/mesh')]
        return {str(p.relative_to(Path(__file__).parents[3])):hash_artifact(p) for p in paths}

    @property
    def action_space(self):
        return {'shape':(8,),'low':np.r_[self.robot.limits[:,0],0.],
                'high':np.r_[self.robot.limits[:,1],.04]}

    @property
    def observation_space(self):
        return {'shape':(35,),'dtype':np.float64}

    def _pose(self,name):
        i=self._part_ids[name]
        return self._data.xpos[i].copy(),self._data.xquat[i].copy()

    def _grasp_position(self,name):
        return self._data.site_xpos[self._model.site(name+'_grasp').id].copy()

    @classmethod
    def disturbance_for_seed(cls,seed):
        rng=np.random.default_rng(seed)
        return {k:float(rng.uniform(*b)) for k,b in cls.reset_bounds.items()}

    def reset(self,seed=0,*,disturbance=None):
        point=(self.disturbance_for_seed(seed) if disturbance is None
               else validate_disturbance(self.reset_bounds,disturbance))
        mujoco.mj_resetData(self._model,self._data)
        for name,pos in self._initial_positions.items():
            a=self._part_q[name]
            self._data.qpos[a:a+3]=pos+[point['position_jitter_m'],0.,0.]
            angle=point['yaw_jitter_rad']
            self._data.qpos[a+3:a+7]=[np.cos(angle/2),0.,0.,np.sin(angle/2)]
        self._model.geom_friction[:,0]=point['friction']
        self._speed=(.04 if self.mode=='transport' else .005)*point['speed_scale'] if self._belt is not None else 0.
        if self._belt is not None: self._model.geom_surfacevel[self._belt]=[self._speed,0,0,0,0,0]
        self.robot.initialize()
        self._order=PARTS if self.mode!='moving_assembly' else PARTS[1:]
        self._op,self._phase,self._ticks,self._steps=0,0,0,0
        self._held_offset=None; self._lift_xy=None
        self._grasp_ticks=self._seat_ticks=self._verify_ticks=0
        self._lifted={name:False for name in PARTS}
        self._released=set()
        self._lost_ticks=0
        self._pending_drop=None
        self._done=False
        snapshot={'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),'ctrl':self._data.ctrl.tolist()}
        self._log={'scenario':self.name,'model':self.MODEL_SCHEMA+'/'+self.mode,'model_digest':self.model_digest,
                   'model_resources':self._resources,'reset_count':1,'seed':int(seed),'reset_readback':point,
                   'applied_reset_state':snapshot,'initial_state_digest':hash_dict(snapshot),
                   'observation_source':'simulator_state','action_schema':'panda-joint-targets-and-grip/2',
                   'control_dt_s':DT,'physics_dt_s':PHYSICS_DT,'measurement_frame':'world',
                   'wrench_origin':'contact_point','contact_force_source':'mj_contactForce after mj_forward',
                   'assembly_protocol':{'position_tolerance_m':POSITION_TOLERANCE,'angle_tolerance_rad':ANGLE_TOLERANCE,
                                        'stable_hold_s':HOLD_TIME,'relative_speed_ms':.008,
                                        'relative_spin_rads':.05,'pose_error_frame':'mating_frame'},
                   'belt_speed_ms':self._speed,'commands':[],'times_s':[],'stages':[],'physics_samples':[],
                   'stage_events':[{'stage':self.stage,'step':0,'time_s':0.}], 'stage_records':[],
                   'robot_collisions':[],'drop_events':[],'part_drops':[],'peak_loads_n':[],'jamming_durations':[],
                   'assembly_metrics':[],'success':False,'failure_reason':None,'control_step_count':0}
        self._entry_state=snapshot
        self._entry_step=0
        self._jam_ticks=0
        return self._observe()

    @property
    def active_part(self): return self._order[min(self._op,len(self._order)-1)]

    @property
    def phase(self): return PHASES[self._phase] if self._op<len(self._order) else 'verify'

    @property
    def stage(self): return self.active_part+':'+self.phase if self._op<len(self._order) else 'verify'

    def target(self,name):
        if self.mode=='transport': return self._destinations[name].copy(),np.eye(3)
        if name=='base':
            if self.mode=='assembly': return np.array([0.,.17,.02+ASSEMBLED_Z[name]]),np.eye(3)
            body=self._model.body('pallet').id
            rotation=self._data.xmat[body].reshape(3,3)
            return self._data.xpos[body]+rotation@np.array([0.,0.,.007+HALF_HEIGHT[name]]),rotation
        base=self._part_ids['base']
        rotation=self._data.xmat[base].reshape(3,3)
        return self._data.xpos[base]+rotation@np.array([0.,0.,ASSEMBLED_Z[name]-ASSEMBLED_Z['base']]),rotation

    def body_velocity(self,body):
        value=np.zeros(6)
        mujoco.mj_objectVelocity(self._model,self._data,mujoco.mjtObj.mjOBJ_BODY,body,value,0)
        # MuJoCo reports BODY linear velocity at its inertial COM. Our poses
        # and mating offsets use the body origin, which can differ for a fork.
        value[3:]+=np.cross(value[:3],self._data.xpos[body]-self._data.xipos[body])
        return value

    def metrics(self):
        result={}
        for name in PARTS:
            p,q=self._pose(name); goal,rotation=self.target(name)
            velocity=self.body_velocity(self._part_ids[name])
            reference=(self._part_ids['base'] if self.mode!='transport' and name!='base' else
                       self._model.body('pallet').id if self.mode=='moving_assembly' else None)
            if reference is not None:
                frame_velocity=self.body_velocity(reference)
                velocity=velocity.copy()
                velocity[3:]-=frame_velocity[3:]+np.cross(frame_velocity[:3],p-self._data.xpos[reference])
                velocity[:3]-=frame_velocity[:3]
            error=rotation.T@(p-goal)
            relative_rotation=rotation.T@self._data.xmat[self._part_ids[name]].reshape(3,3)
            angle_error=float(np.arctan2(relative_rotation[1,0],relative_rotation[0,0]))
            tilt=float(np.arccos(np.clip(relative_rotation[2,2],-1.,1.)))
            seated=bool(np.linalg.norm(error[:2])<=POSITION_TOLERANCE and abs(error[2])<=POSITION_TOLERANCE
                        and abs(angle_error)<=ANGLE_TOLERANCE and tilt<=ANGLE_TOLERANCE)
            required=.012 if self.mode=='transport' or name=='base' else {'connector':.038,'cap':.064}[name]
            result[name]={'position_error_m':error.tolist(),'yaw_error_rad':angle_error,'tilt_rad':tilt,
                          'depth_m':float(max(0.,required-error[2])),'required_depth_m':required,
                          'relative_speed_ms':float(np.linalg.norm(velocity[3:])),
                          'relative_spin_rads':float(np.linalg.norm(velocity[:3])),'seated':seated}
        return result

    def _observe(self):
        name=self.active_part
        p,q=self._pose(name)
        position=self._grasp_position(name)
        goal,desired_rotation=self.target(name)
        angle=float(np.arctan2(desired_rotation[1,0],desired_rotation[0,0]))
        rotation=desired_rotation@yaw_rotation(GRASP_YAW[name])
        offset=self._held_offset if self._held_offset is not None else -yaw_rotation(GRASP_YAW[name]).T @ GRASP[name]
        target_grip=goal-rotation @ offset
        if self._held_offset is not None:
            current_rotation=self._data.xmat[self._part_ids[name]].reshape(3,3)
            grip_rotation=self._data.site_xmat[self.robot.site].reshape(3,3)
            rotation=desired_rotation @ current_rotation.T @ grip_rotation
            target_grip=goal+rotation @ grip_rotation.T @ (self._data.site_xpos[self.robot.site]-p)
        velocities={n:self.body_velocity(self._part_ids[n])[3:].tolist() for n in PARTS}
        info={**self.robot.information(),'stage':self.stage,'phase':self.phase,'phase_ticks':self._ticks,
              'active_part':name,'time_s':float(self._data.time),'seat_ticks':self._seat_ticks,'grasp_ticks':self._grasp_ticks,
              'part_poses':{n:[*self._pose(n)[0],*self._pose(n)[1]] for n in PARTS},
              'part_velocities':velocities,'pick_grip':position.tolist(),'pick_yaw':yaw(q)+GRASP_YAW[name],
              'place_grip':target_grip.tolist(),'place_yaw':angle+GRASP_YAW[name], 'place_rotation':rotation.tolist(),
              'destination_position':goal.tolist(),'destination_rotation':desired_rotation.tolist(),
              'lift_xy':self._lift_xy.tolist() if self._lift_xy is not None else position[:2].tolist(),
              'belt_speed_ms':self._speed,'metrics':self.metrics()}
        state=np.r_[self._data.qpos[self.robot.q],self._data.qvel[self.robot.v],
                    np.concatenate([np.r_[*self._pose(n)] for n in PARTS])]
        return TaskObservation(state,info)

    def _contact_readback(self):
        contacts=[]; bilateral=[0.,0.]; unexpected=[]; load=0.
        for i in range(self._data.ncon):
            c=self._data.contact[i]
            if c.dist>0 or c.efc_address<0: continue
            w=np.zeros(6); mujoco.mj_contactForce(self._model,self._data,i,w)
            if w[0]<.01: continue
            a,b=int(c.geom1),int(c.geom2)
            ba,bb=int(self._model.geom_bodyid[a]),int(self._model.geom_bodyid[b])
            pa,pb=self._body_parts.get(ba),self._body_parts.get(bb)
            classification='support'
            if a in self.robot.geoms or b in self.robot.geoms:
                robot_geom=a if a in self.robot.geoms else b
                part=pb if robot_geom==a else pa
                if robot_geom in self.robot.pads and part==self.active_part:
                    classification='grasp'
                    for side,group in enumerate(self.robot.pad_groups):
                        if robot_geom in group: bilateral[side]+=float(w[0])
                else: classification='robot_collision'
            elif pa and pb: classification='assembly_contact'
            if classification in {'assembly_contact','support'} and self.active_part in (pa,pb):
                load=max(load,float(np.linalg.norm(w[:3])))
            row={'geom1':self._model.geom(a).name,'geom2':self._model.geom(b).name,'geom_ids':[a,b],
                 'classification':classification,'position_world_m':c.pos.tolist(),
                 'force_contact_n':w[:3].tolist(),'force_world_n':(c.frame.reshape(3,3).T@w[:3]).tolist()}
            contacts.append(row)
            if classification=='robot_collision': unexpected.append(row)
        return contacts,bilateral,unexpected,load

    def _transition(self, condition, *, failure=None):
        state={'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),'ctrl':self._data.ctrl.tolist()}
        self._log['stage_records'].append({'stage':self.stage,'enter_step':self._entry_step,'exit_step':self._steps,
            'enter_time_s':self._entry_step*DT,'exit_time_s':self._steps*DT,
            'entry_state':self._entry_state,'exit_state':state,'exit_condition':condition,
            'status':'failed' if failure else 'completed','failure_reason':failure})
        if failure:
            self._done=True; self._log['failure_reason']=failure
            return
        if self._op>=len(self._order):
            self._done=True; self._log['success']=True
            return
        if self._phase==len(PHASES)-1:
            self._op+=1; self._phase=0; self._held_offset=None; self._lift_xy=None
        else: self._phase+=1
        self._ticks=self._grasp_ticks=self._seat_ticks=self._verify_ticks=0
        self._lost_ticks=0
        self._entry_state=state; self._entry_step=self._steps
        self._log['stage_events'].append({'stage':self.stage,'step':self._steps,'time_s':float(self._data.time)})

    def _advance(self,bilateral):
        name,phase=self.active_part,self.phase
        info=self._observe().info
        tcp=np.asarray(info['grip_position']); pose=np.asarray(info['pick_grip'])
        target=np.asarray(info['place_grip'])
        grip=min(bilateral)>.15
        metric=info['metrics'][name]
        if phase in {'lift','transfer','lower'} and self._pose(name)[0][2]>self._initial_positions[name][2]+.045:
            self._lifted[name]=True
        if self._pending_drop is not None:
            if self._data.time-self._pending_drop>=.5:
                self._transition('part_dropped',failure='part_dropped')
            return
        if phase=='approach':
            if self.mode=='transport' and self._pose(name)[0][0]<-.25: return
            if np.linalg.norm(tcp[:2]-pose[:2])<.006 and abs(tcp[2]-.36)<.006: self._transition('above_part')
        elif phase=='descend':
            if np.linalg.norm(tcp-pose)<.004: self._transition('at_grasp_pose')
        elif phase=='close':
            self._grasp_ticks=self._grasp_ticks+1 if grip else 0
            if self._grasp_ticks>=25:
                rotation=np.asarray(info['grip_rotation'])
                self._held_offset=rotation.T@(self._pose(name)[0]-tcp)
                self._lift_xy=tcp[:2].copy()
                self._transition('bilateral_grasp_stable')
        elif phase=='lift':
            if tcp[2]>.35 and self._pose(name)[0][2]>self._initial_positions[name][2]+.045:
                self._lifted[name]=True; self._transition('part_lifted')
        elif phase=='transfer':
            if np.linalg.norm(tcp[:2]-target[:2])<.004 and abs(tcp[2]-.36)<.006: self._transition('over_destination')
        elif phase=='lower':
            self._seat_ticks=self._seat_ticks+1 if metric['seated'] else 0
            if self._seat_ticks>=25: self._released.add(name); self._transition('measured_seating')
        elif phase=='release':
            if max(bilateral)<.1 and self._data.ctrl[7]>250 and self._ticks>=70: self._transition('gripper_open')
        elif phase=='retreat':
            if tcp[2]>.35: self._transition('gripper_withdrawn')
        elif phase=='verify_object':
            good=metric['seated'] and metric['relative_speed_ms']<.008 and metric['relative_spin_rads']<.05
            self._verify_ticks=self._verify_ticks+1 if good else 0
            if self._verify_ticks>=25: self._transition('released_part_stable_0.5s')
        elif phase=='verify':
            good=all(v['seated'] and v['relative_speed_ms']<.008 and v['relative_spin_rads']<.05 for v in info['metrics'].values())
            if self.mode=='moving_assembly': good=good and self.continuous_motion(self._log)
            self._verify_ticks=self._verify_ticks+1 if good else 0
            if self._verify_ticks>=25: self._transition('all_parts_stable_0.5s')
        if not self._done and self._lifted[name] and name not in self._released and phase in {'lift','transfer','lower'}:
            expected=tcp+np.asarray(info['grip_rotation'])@self._held_offset
            # A blocked part losing finger contact is a task failure, not a
            # free-fall event. Require downward separation and actual falling.
            vz=self._data.qvel[self._part_v[name]+2]
            lost=not grip and expected[2]-self._pose(name)[0][2]>.035 and vz<-.08
            self._lost_ticks=self._lost_ticks+1 if lost else 0
            if self._lost_ticks>=4:
                self._log['drop_events'].append({'part':name,'time_s':float(self._data.time),'stage':self.stage})
                self._pending_drop=float(self._data.time)

    def step(self,action):
        if self._done: raise RuntimeError('episode already terminated')
        self.robot.apply(action)
        stage=self.stage
        rows=[]; bilateral=[0.,0.]; collision=False; maximum=0.
        for _ in range(10):
            mujoco.mj_step(self._model,self._data)
            # Store contact solver results and their matching state, not stale pre-integration contacts.
            mujoco.mj_forward(self._model,self._data)
            contacts,bilateral,unexpected,load=self._contact_readback()
            collision|=bool(unexpected); maximum=max(maximum,load)
            rows.append({'time_s':float(self._data.time),'stage':stage,'active_part':self.active_part,
                         'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),
                         'ctrl':self._data.ctrl.tolist(),'qacc_warmstart':self._data.qacc_warmstart.tolist(),
                         'actuator_torque_nm':self._data.qfrc_actuator[self.robot.v].tolist(),
                         'contacts':contacts,'grip_normal_forces_n':bilateral,'belt_speed_ms':self._speed})
        self._steps+=1; self._ticks+=1
        metrics=self.metrics()
        depth=metrics[self.active_part]['position_error_m'][2]
        previous=self._log['assembly_metrics'][-1][self.active_part]['position_error_m'][2] if self._log['assembly_metrics'] else depth
        jam=self.phase=='lower' and maximum>1. and abs(depth-previous)<.0001 and not metrics[self.active_part]['seated']
        self._jam_ticks=self._jam_ticks+1 if jam else 0
        self._log['commands'].append(np.asarray(action).tolist()); self._log['times_s'].append(float(self._data.time))
        self._log['stages'].append(stage); self._log['physics_samples'].extend(rows)
        self._log['robot_collisions'].append(int(collision)); self._log['peak_loads_n'].append(maximum)
        self._log['jamming_durations'].append(self._jam_ticks*DT)
        self._log['assembly_metrics'].append(metrics); self._log['control_step_count']=self._steps
        self._advance(bilateral)
        self._log['part_drops'].append(int(bool(self._log['drop_events']) and
            self._log['drop_events'][-1]['time_s']==float(self._data.time)))
        if not self._done:
            p,_=self._pose(self.active_part)
            if self.mode=='transport' and self.phase in {'approach','descend','close'} and p[0]>.4:
                self._transition('missed_pick_window',failure='missed_pick_window')
            elif self.mode=='moving_assembly' and self._data.xpos[self._model.body('pallet').id,0]>.32:
                self._transition('fixture_left_window',failure='fixture_left_window')
            elif self._ticks>=1500 or self._steps>=self.MAX_STEPS:
                self._transition(self.stage+'_timeout',failure=self.stage+'_timeout')
        self._log['final_metrics']=self.metrics()
        return self._observe(),float(self._log['success']),self._done,dict(self._log)

    @staticmethod
    def continuous_motion(trace):
        """Both insertion operations must take place on an actually moving pallet."""
        if trace.get('scenario')!='moving_assembly': return True
        for name in ('connector','cap'):
            speeds=[r['qvel'][9] for r in trace['physics_samples'] if r['stage']==name+':lower']
            if not speeds or np.mean(speeds)<.5*trace['belt_speed_ms']: return False
        return True

    @staticmethod
    def evaluate(episode_data):
        metrics=episode_data.get('final_metrics',{})
        history=episode_data.get('assembly_metrics',[])
        samples=episode_data.get('physics_samples',[])
        n=len(history)
        stable=(n>=25 and episode_data.get('stages',[])[-25:]==['verify']*25 and
                all(all(row.get(name,{}).get('seated') and row[name].get('relative_speed_ms',1.)<.008
                        and row[name].get('relative_spin_rads',1.)<.05 for name in PARTS)
                    for row in history[-25:]))
        released=(len(samples)==n*10 and n>=25 and all(samples[k*10+9]['ctrl'][7]>250 and
                  max(samples[k*10+9]['grip_normal_forces_n'])<.1 for k in range(n-25,n)))
        success=bool(episode_data.get('success') and stable and released and ProductionWorkcellTask.continuous_motion(episode_data))
        return TaskResult(sum(bool(metrics.get(n,{}).get('seated')) for n in PARTS)/3,success,episode_data,
                          {'model':ProductionWorkcellTask.MODEL_SCHEMA,'observation_source':'simulator_state'})

    def render_frame(self,width=960,height=720,*,overview=False):
        if self._renderer is None:
            self._model.vis.global_.offwidth=max(1280,width); self._model.vis.global_.offheight=max(960,height)
            self._renderer=mujoco.Renderer(self._model,width=width,height=height)
        camera=self._model.camera('overview').id
        position=np.array([1.10,-1.30,1.05] if overview else [.72,.95,.90])
        target=np.array([-.02,.02,.12])
        if self.mode=='transport' and not overview: position=np.array([.78,1.05,.98]); target=np.array([-.08,0.,.13])
        backward=position-target; backward/=np.linalg.norm(backward)
        right=np.cross([0.,0.,1.],backward); right/=np.linalg.norm(right)
        self._model.cam_pos[camera]=position
        mujoco.mju_mat2Quat(self._model.cam_quat[camera],np.column_stack((right,np.cross(backward,right),backward)).ravel())
        self._model.cam_fovy[camera]=42 if overview else 39
        mujoco.mj_camlight(self._model,self._data)
        option=mujoco.MjvOption(); option.geomgroup[3]=0
        self._renderer.update_scene(self._data,camera='overview',scene_option=option)
        from esrb.tasks.tool_handling.appearance import decorate_scene
        decorate_scene(self._renderer.scene)
        return self._renderer.render().copy()

    def close(self):
        if self._renderer is not None: self._renderer.close(); self._renderer=None
