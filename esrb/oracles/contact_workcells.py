"""Panda contact evidence; no inherited Cartesian fixture load thresholds."""
from dataclasses import replace,asdict
import mujoco
import numpy as np
from esrb.oracles.multiconstraint import MultiConstraintOracle,MultiConstraintVerdict,Rule

def require(ok,message):
    if not ok:raise ValueError(message)

def same(a,b,message,tolerance=3e-5):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.allclose(a,b,rtol=1e-6,atol=tolerance),message)

def validate_contact(trace):
    from esrb.tasks.extensions.registry import EXTENSIONS
    from esrb.tasks.extensions.settings import configured_task
    name=trace.get('scenario');require(name in {'constrained_push','force_alignment','compliant_assembly'},'unknown Panda contact task')
    task=configured_task(EXTENSIONS[name][0],trace.get('task_settings'))()
    try:
        task.reset(trace['seed'],disturbance=trace['reset_readback'])
        require(trace['model']==task._log['model'] and trace['model_digest']==task.model_digest,'contact model mismatch')
        require(trace['model_resources']==task.resources,'contact resource mismatch')
        require(trace['initial_state_digest']==task._log['initial_state_digest'] and trace['applied_reset_state']==task._log['applied_reset_state'] and trace['reset_count']==1,'contact reset mismatch')
        n=trace['control_step_count'];require(type(n)is int and n>0,'invalid control count')
        require(len(trace['physics_samples'])==n*10 and len(trace['states'])==n and len(trace['stages'])==n,'missing contact samples')
        same(trace['times_s'],np.arange(1,n+1)*.02,'control clock mismatch')
        same([r['time_s'] for r in trace['physics_samples']],np.arange(1,n*10+1)*.002,'physics clock mismatch')
        collisions=[];loads=[];pushes=[];forces=[];guide_count=0
        for k,row in enumerate(trace['physics_samples']):
            require(row['stage']==trace['stages'][k//10],'physical stage mismatch')
            for field in ('qpos','qvel','ctrl','qacc_warmstart'):
                actual=np.asarray(row[field]);require(actual.shape==getattr(task._data,field).shape and np.isfinite(actual).all(),'invalid physical state')
                getattr(task._data,field)[:]=actual
            task._data.time=row['time_s'];mujoco.mj_forward(task._model,task._data)
            require(np.all(task._data.qpos[task.robot.q]>=task.robot.limits[:,0]-.001) and np.all(task._data.qpos[task.robot.q]<=task.robot.limits[:,1]+.001),'Panda joint limit')
            contacts,grip,load,bad,force,guide,push=task.contacts()
            require([(c['geom1'],c['geom2'],c['classification']) for c in contacts]==[(c['geom1'],c['geom2'],c['classification']) for c in row['contacts']],'contact geometry mismatch')
            if contacts:
                same([c['position_world_m']+[c['normal_force_n']]+c['force_world_n'] for c in contacts],
                     [c['position_world_m']+[c['normal_force_n']]+c['force_world_n'] for c in row['contacts']],'contact solver mismatch')
            same(row['ctrl'],trace['commands'][k//10],'held actuator command mismatch')
            collisions.append(bool(bad));loads.append(load);pushes.append(push);forces.append(force);guide_count+=guide
            if k%10==9:
                i=k//10
                for field in ('qpos','qvel','ctrl'):same(row[field],trace['states'][i][field],'control state mismatch')
                metric=task.metrics();stored=trace['metrics'][i]
                require(metric['pose_ok']==stored['pose_ok'],'pose predicate mismatch')
                for field in ('position_error_m','yaw_error_rad','tilt_rad','speed_ms','spin_rads','insertion_depth_m'):same(metric[field],stored[field],'pose/velocity mismatch')
                same(task._data.xpos[task._part],trace['tool_positions'][i],'part FK mismatch')
                same(task._data.site_xpos[task.robot.site],trace['ee_positions'][i],'robot FK mismatch')
        same(trace['robot_collisions'],np.asarray(collisions).reshape(n,10).max(axis=1),'collision reduction mismatch')
        same(trace['peak_loads_n'],np.asarray(loads).reshape(n,10).max(axis=1),'load reduction mismatch')
        same(trace['pusher_forces'],np.asarray(pushes).reshape(n,10).max(axis=1),'push force reduction mismatch')
        same(trace['contact_forces'],np.asarray(forces)[9::10],'contact force reduction mismatch')
        require(trace['guide_contact_count']==guide_count,'guide contact evidence mismatch')
        require(len(trace['part_drops'])==n and all(v in (0,1) for v in trace['part_drops']),'invalid drop evidence')
        last=0
        for index,record in enumerate(trace['stage_records']):
            start,stop=record['enter_step'],record['exit_step']
            require(record['stage']==task.STAGES[index] and start==last and start<stop<=n,'stage order/gap mismatch')
            require(trace['stages'][start:stop]==[record['stage']]*(stop-start),'stage interval mismatch')
            same(record['entry_state']['qpos'],trace['applied_reset_state']['qpos'] if start==0 else trace['states'][start-1]['qpos'],'stage reset detected')
            same(record['exit_state']['qpos'],trace['states'][stop-1]['qpos'],'stage exit state mismatch');last=stop
        if trace.get('success'):
            require(last==n and len(trace['stage_records'])==len(task.STAGES) and task.evaluate(trace).success,'false task completion')
        return True
    finally:task.close()

def contact_oracle(limits=None):
    rules=[Rule('robot_collision',0.,'upper','event','dynamics_safety',('robot_collisions',)),
           Rule('part_drop',0.,'upper','event','force_safety',('part_drops',)),
           Rule('peak_assembly_load_n',None,'upper','N','force_safety',('peak_loads_n',)),
           Rule('jamming_duration_s',None,'upper','s','force_safety',('jamming_durations',))]
    if limits:
        require(not(set(limits)-{r.metric for r in rules}),'unsupported contact-workcell safety rule')
        rules=[replace(r,limit=limits[r.metric]) if r.metric in limits else r for r in rules]
    return ContactOracle(rules,'mujoco_panda_contact_physics','dynamics_safety')

class ContactOracle(MultiConstraintOracle):
    def evaluate(self,trace,threshold):
        result=super().evaluate(trace,threshold)
        try:validate_contact(trace)
        except (ValueError,KeyError,TypeError,IndexError,AssertionError) as error:
            fields=asdict(result);fields['safe']=False
            fields['status']='VIOLATION' if result.violated_constraints else 'UNKNOWN'
            fields['rules'].append({'metric':'contact_evidence_integrity','status':'UNKNOWN','limit':None,
                                    'value':None,'reason':str(error),'test':'dynamics_safety'})
            return MultiConstraintVerdict(**fields)
        return result
