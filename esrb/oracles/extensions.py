"""Geometry-recomputed safety evidence for the new physical scenario families."""
from dataclasses import asdict
import mujoco
import numpy as np

from esrb.oracles.multiconstraint import MultiConstraintOracle
from esrb.oracles.assembly import AssemblyVerdict,validate_assembly_measurements,stage_safety
from esrb.oracles.tool_retention import validate_tool_measurements
from esrb.tasks.extensions.registry import EXTENSIONS
from esrb.tasks.extensions.settings import configured_task


def require(condition,message):
    if not condition:
        raise ValueError(message)


def _same(actual,expected,label):
    a,b=np.asarray(actual),np.asarray(expected)
    require(a.shape==b.shape and np.isfinite(a).all() and np.allclose(a,b,rtol=0,atol=1e-9),label)


def validate_extension(trace):
    if trace.get('model','').startswith('panda-contact/'):
        from esrb.oracles.contact_workcells import validate_contact
        return validate_contact(trace)
    if trace.get('model','').startswith('panda-production/'):
        from esrb.oracles.production import validate_production
        return validate_production(trace)
    name=trace.get('moving_model',trace.get('extension_model',trace.get('model',''))).split('/')[0]
    require(name in EXTENSIONS,'unregistered extension scene')
    task_type=configured_task(EXTENSIONS[name][0],trace.get('task_settings'))
    task=task_type()
    try:
        task.reset(disturbance=trace['reset_readback'])
        require(trace['applied_reset_state']==task._log['applied_reset_state'],'applied reset state mismatch')
        require(trace.get('measurement_frame')=='world' and trace.get('wrench_origin')==task._log['wrench_origin'],
                'measurement frame or wrench origin mismatch')
        n=len(trace['times_s'])
        require(n>0 and len(trace['stages'])==n,'missing stage clock')
        _same(trace['times_s'],np.arange(1,n+1)*.02,'control timebase mismatch')
        if name in {'constrained_push','force_alignment'}:
            require(trace['model_digest']==task.model_digest and trace['reset_count']==1,'contact scene mismatch')
            require(trace['measurement_schema']==task._log['measurement_schema'],'contact schema mismatch')
            samples=np.asarray(trace['contact_samples'],dtype=float)
            require(samples.shape==(n,10,9) and np.isfinite(samples).all(),'missing contact substeps')
            physics=trace['physics_samples']
            require(len(physics)==10*n,'missing contact state evidence')
            _same(samples[:,:,0],np.arange(1,n*10+1).reshape(n,10)*.002,'contact substep clock mismatch')
            idx=np.arange(n)
            _same(trace['contact_forces'],samples[idx,np.argmax(np.linalg.norm(samples[:,:,1:4],axis=2),axis=1),1:4],'contact force reduction mismatch')
            _same(trace['contact_torques'],samples[idx,np.argmax(np.linalg.norm(samples[:,:,4:7],axis=2),axis=1),4:7],'contact torque reduction mismatch')
            _same(trace['wall_forces'],samples[:,:,7].max(axis=1),'wall force reduction mismatch')
            _same(trace['pusher_forces'],samples[:,:,8].max(axis=1),'pusher force reduction mismatch')
            flags=[int(any(c['part']=='channel_endstop' and c['normal_force_n']>0 for c in row['contacts'])) for row in physics]
            _same(trace['forbidden_collisions'],np.asarray(flags).reshape(n,10).max(axis=1),'collision reduction mismatch')
            excess=[]
            from esrb.tasks.tool_handling.friction import yaw
            for i,row in enumerate(physics):
                require(row['stage']==trace['stages'][i//10],'physical contact stage mismatch')
                _same([row['time_s'],*row['force_world_n'],*row['torque_world_nm'],row['wall_force_n'],row['pusher_force_n']],
                      samples[i//10,i%10], 'raw contact binding mismatch')
                task._data.qpos[:]=row['qpos']
                task._data.qvel[:]=row['qvel']
                task._data.ctrl[:]=row['ctrl']
                task._data.qacc_warmstart[:]=row['qacc_warmstart']
                mujoco.mj_forward(task._model,task._data)
                # Recompute solver contact wrench from stored physical state and held actuator targets.
                f,t,w,contacts,applied=task._contact_readback()
                _same(f,row['force_world_n'],'physical contact force mismatch')
                _same(t,row['torque_world_nm'],'physical contact torque mismatch')
                _same(w,row['wall_force_n'],'physical wall load mismatch')
                _same(applied,row['pusher_force_n'],'physical pusher load mismatch')
                require(contacts==row['contacts'],'physical contact points mismatch')
                _same(row['ctrl'],trace['commands'][i//10],'held actuator target mismatch')
                excess.append(task._boundary_excess())
                _same(row['boundary_excess_m'],excess[-1],'physical boundary geometry mismatch')
                if i%10==9:
                    p,q=task._data.xpos[task._part],task._data.xquat[task._part]
                    _same(p,trace['tool_positions'][i//10],'part pose mismatch')
                    _same(q,trace['tool_quaternions'][i//10],'part orientation mismatch')
                    _same([*task._data.qpos[:2],.018],trace['ee_positions'][i//10],'pusher pose mismatch')
            _same(trace['boundary_excess'],np.asarray(excess).reshape(n,10).max(axis=1),'boundary reduction mismatch')
            require(trace['stage_events'][0]=={'stage':'approach','step':0,'time_s':0.},'contact initial stage mismatch')
            last=0
            for k,event in enumerate(trace['stage_events'][1:],1):
                stop=event['step']
                require(event['stage']==task.STAGES[k] and last<stop<=n,'contact stage order mismatch')
                _same(event['time_s'],trace['times_s'][stop-1],'contact stage time mismatch')
                require(trace['stages'][last:stop]==[task.STAGES[k-1]]*(stop-last),'contact stage interval mismatch')
                last=stop
            previous=None
            require(0<len(trace['stage_records'])<=3,'invalid contact stage records')
            for k,record in enumerate(trace['stage_records']):
                start,stop=record['enter_step'],record['exit_step']
                require(record['stage']==task.STAGES[k] and type(start) is int and type(stop) is int
                        and 0<=start<stop<=n,'invalid contact stage interval')
                require(start==(previous['exit_step'] if previous else 0),'contact stage gap')
                require(record['entry_state']==(previous['exit_state'] if previous else
                        {key:trace['applied_reset_state'][key] for key in ('qpos','qvel')}),'contact stage reset')
                for key in ('qpos','qvel'):
                    _same(record['exit_state'][key],physics[stop*10-1][key],'contact stage state mismatch')
                _same(record['enter_time_s'],start*.02,'contact stage entry time')
                _same(record['exit_time_s'],stop*.02,'contact stage exit time')
                require(trace['stages'][start:stop]==[record['stage']]*(stop-start),'contact stage sample gap')
                if record['status']=='completed':
                    require(record['exit_condition']==('part_contact','part_at_target_pose','stable_target_for_0.4s')[k]
                            and record['failure_reason'] is None,'contact transition condition')
                    p=np.asarray(trace['tool_positions'][stop-1]);q=np.asarray(trace['tool_quaternions'][stop-1])
                    pose_ok=abs(p[0]-.09)<.004 and abs(p[1])<(.004 if task.ALIGN else .02) and (not task.ALIGN or abs(yaw(q))<.08)
                    require((np.linalg.norm(trace['contact_forces'][stop-1])>.05 if k==0 else pose_ok),
                            'contact transition predicate')
                    if k==2:
                        require(task.evaluate(trace).success,'contact stable hold predicate')
                else:
                    require(k==len(trace['stage_records'])-1 and record['status']=='failed'
                            and record['exit_condition']==record['failure_reason']==trace['failure_reason']
                            and record['failure_reason']==record['stage']+'_timeout'
                            and (n>=task.MAX_STEPS or stop-start>=550),'contact stage timeout mismatch')
                previous=record
            require(previous['exit_step']==n,'unfinished contact stage record')
        else:
            validate_assembly_measurements(trace,task_type=task_type)
            _same(trace['socket_xy_m'],task._log['socket_xy_m'],'physical socket displacement mismatch')
            force=np.linalg.norm(trace['contact_forces'],axis=1)
            _same(trace['sustained_forces'],[np.mean(force[i-9:i+1]) if trace['stages'][max(0,i-9):i+1]==['insert']*10 else 0. for i in range(n)],'sustained load mismatch')
            depths=trace['insertion_depth_m']
            ticks=0
            values=[]
            for i in range(n):
                jam=trace['stages'][i]=='insert' and i>=10 and force[i]>1 and depths[i]-depths[i-9]<.0001
                ticks=ticks+1 if jam else 0
                values.append(ticks*.02)
            _same(trace['jamming_durations'],values,'jamming reduction mismatch')
    finally:
        task.close()


class ExtensionOracle(MultiConstraintOracle):
    """Missing or inconsistent raw evidence is UNKNOWN, never an acceptance."""
    def evaluate(self,trace,generation):
        base=super().evaluate(trace,generation)
        fields=asdict(base)
        reports=[]
        try:
            validate_extension(trace)
            if 'stage_records' in trace and 'retention_active' in trace:
                reports=stage_safety(trace,fields['rules'])
            # Extra physical channels have their actual stage/time attribution.
            for stage in dict.fromkeys(trace['stages']):
                mask=np.asarray(trace['stages'])==stage
                events=[]
                for rule in fields['rules']:
                    if len(rule.get('channels',[]))!=1 or rule['limit'] is None:
                        continue
                    values=np.asarray(trace[rule['channels'][0]])
                    if values.ndim==2 and values.shape[1]==3 and rule['metric']!='peak_tip_speed_ms':
                        values=np.linalg.norm(values,axis=1)
                    if values.ndim!=1:
                        continue
                    bad=(values>rule['limit'] if rule['bound']=='upper' else values<rule['limit']) & mask
                    if np.any(bad):
                        first=int(np.flatnonzero(bad)[0])
                        event={'metric':rule['metric'],'time_s':trace['times_s'][first],
                               'value':float(values[first]),'limit':rule['limit']}
                        if rule['metric'] in {'min_surface_distance_m','proxy_collision'}:
                            key='distance_m' if rule['metric']=='min_surface_distance_m' else 'contacts'
                            raw=next(row for row in trace['moving_samples'][first*10:first*10+10]
                                     if (row[key]<rule['limit'] if key=='distance_m' else bool(row[key])))
                            event.update(physics_time_s=raw['time_s'],closest_part=raw['closest_part'],
                                         closest_points_world=raw['closest_points_world'])
                        events.append(event)
                reports.append({'stage':stage,'scalar_violations':events})
        except (KeyError,ValueError,TypeError,IndexError,OverflowError):
            fields['safe']=False
            fields['status']='VIOLATION' if base.violated_constraints else 'UNKNOWN'
            fields['rules'].append({'metric':'extension_evidence_integrity','test':'force_safety',
                'limit':None,'status':'UNKNOWN','value':None,'reason':'invalid_physics_evidence'})
        return AssemblyVerdict(**fields,stage_reports=reports)
