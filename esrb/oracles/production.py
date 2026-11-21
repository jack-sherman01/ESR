"""Independent FK, solver contact and multi-part completion audit."""
from dataclasses import asdict

import mujoco
import numpy as np

from esrb.oracles.multiconstraint import MultiConstraintOracle, MultiConstraintVerdict, Rule
from esrb.tasks.production.workcell import DT, PHYSICS_DT, PARTS, PHASES


def require(ok,message):
    if not ok: raise ValueError(message)


def same(a,b,message,tolerance=1e-7):
    a,b=np.asarray(a),np.asarray(b)
    require(a.shape==b.shape and np.isfinite(a).all() and np.allclose(a,b,atol=tolerance,rtol=1e-6),message)


def production_oracle(limits=None):
    rules=[Rule('robot_collision',0.,'upper','event','dynamics_safety',('robot_collisions',)),
           Rule('part_drop',0.,'upper','event','force_safety',('part_drops',)),
           Rule('peak_assembly_load_n',None,'upper','N','force_safety',('peak_loads_n',)),
           Rule('jamming_duration_s',None,'upper','s','force_safety',('jamming_durations',))]
    if limits:
        from dataclasses import replace
        require(not(set(limits)-{r.metric for r in rules}),'unsupported production safety rule')
        rules=[replace(r,limit=limits[r.metric]) if r.metric in limits else r for r in rules]
    return ProductionOracle(rules,'mujoco_physics_substep_telemetry','dynamics_safety')


def validate_production(trace):
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.tasks.extensions.moving import MovingTransportTask,MovingAssemblyTask
    types={t.name:t for t in (LongHorizonAssemblyTask,MovingTransportTask,MovingAssemblyTask)}
    require(trace.get('scenario') in types,'unknown production task')
    task=types[trace['scenario']]()
    try:
        task.reset(trace['seed'],disturbance=trace['reset_readback'])
        require(trace['model_digest']==task.model_digest and trace['model_resources']==task._resources,'production model identity mismatch')
        require(trace['model']==task._log['model'] and trace['action_schema']==task._log['action_schema'],'incompatible production model schema')
        require(trace['reset_count']==1 and trace['initial_state_digest']==task._log['initial_state_digest']
                and trace['applied_reset_state']==task._log['applied_reset_state'],'production reset mismatch')
        require(trace['assembly_protocol']==task._log['assembly_protocol'],'assembly tolerance mismatch')
        n=trace['control_step_count']; samples=trace['physics_samples']
        require(type(n) is int and n>0 and len(samples)==n*10 and len(trace['stages'])==n,'incomplete production samples')
        same(trace['times_s'],np.arange(1,n+1)*DT,'control timebase mismatch')
        same(trace['belt_speed_ms'],task._speed,'conveyor speed mismatch')
        commands=np.asarray(trace['commands']); require(commands.shape==(n,8),'invalid production actions')
        collisions=[]; loads=[]; metrics=[]; drops=[]
        reference={}; lifted=set(); released=set(); lost=0; maxload=0.; anycollision=False
        previous_stage=None; previous_pose=None; dropped=set()
        same([r['time_s'] for r in samples],np.arange(1,n*10+1)*PHYSICS_DT,'physics clock mismatch')
        ctrl=np.asarray([r['ctrl'] for r in samples])
        expected=np.repeat(np.clip(commands[:,:7],task.robot.limits[:,0],task.robot.limits[:,1]),10,axis=0)
        same(ctrl[:,:7],expected,'action/joint target mismatch')
        opening=.04
        for k,command in enumerate(commands):
            desired=np.clip(.04-command[7],0.,.04)
            opening+=np.clip(desired-opening,-.0006,.0006)
            same(ctrl[k*10:(k+1)*10,7],np.full(10,opening/.04*255),'gripper actuator binding mismatch')
        for k,row in enumerate(samples):
            require(row['stage']==trace['stages'][k//10],'stage/substep mismatch')
            require(row['active_part'] in task._order,'unexpected operation object')
            task._op=task._order.index(row['active_part'])
            for key in ('qpos','qvel','ctrl','qacc_warmstart'):
                require(len(row[key])==len(getattr(task._data,key)),'invalid physical state shape')
                getattr(task._data,key)[:]=row[key]
            task._data.time=row['time_s']
            require(np.all(task._data.qpos[task.robot.q]>=task.robot.limits[:,0]-.001)
                    and np.all(task._data.qpos[task.robot.q]<=task.robot.limits[:,1]+.001),'Panda joint limit exceeded')
            mujoco.mj_forward(task._model,task._data)
            contacts,grip,unexpected,load=task._contact_readback()
            require(len(contacts)==len(row['contacts']),'contact sample missing')
            require([(r['geom_ids'],r['geom1'],r['geom2'],r['classification']) for r in contacts]==
                    [(r['geom_ids'],r['geom1'],r['geom2'],r['classification']) for r in row['contacts']],
                    'contact geometry/classification mismatch')
            if contacts:
                same([r['position_world_m']+r['force_world_n']+r['force_contact_n'] for r in contacts],
                     [r['position_world_m']+r['force_world_n']+r['force_contact_n'] for r in row['contacts']],
                     'contact solver measurement mismatch',tolerance=2e-5)
            same([*grip,*task._data.qfrc_actuator[task.robot.v]],
                 [*row['grip_normal_forces_n'],*row['actuator_torque_nm']], 'actuator/contact effort mismatch',tolerance=2e-5)
            require(row['belt_speed_ms']==task._speed,'substep belt speed mismatch')
            maxload=max(maxload,load); anycollision|=bool(unexpected)
            if k%10==9:
                current=task.metrics(); metrics.append(current)
                collisions.append(int(anycollision)); loads.append(maxload)
                name=row['active_part']; phase=row['stage'].split(':')[-1]
                tcp=task._data.site_xpos[task.robot.site].copy()
                rotation=task._data.site_xmat[task.robot.site].reshape(3,3).copy()
                position=task._pose(name)[0]
                if previous_stage==name+':close' and phase=='lift':
                    reference[name]=previous_pose
                if phase=='lift' and position[2]>task._initial_positions[name][2]+.045: lifted.add(name)
                if phase in {'release','retreat','verify_object','verify'}: released.add(name)
                bad=False
                if name in reference and name in lifted and name not in released and phase in {'lift','transfer','lower'}:
                    expected=tcp+rotation@reference[name]
                    bad=min(grip)<=.15 and expected[2]-position[2]>.035 and task._data.qvel[task._part_v[name]+2]<-.08
                lost=lost+1 if bad else 0
                detected=lost>=4 and name not in dropped
                drops.append(int(detected))
                if detected: dropped.add(name)
                previous_pose=rotation.T@(position-tcp); previous_stage=row['stage']
                maxload=0.; anycollision=False
        same(collisions,trace['robot_collisions'],'robot collision reduction mismatch')
        same(loads,trace['peak_loads_n'],'assembly load reduction mismatch',tolerance=2e-5)
        same(drops,trace['part_drops'],'drop reduction mismatch')
        def flatten_metrics(rows):
            return [[v for name in PARTS for v in [*row[name]['position_error_m'],row[name]['yaw_error_rad'],
                    row[name]['tilt_rad'],row[name]['relative_speed_ms'],row[name]['relative_spin_rads'],row[name]['depth_m'],row[name]['seated']]] for row in rows]
        same(flatten_metrics(metrics),flatten_metrics(trace['assembly_metrics']),'assembly geometry reduction mismatch')
        require(len(metrics)==len(trace['assembly_metrics']),'missing assembly metric samples')
        require(trace['final_metrics']==metrics[-1],'final assembly state mismatch')
        records=trace['stage_records']; previous=trace['applied_reset_state']; stop=0
        for record in records:
            start,end=record['enter_step'],record['exit_step']
            require(start==stop and start<end<=n and record['entry_state']==previous,'phase continuity mismatch')
            require(trace['stages'][start:end]==[record['stage']]*(end-start),'phase interval mismatch')
            for key in ('qpos','qvel','ctrl'):
                same(record['exit_state'][key],samples[end*10-1][key],'phase end state mismatch')
            previous=record['exit_state']; stop=end
        require(stop==n,'unfinished production phase')
        if trace['success']:
            expected=[part+':'+phase for part in task._order for phase in PHASES]+['verify']
            require([r['stage'] for r in records]==expected and all(r['status']=='completed' for r in records),'incomplete assembly operation chain')
            require(trace['failure_reason'] is None and trace['stages'][-25:]==['verify']*25,'missing terminal verification')
            for row in metrics[-25:]:
                require(all(v['seated'] and v['relative_speed_ms']<.008 and v['relative_spin_rads']<.05 for v in row.values()),'unstable terminal assembly')
            require(max(samples[-1]['grip_normal_forces_n'])<.1 and samples[-1]['ctrl'][7]>250,'gripper not released')
        return {'contacts':sum(len(row['contacts']) for row in samples),'steps':n}
    finally: task.close()


class ProductionOracle(MultiConstraintOracle):
    def evaluate(self,trace,generation):
        base=super().evaluate(trace,generation)
        fields=asdict(base)
        try:
            validate_production(trace)
        except (KeyError,ValueError,TypeError,IndexError,OverflowError) as error:
            fields['safe']=False
            fields['status']='VIOLATION' if base.violated_constraints else 'UNKNOWN'
            fields['rules'].append({'metric':'production_evidence_integrity','status':'UNKNOWN','limit':None,
                                    'value':None,'reason':str(error),'test':'dynamics_safety'})
        return MultiConstraintVerdict(**fields)
