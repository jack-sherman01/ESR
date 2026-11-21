"""Isolated deterministic episodes. No agent, perception service or network calls."""
import json
import time
from pathlib import Path
import numpy as np
import mujoco
from .recording import Recorder,identity,sha
from .spec import NATIVE,NO_GRASP


def _result(folder,value):
    target=Path(folder)/'result.json';temporary=target.with_suffix('.json.tmp')
    temporary.write_text(json.dumps(value,indent=2,allow_nan=False)+'\n');temporary.replace(target)


def selfbuilt(scene):
    if scene=='tool_handling':
        from esrb.tasks.tool_handling.tool_handling import ToolHandlingTask
        from esrb.agent.tool_handling_policy import ToolHandlingPolicy
        return ToolHandlingTask(),ToolHandlingPolicy()
    if scene=='long_horizon_assembly':
        from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
        from esrb.agent.assembly_policy import AssemblyPolicy
        return LongHorizonAssemblyTask(),AssemblyPolicy()
    from esrb.tasks.extensions.registry import EXTENSIONS
    task,policy=EXTENSIONS[scene]
    actor=policy()
    if scene=='force_alignment':
        # Independent seed-42 engineering trials are retained under
        # measurement-development/alignment-calibration-v1. A gentler approach
        # prevents the tall workpiece tipping at the entrance to the datum guides.
        actor.force_target_n=2.;actor.admittance_gain=.0001;actor.insertion_step_m=.0003
    return task(),actor


def roles_for(model,data,scene,task=None):
    """Declare component identities from model membership, not contact labels."""
    robot=[];pads=[];tool=[];objects=[];fixtures=[];pad_groups=[]
    names=[model.geom(i).name or '' for i in range(model.ngeom)]
    for i,n in enumerate(names):
        if n.startswith(('panda_','robot0_','robot1_','gripper0_','gripper1_')):
            robot.append(i)
        if any(x in n for x in ('finger','pad')) and i in robot:pads.append(i)
        if scene=='wipe' and any(x in n for x in ('wiping_surface','wiping_corner')):tool.append(i)
        if i not in robot and int(model.geom_contype[i])|int(model.geom_conaffinity[i]):fixtures.append(i)
    if task is not None:
        bot=getattr(task,'robot',getattr(task,'_panda',None))
        robot=sorted(bot.geoms);pad_groups=[sorted(g) for g in bot.pad_groups];pads=sorted(bot.pads)
        if scene=='tool_handling':objects=[int(task._tool)]
        elif hasattr(task,'_part_ids'):objects=[int(b) for b in task._part_ids.values()]
        else:objects=[int(task._part)]
    else:
        candidates={'stack':['cubeA_main'],'pickplacecan':['Can_main'],
            'two-arm-handover':['hammer_root','hammer_main'],'nut-assembly':['SquareNut_main'],'wipe':[]}[scene]
        for i in range(model.nbody):
            n=model.body(i).name or ''
            if n in candidates:objects.append(i)
        if scene=='wipe' and tool:objects=sorted({int(model.geom_bodyid[g]) for g in tool})
        for arm in (0,1):
            for side in ('finger1','finger2'):
                group=[i for i in pads if names[i].startswith(f'gripper{arm}_') and side in names[i]]
                if group:pad_groups.append(group)
    part_geoms={}
    for b in objects:
        descendants=[]
        for i in range(model.ngeom):
            parent=int(model.geom_bodyid[i])
            while parent and parent!=b:parent=int(model.body_parentid[parent])
            if parent==b and (model.geom_contype[i] or model.geom_conaffinity[i]):descendants.append(i)
        part_geoms[str(b)]=descendants
    fixture_ids=sorted(set(fixtures)-{g for v in part_geoms.values() for g in v})
    ground=[]
    for g in fixture_ids:
        if g in robot:continue
        size=model.geom_size[g];normal=data.geom_xmat[g].reshape(3,3)[2,2]
        named=any(s in names[g].lower() for s in ('workbench','work_surface','floor','rack','table_collision','conveyor_surface'))
        slab=(int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_BOX) and min(size[:2])>=.05 and size[2]<=.015)
        if normal>=.9 and (named or slab or int(model.geom_type[g])==int(mujoco.mjtGeom.mjGEOM_PLANE)):
            ground.append(g)
    goals={};goal_bodies={}
    if task is not None and scene=='tool_handling':
        goals[str(objects[0])]=dict(position=data.xpos[objects[0]].tolist(),tolerance_m=[.006,.006,.006])
    elif task is not None and hasattr(task,'target'):
        from esrb.tasks.production.geometry import ASSEMBLED_Z,HALF_HEIGHT
        for name,b in task._part_ids.items():
            target,rotation=task.target(name);parent=None
            if task.mode!='transport' and name!='base':
                parent=task._part_ids['base'];target=[0.,0.,ASSEMBLED_Z[name]-ASSEMBLED_Z['base']]
            elif task.mode=='moving_assembly':
                parent=model.body('pallet').id;target=[0.,0.,.007+HALF_HEIGHT[name]]
            goals[str(b)]=dict(position=np.asarray(target).tolist(),parent_body=parent,tolerance_m=[.006,.006,.006])
            if parent is not None:goal_bodies[str(b)]=parent
    elif task is not None and scene in ('constrained_push','force_alignment','compliant_assembly'):
        from esrb.tasks.contact_workcells.geometry import GOAL
        goals[str(objects[0])]=dict(position=GOAL[scene].tolist(),tolerance_m=[.004,.004,.005])
    return dict(robot=robot,pads=pads,pad_groups=pad_groups,tool=tool,objects=objects,
                part_geoms=part_geoms,fixtures=fixture_ids,
                ground_support_geoms=ground,
                initial_positions={str(b):data.xpos[b].tolist() for b in objects},
                goals=goals,goal_bodies=goal_bodies,
                tcp_sites=[int(bot.site)] if task is not None else [i for i in range(model.nsite) if (model.site(i).name or '') in ('gripper0_grip_site','gripper1_grip_site')],
                control_dt_s=.05 if scene in NATIVE else .02,
                grasp_required=scene not in NO_GRASP,scene=scene)


def _slim(value):
    # Per-step state/contact evidence is in NPZ; keep control-level task metrics.
    skip={'physics_samples','robot_physics'}
    return {k:v for k,v in value.items() if k not in skip}


def run(request):
    folder=Path(request['folder']);folder.mkdir(parents=True,exist_ok=False)
    started=time.monotonic(); scene=request['scene'];case=request['case']; seed=request['seed']
    result=dict(schema='controlled-episode/1',scene=scene,case=case,seed=seed,
                program=request['program'],program_sha256=request['program_sha256'],
                process_id=__import__('os').getpid(),model_calls=0)
    task=None
    try:
        if scene in NATIVE:
            from esrb.backends.aspire.controlled_measurement import execute
            result.update(execute(request))
        else:
            task,policy=selfbuilt(scene);obs=task.reset(seed)
            model,data=task._model,task._data
            roles=roles_for(model,data,scene,task)
            result['initial_state_sha256']=identity(model,data)
            result['policy_parameters']=policy.state_dict()
            actions=[];trace={};done=False;intervened=False; hold_ticks=0;seated_ticks=0
            force_body=[roles['objects'][0] if roles['objects'] else None];anchors={}
            with Recorder(model,data,folder/'raw',roles) as recorder:
                def forcing(m,d):
                    # Explicit external load is an engineering fixture, never a hidden state edit.
                    if case not in ('overload','eccentric_torque','jam') or not intervened:return
                    if not roles['objects']:return
                    b=force_body[0];d.xfrc_applied[b,:]=0.
                    if case=='overload':d.xfrc_applied[b,:3]=[0.,0.,-20.]
                    elif case=='eccentric_torque':
                        f=np.array([0.,0.,-20.]);r=np.array([.05,0.,0.])
                        d.xfrc_applied[b,:]=np.r_[f,np.cross(r,f)]
                    else:
                        part=set(roles['part_geoms'][str(b)])
                        touch=any((c.geom1 in part)!=(c.geom2 in part) and
                                  c.geom1 not in roles['pads'] and c.geom2 not in roles['pads'] and c.dist<=0 for c in d.contact)
                        if touch and b not in anchors:anchors[b]=d.xpos[b].copy()
                        if b in anchors:
                            velocity=d.qvel[m.body_dofadr[b]:m.body_dofadr[b]+3]
                            d.xfrc_applied[b,:3]=np.clip(5000.*(anchors[b]-d.xpos[b])-20.*velocity,-100.,100.)
                recorder.forcing=forcing
                for step in range(getattr(task,'MAX_STEPS',6000)):
                    stage=str(getattr(task,'stage',obs.info.get('stage','unknown')))
                    action=np.asarray(policy.act(obs),dtype=float)
                    active=getattr(task,'active_part',None)
                    b=(task._part_ids[active] if active and hasattr(task,'_part_ids') else
                       roles['objects'][0] if roles['objects'] else None)
                    force_body[0]=b
                    lifted=b is not None and data.xpos[b,2]>roles['initial_positions'][str(b)][2]+.04
                    transport=any(p in stage for p in ('transport','transfer','orient','insert','lower','operate'))
                    if case=='task_failure':
                        action[:7]=data.qpos[getattr(task,'robot',getattr(task,'_panda',None)).q]
                        action[-1]=0.
                        if step>=25:break
                    if case=='primary' and scene not in NO_GRASP and lifted and transport:intervened=True
                    if case=='primary' and scene=='tool_handling' and lifted:intervened=True
                    if case in ('overload','eccentric_torque','jam') and transport:intervened=True
                    if case=='primary' and intervened and scene not in NO_GRASP:
                        action[-1]=.002 if scene=='tool_handling' else 0.
                    if case=='seated_release' and scene=='tool_handling' and transport:
                        from esrb.tasks.robots.panda import joint_target,yaw_rotation
                        target=np.asarray(obs.info['pick']);target[2]=.138
                        arrived=np.linalg.norm(np.asarray(obs.info['grip_position'])-target)<.004
                        action=joint_target(obs.info,target,yaw_rotation(obs.info['initial_yaw']),0. if arrived else .023)
                        seated_ticks=seated_ticks+1 if arrived else 0;intervened=True
                        if seated_ticks>=40:break
                    if case=='robot_contact' or (case=='primary' and scene in NO_GRASP):
                        from esrb.tasks.robots.panda import joint_target
                        bot=getattr(task,'robot',getattr(task,'_panda',None))
                        candidates=[g for g in roles['fixtures'] if any(s in (model.geom(g).name or '') for s in ('guide','workbench','work_surface'))]
                        if candidates:
                            g=candidates[0];target=data.geom_xpos[g].copy()
                            action=joint_target(obs.info,target,np.asarray(obs.info['grip_rotation']),.04)
                            intervened=True
                        if step>=300:break
                    holding=bool(case=='legal_hold' and lifted and transport and hold_ticks<60)
                    if holding:
                        bot=getattr(task,'robot',getattr(task,'_panda',None));action[:7]=data.qpos[bot.q]
                        hold_ticks+=1;intervened=True
                    recorder.control=step;recorder.phase=stage
                    bot=getattr(task,'robot',getattr(task,'_panda',None))
                    advancing=stage.split(':')[-1] in ('lower','insert','push') and np.linalg.norm(action[:7]-data.qpos[bot.q])>1e-3
                    actions.append(dict(step=step,time_s=float(data.time),phase=stage,action=action.tolist(),holding=holding,
                                        intervention=intervened,stage_motion_hint=bool(advancing and not holding)))
                    obs,_,done,trace=task.step(action)
                    if done:break
                # The evaluator remains separate from measurement rules.
                evaluation=task.evaluate(trace)
                result.update(task_success=bool(evaluation.success),task_score=float(evaluation.task_score),
                    native_done=done,intervention_executed=intervened,control_steps=len(actions),
                    physics_steps=recorder.count,model_sha256=recorder.geometry['model_sha256'],
                    status='RECORDED',failure_reason=trace.get('failure_reason'),
                    simulation_time_s=float(data.time))
            (folder/'actions.json').write_text(json.dumps(actions,separators=(',',':'))+'\n')
            import gzip
            with gzip.open(folder/'task.json.gz','wt',compresslevel=1) as stream:
                json.dump(_slim(trace),stream,separators=(',',':'),default=lambda v:np.asarray(v).tolist(),allow_nan=False)
        result['wall_s']=time.monotonic()-started
    except Exception as error:
        import traceback
        result.update(status='BLOCKED',error=f'{type(error).__name__}: {error}',task_success=False,
                      wall_s=time.monotonic()-started)
        (folder/'error.txt').write_text(traceback.format_exc())
    finally:
        if task is not None:task.close()
    _result(folder,result)
    return result


def worker(request_path):
    request=json.loads(Path(request_path).read_text())
    if request.get('kind')=='qualification':
        from .qualification import qualify
        output=Path(request['output']);output.parent.mkdir(parents=True,exist_ok=True)
        temporary=output.with_suffix('.json.tmp')
        temporary.write_text(json.dumps(qualify(),indent=2)+'\n');temporary.replace(output)
        return
    def source_matches():
        if not request.get('frozen_source_manifest'):return True
        import hashlib
        files=json.loads(Path(request['frozen_source_manifest']).read_text())
        encoded=json.dumps(files,sort_keys=True,separators=(',',':')).encode()
        if hashlib.sha256(encoded).hexdigest()!=request['frozen_source_sha256']:return False
        root=Path(request['project_root'])
        return all((root/p).is_file() and sha(root/p)==value for p,value in files.items())
    if not source_matches():
        folder=Path(request['folder']);folder.mkdir(parents=True,exist_ok=False)
        result=dict(scene=request['scene'],case=request['case'],seed=request['seed'],status='BLOCKED',
                    task_success=False,error='frozen source mismatch before execution; use a new revision')
        _result(folder,result);return
    result=run(request)
    if result['status']=='RECORDED':
        from .rules import analyze
        started=time.monotonic();result['measurement']=analyze(request['folder'],request.get('limits'),profile=request.get('event_profile'))
        result['analysis_wall_s']=time.monotonic()-started
        if request.get('reference',True):
            from .reference import check
            started=time.monotonic();result['reference']=check(request['folder'],result['measurement'],limits=request.get('limits'),profile=request.get('event_profile'))
            result['reference_wall_s']=time.monotonic()-started
        raw=Path(request['folder'])/'raw'
        result['raw_manifest_sha256']=sha(raw/'manifest.json')
        result['runtime']=dict(python=__import__('sys').version,mujoco=mujoco.__version__,numpy=np.__version__)
        if request.get('limits',{}).get('runtime') and result['runtime']!=request['limits']['runtime']:
            result.update(status='BLOCKED',error='runtime differs from frozen calibration')
    if not source_matches():
        result.update(status='BLOCKED',error='sources changed during execution; evidence retained, new revision required')
    _result(request['folder'],result)
