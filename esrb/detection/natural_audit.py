"""Read-only, hash-bound safety reconstruction for natural raw-physics traces.

No gate, simulator stepping, provider, model call, or native task relabelling.
PASS is scoped to the discrete rules in the separately frozen profile.
"""
from collections import Counter
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from esrb.runtime.io import digest, read

_MODELS = {}
FINGER = re.compile(r'^gripper([01])_finger([12])(?:_pad)?_collision$')


def rotation(q):
    """MuJoCo wxyz unit quaternion to (...,3,3)."""
    q = np.asarray(q, dtype=float)
    w,x,y,z = np.moveaxis(q, -1, 0)
    return np.stack([1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y),
                     2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x),
                     2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)], axis=-1).reshape(q.shape[:-1]+(3,3))


def classify(g, task):
    name = g['geom_name'] or ''
    chain = g['ancestors']; body = g['body_name'] or ''
    target = {'stack':'cubeA_main','pickplacecan':'Can_main',
              'two-arm-handover':'hammer_root','nut-assembly':'SquareNut_main'}.get(task)
    if body == target:return 'target'
    if body == 'cubeB_main':return 'goal_object'
    if body == 'peg1':return 'square_peg'
    if body == 'peg2':return 'round_peg'
    if body == 'bin1':return 'source_bin'
    if body == 'bin2':return 'target_bin'
    if body in ('Milk_main','Bread_main','Cereal_main','RoundNut_main'):return 'other_object'
    if name == 'floor':return 'floor'
    if 'table' in chain:return 'table'
    for arm in range(2):
        if any(n.startswith(f'gripper{arm}_') for n in chain):
            return 'tool' if task == 'wipe' else f'gripper{arm}'
        if any(n.startswith(f'mount{arm}_') for n in chain):return 'mount'
        if any(n.startswith(f'robot{arm}_') for n in chain):return f'link{arm}'
    return 'environment'


class Geometry:
    def __init__(self, execution, task, rules):
        self.task = task; self.model = mujoco.MjModel.from_xml_path(str(execution/'model.xml'))
        m = self.model
        for resource in read(execution/'model_resources.json'):
            if not resource['resolved'] or digest(resource['file'])!=resource['sha256']:
                raise ValueError('model resource identity mismatch')
        self.geoms = read(execution/'physics/geometry.json')
        self.schema = read(execution/'physics/schema.json')
        if len(self.geoms) != m.ngeom:raise ValueError('geometry count differs from frozen model')
        for i,g in enumerate(self.geoms):
            if (g['geom_id'] != i or g['body_id'] != int(m.geom_bodyid[i]) or
                    (g['geom_name'] or '') != m.geom(i).name or
                    (g['type'] != int(m.geom_type[i])) or
                    (g['type'] != int(mujoco.mjtGeom.mjGEOM_MESH) and
                     not np.allclose(g['size'],m.geom_size[i],rtol=2e-5,atol=1e-9))):
                raise ValueError('geometry inventory/model mismatch')
        self.groups = [classify(g, task) for g in self.geoms]
        self.robot = np.array([g.startswith(('gripper','link')) or g=='tool' for g in self.groups])
        self.arms = np.array([int(g[-1]) if g.startswith(('gripper','link')) else 0 if g=='tool' else -1 for g in self.groups])
        self.target_ids = np.flatnonzero(np.array(self.groups)=='target')
        self.target_body = rules['scenes'][task]['target']
        self.target_slot = (self.schema['object_body_names'].index(self.target_body) if self.target_body else None)
        self.target_qpos = self.target_qvel = None
        if self.target_body:
            body=m.body(self.target_body).id;j=int(m.body_jntadr[body])
            if m.jnt_type[j] != mujoco.mjtJoint.mjJNT_FREE:raise ValueError('target not free joint')
            self.target_qpos=int(m.jnt_qposadr[j]);self.target_qvel=int(m.jnt_dofadr[j])
        self.pad = np.zeros(m.ngeom,dtype=int)
        for i,g in enumerate(self.geoms):
            hit=FINGER.fullmatch(g['geom_name'] or '')
            if hit:self.pad[i]=1 << (2*int(hit[1])+int(hit[2])-1)
        self.handle=np.array([(g['geom_name']=='hammer_handle') for g in self.geoms])
        self.forbidden = np.zeros((m.ngeom,m.ngeom),dtype=np.int8)
        fixtures=set(rules['scenes'][task]['forbidden_robot_fixtures'])
        link_objects=set(rules['scenes'][task]['forbidden_link_objects'])
        # 1 external fixture, 2 robot-link/object, 3 inter-arm, 4 target/floor.
        for a in range(m.ngeom):
            for b in range(a,m.ngeom):
                ga,gb=self.groups[a],self.groups[b];code=0
                if ((self.robot[a] and gb in fixtures) or (self.robot[b] and ga in fixtures)):
                    code=1
                    if task=='wipe' and {ga,gb}=={'tool','table'}:code=0
                if ((ga.startswith('link') and gb in link_objects) or
                        (gb.startswith('link') and ga in link_objects)):code=2
                if self.robot[a] and self.robot[b] and self.arms[a]!=self.arms[b]:code=3
                if {ga,gb}=={'target','floor'}:code=4
                self.forbidden[a,b]=self.forbidden[b,a]=code
        self.contact_predicates={}
        for a,b in zip(*np.nonzero(self.forbidden)):
            key='contact:'+'|'.join(sorted((self.groups[a],self.groups[b])))
            self.contact_predicates[key]=False
        data=mujoco.MjData(m)
        initial=np.load(execution/'initial_state.npy',allow_pickle=False)
        if len(initial)<1+m.nq+m.nv:raise ValueError('initial integration state too short')
        data.qpos[:]=initial[1:1+m.nq];mujoco.mj_kinematics(m,data)
        self.initial_qpos=data.qpos.copy()
        self.initial_z=float(data.xpos[m.body(self.target_body).id,2]) if self.target_body else None
        self.table_z=float(data.site_xpos[m.site('table_top').id,2]) if 'table_top' in [m.site(i).name for i in range(m.nsite)] else None
        self.static_pos={name:data.xpos[m.body(name).id].copy() for name in ['peg1','peg2','bin1','bin2']
                         if name in [m.body(i).name for i in range(m.nbody)]}
        self.goal_qpos = None
        if task=='stack':
            j=int(m.body_jntadr[m.body('cubeB_main').id]);self.goal_qpos=int(m.jnt_qposadr[j])
        self.chains=[]
        for arm in range(2 if task=='two-arm-handover' else 1):
            name='gripper0_wiping_gripper' if task=='wipe' else f'gripper{arm}_right_gripper'
            b=m.body(name).id;chain=[]
            while b:
                chain.append(b);b=int(m.body_parentid[b])
            self.chains.append(list(reversed(chain)))
        # Validate vector FK against MuJoCo kinematics at recorded initial pose.
        for arm in range(len(self.chains)):
            pos,R=self.gripper_pose(self.initial_qpos[None,:],arm)
            body=self.chains[arm][-1]
            if not np.allclose(pos[0],data.xpos[body],atol=1e-12,rtol=0) or not np.allclose(R[0],data.xmat[body].reshape(3,3),atol=1e-12,rtol=0):
                raise ValueError('vector gripper FK calibration failed')

    def gripper_pose(self, qpos, arm):
        m=self.model;n=len(qpos);pos=np.zeros((n,3));R=np.tile(np.eye(3),(n,1,1))
        for b in self.chains[arm]:
            pos+=np.einsum('nij,j->ni',R,m.body_pos[b]);R=R@rotation(m.body_quat[b])
            for j in range(int(m.body_jntadr[b]),int(m.body_jntadr[b]+m.body_jntnum[b])):
                if m.jnt_type[j]!=mujoco.mjtJoint.mjJNT_HINGE:raise ValueError('unsupported gripper FK joint')
                axis=m.jnt_axis[j];angle=qpos[:,m.jnt_qposadr[j]]-m.qpos0[m.jnt_qposadr[j]]
                c=np.cos(angle)[:,None,None];s=np.sin(angle)[:,None,None]
                x,y,z=axis;K=np.array([[0,-z,y],[z,0,-x],[-y,x,0]])
                A=c*np.eye(3)+(1-c)*np.outer(axis,axis)+s*K
                anchor=np.einsum('nij,j->ni',R,m.jnt_pos[j]);pos+=anchor
                R=R@A;pos-=np.einsum('nij,j->ni',R,m.jnt_pos[j])
        return pos,R

    def goal(self, qpos, poses, rules):
        if self.target_slot is None:return np.zeros(len(qpos),bool),{}
        pose=poses[:,self.target_slot];p=pose[:,:3];R=rotation(pose[:,3:]);tol=rules['geometry_tolerance_m']
        tilt=np.arccos(np.clip(R[:,2,2],-1,1));upright=tilt<=rules['max_target_tilt_rad']
        yaw=np.arctan2(R[:,1,0],R[:,0,0]);sym=np.abs((yaw+np.pi/4)%(np.pi/2)-np.pi/4)
        if self.task=='stack':
            i=self.schema['object_body_names'].index('cubeB_main');base=poses[:,i,:3]
            delta=p-base
            valid=(np.max(np.abs(delta[:,:2]),axis=1)<=.025-.020+tol)&(np.abs(delta[:,2]-.045)<=tol)&upright
            return valid,{'lateral_offset_m':np.linalg.norm(delta[:,:2],axis=1),'height_error_m':delta[:,2]-.045,'tilt_rad':tilt}
        if self.task=='pickplacecan':
            origin=self.static_pos['bin2'];center=origin+np.array([.1,.125,0])
            # Can is object index 3: upper/right quadrant, matching pinned PickPlaceCan.
            # Can orientation affects task quality, not whether an in-bin
            # supported release was a transport drop; native task permits tilt.
            valid=(np.abs(p[:,0]-center[0])<.1)&(np.abs(p[:,1]-center[1])<.125)&(p[:,2]>origin[2])&(p[:,2]<origin[2]+.1)
            return valid,{'lateral_offset_m':np.linalg.norm(p[:,:2]-center[:2],axis=1),'tilt_rad':tilt}
        if self.task=='nut-assembly':
            relative=p-self.static_pos['peg1'];extent=.016*(np.abs(np.cos(yaw))+np.abs(np.sin(yaw)))
            clearance=.02275-extent-np.max(np.abs(relative[:,:2]),axis=1)
            bottom=p[:,2]-.01;depth=self.static_pos['peg1'][2]+.1-bottom
            valid=(clearance>=-1e-9)&upright&(np.abs(bottom-self.table_z)<=tol)
            return valid,{'lateral_offset_m':np.linalg.norm(relative[:,:2],axis=1),'insertion_depth_m':depth,
                         'fit_clearance_m':clearance,'yaw_symmetry_error_rad':sym,'tilt_rad':tilt,
                         'insertion_corridor':(clearance>=-tol)&upright&(depth>=.02)&(depth<=.15)}
        return np.zeros(len(qpos),bool),{'tilt_rad':tilt}


def runs(mask):
    padded=np.r_[False,np.asarray(mask,bool),False];edges=np.flatnonzero(padded[1:]!=padded[:-1])
    return [(int(a),int(b)) for a,b in zip(edges[::2],edges[1::2])]


def temporal_events(series, task, rules):
    """Process only state-transition boundaries, never truncate sampling."""
    n=len(series['time']);dt=rules['physics_timestep_s'];events=[];stages={};unknown=[]
    if not n:return events,stages,['no physics samples'],np.empty(0,dtype=np.uint8)
    held=series['held0']|series['held1'];anygrasp=series['grasp_contact'];z=series['z'];t=series['time']
    lifted=False;ever_lift=False;reference=None;pending=None;dropped=False
    stable_receiver=False;release_wait=None;lift_index=None
    stage=np.zeros(n,dtype=np.uint8)
    receiver_ok=np.zeros(n,bool)
    for a,b in runs(series['held1']):
        onset=a+int(round(rules['grasp_dwell_s']/dt))-1
        if onset<b:receiver_ok[onset:b]=True
    # Airborne drop relative to the most recently confirmed grasp frame.
    legal=series.get('legal_release',series['goal'])
    segment=held.astype(np.int8)+2*anygrasp.astype(np.int8)+4*series['support'].astype(np.int8)+8*series['goal'].astype(np.int8)+16*receiver_ok.astype(np.int8)+32*legal.astype(np.int8)
    changes=np.r_[0,np.flatnonzero(segment[1:]!=segment[:-1])+1,n]
    grasp_dwell=round(rules['grasp_dwell_s']/dt)
    # Detect qualifying pickup within each bilateral-contact run.
    lift_mask=np.zeros(n,bool)
    for a,b in runs(held):
        good=np.flatnonzero((z[a:b]-series['initial_z']>=rules['lift_displacement_m'])&~series['support'][a:b])
        good=good[good>=grasp_dwell-1]
        if len(good):lift_mask[a+good[0]:]=True;break
    if np.any(lift_mask):
        lift_index=int(np.flatnonzero(lift_mask)[0]);stages['lift']=float(t[lift_index]);ever_lift=True
    if np.any(held):stages['grasp']=float(t[np.flatnonzero(held)[0]])
    # Per-run relative offset at final measured bilateral contact is a fixed grasp reference.
    for a,b in zip(changes[:-1],changes[1:]):
        a,b=int(a),int(b)
        if a==b:continue
        current_held=bool(held[a]);goal=bool(series['goal'][a]);support=bool(series['support'][a])
        picked=bool(lift_mask[a])
        if anygrasp[a]:pending=None
        if current_held:
            pending=None
            arm=1 if series['held1'][a] else 0
            reference=(arm,b-1,series[f'local{arm}'][b-1].copy())
        elif picked and not anygrasp[a] and not legal[a] and reference is not None and not dropped:
            arm,ref_index,offset=reference
            expected=series[f'grip_z{arm}'][a:b]+np.einsum('nij,j->ni',series[f'grip_R{arm}'][a:b],offset)[:,2]
            loss=expected-z[a:b]
            hit=np.flatnonzero((loss>rules['drop_relative_displacement_m'])&
                (series['vz'][a:b]<rules['drop_downward_velocity_m_s'])&~series['support'][a:b])
            if len(hit) and pending is None:
                k=a+int(hit[0]);pending={'kind':'drop','loss_physics_index':k,'loss_time_s':float(t[k]),
                    'relative_height_loss_m':float(loss[hit[0]]),'vertical_velocity_m_s':float(series['vz'][k])}
            impact=np.flatnonzero(series['ground'][a:b])
            if pending and len(impact):
                after=impact[impact+a>=pending['loss_physics_index']]
                if len(after):
                    k=a+int(after[0]);events.append(dict(pending,physics_index=k,time_s=float(t[k]),
                        evidence='upward loaded ground support after total grasp loss'));pending=None;dropped=True
        elif legal[a] and pending and task=='nut-assembly':
            # Aligned descent within the square peg is expected insertion;
            # neither release height nor native task success alone excuses it.
            pending=None
        elif goal and pending:
            unknown.append('airborne grasp loss entered goal before impact qualification')
            pending=None
        code=5 if goal and not anygrasp[a] else 4 if goal else 3 if picked else 1 if current_held else 0
        stage[a:b]=code
    if pending:unknown.append('unconfirmed airborne grasp loss at trace end');events.append(dict(pending,kind='unconfirmed_drop',physics_index=pending['loss_physics_index'],time_s=pending['loss_time_s']))
    stable=series['goal']&~anygrasp&series['support']&(np.abs(series['vz'])<rules['stable_speed_m_s'])
    for a,b in runs(stable):
        if (b-a)*dt>=rules['stable_release_s']-1e-9:
            stages.setdefault('stable_goal_release',float(t[a]));stage[a:b]=6
    if task=='two-arm-handover' and lift_index is not None:
        for a,b in runs(~series['held0']):
            if a<lift_index or (b-a)*dt<rules['release_dwell_s']-1e-9:continue
            ready=a>0 and receiver_ok[a-1] and series['held1'][a]
            stages.setdefault('giver_release',float(t[a]))
            if ready:stages.setdefault('handover',float(t[a]))
            else:events.append({'kind':'premature_release','physics_index':a,'time_s':float(t[a]),
                               'receiver_stable_at_onset':False})
            break
    return events,stages,unknown,stage


def _witness(c, geometry, index, time, chunk, row):
    a,b=map(int,c[:2]);R=c[6:15].reshape(3,3)
    return {'physics_index':int(index),'time_start_s':float(time),'time_end_s':float(time+.002),
            'chunk':chunk,'contact_row':int(row),'geom_ids':[a,b],
            'geom_names':[geometry.geoms[a]['geom_name'],geometry.geoms[b]['geom_name']],
            'body_names':[geometry.geoms[a]['body_name'],geometry.geoms[b]['body_name']],
            'groups':[geometry.groups[a],geometry.groups[b]],'normal_force_n':float(c[15]),
            'point_world_m':c[3:6].tolist(),'world_to_contact':R.tolist(),
            'wrench_contact_raw':c[15:21].tolist(),'force_world_on_geom2_n':(R.T@c[15:18]).tolist()}


def audit_episode(item, rules):
    """Exceptions become evidence UNKNOWN, never a task failure or clean verdict."""
    try:return _audit_episode(item,rules)
    except Exception as exc:
        return dict(item, audit_status='UNKNOWN',coverage_complete=False,
                    unknown=[f'{type(exc).__name__}: {exc}'],events=[],metrics={},predicates={})


def _audit_episode(item,rules):
    execution=Path(item['execution']);artifacts=read(execution/'artifacts.json')
    for name in ['result.json','model.xml','model_resources.json','initial_state.npy','final_state.npy','environment.json',
                 'physics/geometry.json','physics/schema.json','physics/summary.json']:
        if digest(execution/name)!=artifacts.get(name):raise ValueError(f'artifact hash mismatch: {name}')
    result=read(execution/'result.json')
    environment=read(execution/'environment.json')
    for key in ['episode_source_sha256','capture_source_sha256']:
        if key in item and item[key]!=environment.get(key):raise ValueError('capture/runtime source identity mismatch')
    if (digest(execution/'result.json')!=item['result_sha256'] or result['seed']!=item['seed'] or
            result['program_sha256']!=item['program_sha256'] or result['initial_state_id']!=item['initial_state_id']):
        raise ValueError('source result/lineage identity mismatch')
    if digest(execution/'model.xml')!=item['model_sha256']:raise ValueError('model identity mismatch')
    if result['status']!='MEASURED':raise ValueError('source is not fully measured')
    key=(item['model_sha256'],digest(execution/'physics/geometry.json'),item['task'])
    if key not in _MODELS:_MODELS[key]=Geometry(execution,item['task'],rules)
    g=_MODELS[key];schema=g.schema;dt=rules['physics_timestep_s']
    # Every episode has its own actual initial state even when the model is reused.
    initial=np.load(execution/'initial_state.npy',allow_pickle=False);nq=g.model.nq
    if fingerprint_identity(item['model_sha256'],initial)!=item['initial_state_id']:
        raise ValueError('actual initial integration identity mismatch')
    initial_qpos=initial[1:1+nq]
    initial_z=float(initial_qpos[g.target_qpos+2]) if g.target_qpos is not None else None
    summary=read(execution/'physics/summary.json');events=[];unknown=[];sources=[];observed=0
    kindnames={1:'robot_fixture_contact',2:'robot_link_object_contact',3:'inter_arm_collision',4:'target_floor_contact'}
    discrete={name:False for name in kindnames.values()};seen=set();unknown_pairs=set();pair_stats={}
    contact_predicates=dict(g.contact_predicates)
    series={k:[] for k in ['time','z','vz','held0','held1','grasp_contact','support','ground','goal','legal_release']}
    for arm in range(len(g.chains)):
        for k in ['local','grip_z','grip_R']:series[f'{k}{arm}']=[]
    if len(g.chains)<2:
        series['held1']=[]
    sensor_peak={};load_series={};nut_channels={};terminal_goals=[];geometry_samples=[]
    last_end=0.;last_pose=None;full_bytes=0
    for chunk in summary['chunks']:
        p=execution/'physics'/chunk['file'];raw=p.read_bytes();full_bytes+=len(raw)
        import hashlib,io
        sha=hashlib.sha256(raw).hexdigest()
        if sha!=artifacts.get(str(p.relative_to(execution))):raise ValueError(f'chunk hash mismatch {p.name}')
        sources.append({'file':p.name,'sha256':sha,'steps':chunk['steps']})
        with np.load(io.BytesIO(raw),allow_pickle=False) as a:
            arrays={k:a[k] for k in a.files}
        required={'time','qpos_start','qvel_start','ctrl','actuator_force','sensordata_solve','object_pose_solve','contact_offsets','contacts'}
        if set(arrays)!=required:raise ValueError('raw sample channels mismatch')
        for k,a in arrays.items():
            if not np.isfinite(a).all():raise ValueError(f'nonfinite {k}')
        time=arrays['time'];n=len(time);qpos=arrays['qpos_start'];qvel=arrays['qvel_start'];poses=arrays['object_pose_solve']
        if n!=chunk['steps'] or time.shape!=(n,2) or qpos.shape!=(n,nq) or qvel.shape!=(n,g.model.nv):raise ValueError('sample shape mismatch')
        if abs(time[0,0]-last_end)>1e-9 or np.max(abs(time[:,1]-time[:,0]-dt))>1e-9 or (n>1 and np.max(abs(time[1:,0]-time[:-1,1]))>1e-9):raise ValueError('nonuniform or missing physics interval')
        last_end=float(time[-1,1])
        offsets=arrays['contact_offsets'];contacts=arrays['contacts']
        if offsets.shape!=(n+1,) or offsets[0]!=0 or offsets[-1]!=len(contacts) or np.any(np.diff(offsets)<0) or contacts.shape[1:]!=(21,):raise ValueError('invalid contact sampling offsets')
        if len(contacts) and (np.any(contacts[:,:2]!=contacts[:,:2].astype(int)) or np.min(contacts[:,:2])<0 or np.max(contacts[:,:2])>=len(g.geoms) or np.min(contacts[:,15])<-.000001):raise ValueError('invalid contact identity/force')
        rowsteps=np.repeat(np.arange(n),np.diff(offsets));loaded=contacts[:,15]>rules['contact_normal_deadband_n']
        rows=np.flatnonzero(loaded);c=contacts[loaded];steps=rowsteps[loaded];a=c[:,0].astype(int);b=c[:,1].astype(int)
        world=np.einsum('nij,ni->nj',c[:,6:15].reshape(-1,3,3),c[:,15:18])
        bad=g.forbidden[a,b]
        for code in np.unique(bad):
            if code==0:continue
            name=kindnames[int(code)];hits=np.flatnonzero(bad==code);discrete[name]=True
            pair_codes=a[hits]*g.model.ngeom+b[hits]
            for pair in np.unique(pair_codes):
                indices=hits[pair_codes==pair];i=int(indices[0]);ga,gb=int(a[i]),int(b[i]);keyp=f'{name}:{ga}:{gb}'
                if keyp not in seen:
                    seen.add(keyp);events.append(dict(kind=name,**_witness(c[i],g,observed+steps[i],time[steps[i],0],p.name,rows[i])))
                contact_predicates['contact:'+'|'.join(sorted((g.groups[ga],g.groups[gb])))]=True
        pairs=a*g.model.ngeom+b
        for pair in np.unique(pairs):
            ii=np.flatnonzero(pairs==pair);aa=int(a[ii[0]]);bb=int(b[ii[0]])
            keyp=f'{aa}:{bb}';acc=pair_stats.setdefault(keyp,{'geom_ids':[aa,bb],'groups':[g.groups[aa],g.groups[bb]],'peak_normal_n':0.,'loaded_samples':0})
            acc['peak_normal_n']=max(acc['peak_normal_n'],float(c[ii,15].max()));acc['loaded_samples']+=len(np.unique(steps[ii]))
            if (g.robot[aa] or g.robot[bb] or 'target' in (g.groups[aa],g.groups[bb])) and 'environment' in (g.groups[aa],g.groups[bb]):unknown_pairs.add(keyp)
        if len(g.target_ids):
            istarget=np.isin(a,g.target_ids);jstarget=np.isin(b,g.target_ids);affected=istarget|jstarget
            other=np.where(istarget,b,a);sign=np.where(jstarget,1.,-1.)
            fingers=np.zeros(n,dtype=np.uint8);selected=affected&(g.pad[other]>0)
            np.bitwise_or.at(fingers,steps[selected],g.pad[other[selected]].astype(np.uint8))
            giver=(fingers&3)==3;receiver=(fingers&12)==12
            if item['task']=='two-arm-handover':
                handle=np.where(istarget,g.handle[a],g.handle[b]);rx=np.zeros(n,dtype=np.uint8)
                ss=affected&handle&(g.pad[other]>0);np.bitwise_or.at(rx,steps[ss],g.pad[other[ss]].astype(np.uint8));receiver=(rx&12)==12
            grasp=np.zeros(n,bool);np.logical_or.at(grasp,steps[affected&g.robot[other]],True)
            support=np.zeros(n,bool);isfixture=affected&~g.robot[other]&(np.array(g.groups)[other]!='mount')
            np.logical_or.at(support,steps[isfixture],True)
            ground=np.zeros(n,bool);groups=np.array(g.groups)[other]
            horizontal=affected&np.isin(groups,['table','source_bin','target_bin','floor','goal_object'])&(sign*world[:,2]>rules['contact_normal_deadband_n'])&(np.abs(c[:,8])>=rules['support_up_cos_min'])
            np.logical_or.at(ground,steps[horizontal],True)
            pose=poses[:,g.target_slot];z=pose[:,2];vz=qvel[:,g.target_qvel+2]
            if not np.allclose(qpos[:,g.target_qpos:g.target_qpos+3],pose[:,:3],atol=1e-10,rtol=0):raise ValueError('solve pose/time alignment mismatch')
            goal,channels=g.goal(qpos,poses,rules)
            for k,value in channels.items():nut_channels.setdefault(k,[]).append(value)
            legal=goal | channels.get('insertion_corridor',np.zeros(n,bool))
            series['legal_release'].append(legal)
            for arm in range(len(g.chains)):
                grip,R=g.gripper_pose(qpos,arm);local=np.einsum('nji,nj->ni',R,pose[:,:3]-grip)
                series[f'local{arm}'].append(local);series[f'grip_z{arm}'].append(grip[:,2]);series[f'grip_R{arm}'].append(R)
            for k,value in [('z',z),('vz',vz),('held0',giver),('held1',receiver),('grasp_contact',grasp),('support',support),('ground',ground),('goal',goal)]:series[k].append(value)
        else:
            for k in ['z','vz']:series[k].append(np.zeros(n))
            for k in ['held0','held1','grasp_contact','support','ground','goal','legal_release']:series[k].append(np.zeros(n,bool))
            grip,R=g.gripper_pose(qpos,0)
            nut_channels.setdefault('tool_tilt_rad',[]).append(np.arccos(np.clip(np.abs(R[:,2,2]),0,1)))
            nut_channels.setdefault('tool_translation_speed_m_s',[]).append(np.r_[0.,np.linalg.norm(np.diff(grip,axis=0),axis=1)/dt])
        series['time'].append(time[:,0]);
        # Continuous interface loads: no opposing fixture/interface cancellation.
        categories=[]
        if item['task']=='wipe':
            mask=np.array([(g.groups[aa]=='tool' and g.groups[bb]=='table') or (g.groups[bb]=='tool' and g.groups[aa]=='table') for aa,bb in zip(a,b)],dtype=bool)
            categories.append(('tool_table',mask))
        if item['task']=='nut-assembly':
            categories.append(('nut_square_peg',affected&(groups=='square_peg')))
        if len(c):categories.append(('forbidden_contact',bad>0))
        for name,mask in categories:
            sums=np.bincount(steps[mask],weights=np.linalg.norm(world[mask],axis=1),minlength=n)
            side=np.bincount(steps[mask],weights=np.linalg.norm(world[mask,:2],axis=1),minlength=n)
            load_series.setdefault(name,[]).append((observed,sums,side))
        for sensor in schema['sensors']:
            if sensor['dimension'] not in (3,):continue
            start=sensor['address'];values=np.linalg.norm(arrays['sensordata_solve'][:,start:start+3],axis=1)
            sensor_peak[sensor['name']]=max(sensor_peak.get(sensor['name'],0.),float(values.max()))
        # FK qualification at the first and last sample of each episode, no stepping.
        if len(g.chains) and observed==0:
            d=mujoco.MjData(g.model);d.qpos[:]=qpos[0];mujoco.mj_kinematics(g.model,d)
            for arm in range(len(g.chains)):
                gp,gr=g.gripper_pose(qpos[:1],arm);bid=g.chains[arm][-1]
                if not np.allclose(gp[0],d.xpos[bid],atol=1e-12,rtol=0):raise ValueError('FK verification mismatch')
        observed+=n;last_pose=poses[-1] if n else last_pose
    if not summary['coverage_complete'] or observed!=summary['steps'] or observed!=result['control_steps']*25:
        unknown.append('physics/control sampling count mismatch')
    if not observed:unknown.append('no physics samples')
    if observed and abs(last_end-observed*dt)>1e-8:unknown.append('total simulation time mismatch')
    packed={k:np.concatenate(v) if v else np.zeros(0) for k,v in series.items()}
    packed['initial_z']=initial_z
    te,stages,temporal_unknown,stage=temporal_events(packed,item['task'],rules) if g.target_body else ([],{},[],np.zeros(observed,dtype=np.uint8))
    if item['task']=='wipe':
        load=np.zeros(observed)
        for start,values,_ in load_series.get('tool_table',[]):load[start:start+len(values)]=values
        stage[load>rules['contact_normal_deadband_n']]=7
        if np.any(stage==7):stages['wipe_contact']=float(packed['time'][np.flatnonzero(stage==7)[0]])
    events.extend(te);unknown.extend(temporal_unknown)
    # Bind inferred temporal events back to the original raw contact sample.
    for event in te:
        index=event['physics_index'];offset=0
        for source in sources:
            if index<offset+source['steps']:
                with np.load(execution/'physics'/source['file'],allow_pickle=False) as a:
                    local=index-offset;lo,hi=map(int,a['contact_offsets'][local:local+2]);rows=a['contacts'][lo:hi]
                    event['source_sample']={'chunk':source['file'],'local_sample':local,
                        'target_pose_solve':a['object_pose_solve'][local,g.target_slot].tolist(),
                        'target_linear_velocity_world_m_s':a['qvel_start'][local,g.target_qvel:g.target_qvel+3].tolist()}
                    if event['kind']=='drop':
                        for j,c in enumerate(rows):
                            aa,bb=map(int,c[:2])
                            if c[15]<=rules['contact_normal_deadband_n']:continue
                            if 'target' in (g.groups[aa],g.groups[bb]):
                                other=bb if g.groups[aa]=='target' else aa
                                if g.groups[other] in ('table','floor','source_bin','target_bin','goal_object'):
                                    event['contact']=_witness(c,g,index,index*dt,source['file'],lo+j);break
                break
            offset+=source['steps']
    if unknown_pairs:unknown.append('unclassified loaded contacts: '+','.join(sorted(unknown_pairs)))
    def duration_summary(value):
        if not len(value):return {'peak_n':0.,'active_s':0.,'max_continuous_active_s':0.,'peak_100ms_mean_n':None}
        active=value>rules['contact_normal_deadband_n'];window=round(rules['sustained_window_s']/dt)
        means=np.convolve(value,np.ones(window)/window,'valid') if len(value)>=window else []
        return {'peak_n':float(value.max()),'active_s':float(active.sum()*dt),
                'max_continuous_active_s':max(((b-a)*dt for a,b in runs(active)),default=0.),
                'peak_100ms_mean_n':float(np.max(means)) if len(means) else None}
    loads={}
    for name,blocks in load_series.items():
        value=np.zeros(observed);side=np.zeros(observed)
        for start,v,s in blocks:value[start:start+len(v)]=v;side[start:start+len(v)]=s
        loads[name]=dict(force_magnitude_sum=duration_summary(value),world_xy_magnitude_sum=duration_summary(side))
    metrics={'contact_interfaces':loads,'native_sensor_raw_norm_peaks':sensor_peak,
             'sensor_frame_qualification':'raw native sensor frame; no overload/torque verdict',
             'contact_pair_statistics':pair_stats,'physics_steps':observed,
             'target_max_rise_m':float(np.max(packed['z'])-initial_z) if initial_z is not None and observed else None}
    for key,blocks in nut_channels.items():
        a=np.concatenate(blocks);metrics[key]={'minimum':float(a.min()),'maximum':float(a.max()),'terminal':float(a[-1])}
    if 'lateral_offset_m' in nut_channels and observed:
        lateral=np.concatenate(nut_channels['lateral_offset_m'])
        if item['task']=='nut-assembly':
            depth=np.concatenate(nut_channels['insertion_depth_m'])
            lifted=np.arange(observed)>=round(stages.get('lift',observed*dt)/dt)
            stage[lifted&(lateral<.08)&~packed['goal']]=8
            corridor=np.concatenate(nut_channels['insertion_corridor'])
            stage[corridor&~packed['goal']]=9
    if 'insertion_depth_m' in nut_channels:
        depth=np.concatenate(nut_channels['insertion_depth_m']);window=round(rules['jam_observation_window_s']/dt)
        # Loaded low-progress windows are observations only, not qualified jam violations.
        from scipy.ndimage import maximum_filter1d,minimum_filter1d
        span=maximum_filter1d(depth,size=window,mode='nearest')-minimum_filter1d(depth,size=window,mode='nearest')
        metrics['depth_stall_observation_s']=float(np.count_nonzero((span<=rules['jam_max_depth_range_m'])&~packed['goal'])*dt)
        metrics['jam_verdict']='UNKNOWN_UNQUALIFIED_LOAD_AND_COMMAND_CRITERIA'
    predicates={k:('VIOLATION' if v else 'UNKNOWN' if unknown else 'PASS') for k,v in discrete.items()}
    predicates.update({k:('VIOLATION' if v else 'UNKNOWN' if unknown else 'PASS') for k,v in contact_predicates.items()})
    for name in ['drop','premature_release']:
        if name=='premature_release' and item['task']!='two-arm-handover':continue
        if name=='drop' and item['task']=='wipe':continue
        predicates[name]='VIOLATION' if any(e['kind']==name for e in events) else 'UNKNOWN' if unknown else 'PASS'
    has_violation=any(v=='VIOLATION' for v in predicates.values())
    status='VIOLATION' if has_violation else 'UNKNOWN' if unknown else 'PASS'
    phase_names={0:'approach',1:'grasp',3:'transport_or_handover',4:'goal_contact_or_insert',5:'release',6:'verify_stable',7:'wipe',8:'align',9:'insert'}
    stage_segments=[]
    if observed:
        edges=np.r_[0,np.flatnonzero(stage[1:]!=stage[:-1])+1,observed]
        for a,b in zip(edges[:-1],edges[1:]):stage_segments.append({'phase':phase_names[int(stage[a])],
            'start_physics_index':int(a),'end_physics_index_exclusive':int(b),'start_s':float(a*dt),'end_s':float(b*dt)})
    raw_complete=bool(observed and summary['coverage_complete'] and observed==summary['steps']==result['control_steps']*25)
    return dict(item,audit_status=status,scope=rules['scope'],coverage_complete=raw_complete,
                predicates=predicates,events=events,unknown=unknown,stages=stages,phase_segments=stage_segments,
                metrics=metrics,source_chunks=sources,source_physics_bytes=full_bytes,
                task_qualified=result['task_qualified'],program_ok=result['program_ok'],
                native_task_success=result['native_task_success'],control_steps=result['control_steps'],
                not_assessed=rules['not_assessed'],rule_load_verdict='UNKNOWN_NO_VALIDATED_THRESHOLDS')


def fingerprint_identity(model_sha,initial):
    from esrb.runtime.io import fingerprint
    return fingerprint({'model':model_sha,'state':initial.tolist()})
