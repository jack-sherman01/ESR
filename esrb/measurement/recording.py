"""Per integration-step evidence, captured without terminating on alarms.

The solver contact buffer after mj_step describes the solve at t_start, whereas
qpos_end describes integration to t_end. Both are saved; no mj_forward is inserted
between the solve and capture. Each job owns its process and model.
"""
from contextlib import AbstractContextManager
import hashlib
import json
import gzip
import shutil
import tempfile
from pathlib import Path
import numpy as np
import mujoco


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(model, data):
    h=hashlib.sha256()
    for value in (data.qpos,data.qvel,data.ctrl,data.act,data.mocap_pos,data.mocap_quat,
                  data.qacc_warmstart,data.qfrc_applied,data.xfrc_applied,data.userdata):
        h.update(np.asarray(value,dtype='<f8').tobytes())
    h.update(np.asarray([data.time],dtype='<f8').tobytes())
    return h.hexdigest()


def inventory(model):
    bodies=[]
    for i in range(model.nbody):
        chain=[]; b=i
        while b:
            chain.append(model.body(b).name or f'body#{b}'); b=int(model.body_parentid[b])
        bodies.append(dict(id=i,name=model.body(i).name,ancestors=chain,mass_kg=float(model.body_mass[i]),
                           inertial_origin_m=model.body_ipos[i].tolist(),inertial_quaternion=model.body_iquat[i].tolist()))
    return dict(bodies=bodies, geoms=[dict(id=i,name=model.geom(i).name,
                body=int(model.geom_bodyid[i]),type=int(model.geom_type[i]),
                size=model.geom_size[i].tolist(), contype=int(model.geom_contype[i]),
                local_position=model.geom_pos[i].tolist(),local_quaternion=model.geom_quat[i].tolist(),
                conaffinity=int(model.geom_conaffinity[i])) for i in range(model.ngeom)])


class Recorder(AbstractContextManager):
    def __init__(self,model,data,folder,roles,chunk_size=512):
        self.model,self.data,self.folder=model,data,Path(folder)
        self.folder.mkdir(parents=True,exist_ok=True)
        self.rows=[]; self.contacts=[]; self.chunks=[]; self.count=0; self.control=-1
        self.phase='reset'; self.phase_names=[]; self.chunk_size=chunk_size
        self.roles=roles; self.forcing=None
        self.body_indices=sorted(set(roles['objects'])|set(roles.get('goal_bodies',{}).values()))
        self.site_indices=list(roles.get('tcp_sites',[]))
        if model.opt.integrator==mujoco.mjtIntegrator.mjINT_RK4:
            raise ValueError('RK4 internal stages are unsupported by the t_start contact evidence schema')
        mujoco.mj_saveModel(model,str(self.folder/'model.mjb'))
        model_digest=sha(self.folder/'model.mjb')
        with (self.folder/'model.mjb').open('rb') as source,(self.folder/'model.mjb.gz').open('wb') as compressed:
            with gzip.GzipFile(filename='',mode='wb',fileobj=compressed,compresslevel=3,mtime=0) as target:
                shutil.copyfileobj(source,target)
        (self.folder/'model.mjb').unlink()
        self.geometry=inventory(model)
        self.geometry.update(roles=roles,model_sha256=model_digest)
        (self.folder/'geometry.json').write_text(json.dumps(self.geometry,indent=2)+'\n')
        self.initial=identity(model,data)
        np.savez_compressed(self.folder/'initial.npz',**{k:np.array(getattr(data,k),copy=True)
            for k in ('qpos','qvel','ctrl','act','mocap_pos','mocap_quat','qacc_warmstart','qfrc_applied','xfrc_applied','userdata')},
            time=np.array([data.time]))

    def __enter__(self):
        self.original=mujoco.mj_step
        def wrapped(model,data,*args,**kwargs):
            if model is not self.model or data is not self.data:
                return self.original(model,data,*args,**kwargs)
            if args or kwargs:
                raise ValueError('multi-step mj_step bypasses per-physics-step evidence')
            if self.forcing: self.forcing(model,data)
            before={k:np.array(getattr(data,k),copy=True) for k in (
                'qpos','qvel','act','ctrl','qacc_warmstart','qfrc_applied','xfrc_applied','mocap_pos','mocap_quat')}
            t=float(data.time)
            self.original(model,data)
            # Derived poses now match this solve at t_start, even if the caller
            # did not refresh its previous integration's kinematic cache.
            positions=data.xpos[self.body_indices].copy();rotations=data.xmat[self.body_indices].copy()
            sites=data.site_xpos[self.site_indices].copy();site_rot=data.site_xmat[self.site_indices].copy()
            start=len(self.contacts)
            for j,c in enumerate(data.contact):
                f=np.zeros(6); mujoco.mj_contactForce(model,data,j,f)
                self.contacts.append(np.r_[c.geom1,c.geom2,c.dist,c.pos,c.frame,f])
            if self.phase not in self.phase_names:self.phase_names.append(self.phase)
            self.rows.append(dict(time=np.array([t,data.time]), control=self.control,
                phase=self.phase_names.index(self.phase), **{'start_'+k:v for k,v in before.items()},
                end_qpos=data.qpos.copy(),end_qvel=data.qvel.copy(),body_pos=positions,body_rot=rotations,
                site_pos=sites,site_rot=site_rot,sensors=data.sensordata.copy(),
                actuator_force=data.actuator_force.copy(),qfrc_actuator=data.qfrc_actuator.copy(),
                contact_range=np.array([start,len(self.contacts)],dtype=np.int64)))
            self.count+=1
            if len(self.rows)>=self.chunk_size:self.flush()
        mujoco.mj_step=wrapped
        return self

    def flush(self):
        if not self.rows:return
        arrays={k:np.asarray([r[k] for r in self.rows]) for k in self.rows[0]}
        arrays['contacts']=np.asarray(self.contacts,dtype=float).reshape(-1,21)
        path=self.folder/f'physics-{len(self.chunks):05d}.npz'
        # Lossless temporal XOR + column layout substantially reduces repetitive
        # rigid-body poses without reducing sampling rate or floating precision.
        encoding={}
        for key,a in list(arrays.items()):
            if a.dtype==np.float64 and a.size:
                bits=np.ascontiguousarray(a).view(np.uint64).reshape(len(a),-1)
                delta=bits.copy();delta[1:]=bits[1:]^bits[:-1]
                arrays[key]=np.ascontiguousarray(delta.T)
                encoding[key]=list(a.shape)
        arrays['_encoding']=np.array(json.dumps(encoding))
        np.savez_compressed(path,**arrays)
        self.chunks.append(dict(file=path.name,sha256=sha(path),steps=len(self.rows)))
        self.rows=[];self.contacts=[]

    def __exit__(self,*error):
        mujoco.mj_step=self.original
        self.flush()
        manifest=dict(schema='passive-physics/1',physics_steps=self.count,control_steps=self.control+1,
            physics_dt_s=float(self.model.opt.timestep),initial_state_sha256=self.initial,
            integrator=int(self.model.opt.integrator),solver=int(self.model.opt.solver),
            initial_snapshot_sha256=sha(self.folder/'initial.npz'),
            body_indices=self.body_indices,site_indices=self.site_indices,
            pose_storage='direct object/fixture/TCP poses; other link poses reconstruct from full state + model',
            physics_steps_per_control=round(self.roles.get('control_dt_s',self.model.opt.timestep)/self.model.opt.timestep),
            model_sha256=self.geometry['model_sha256'],geometry_sha256=sha(self.folder/'geometry.json'),
            model_archive_sha256=sha(self.folder/'model.mjb.gz'),
            phases=self.phase_names,chunks=self.chunks,complete=error[0] is None,
            time_convention='contacts/poses at t_start solve; end_qpos at t_end; force on geom2 in contact frame',
            wrench_origin='contact point; frame rows are contact basis in world; world vector = frame.T @ local',
            applied_wrench='xfrc_applied: world force followed by world torque about the body center of mass',
            units=dict(position='m',time='s',force='N',torque='N m',angle='rad'))
        (self.folder/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')


def chunks(folder,verify=True):
    folder=Path(folder); m=json.loads((folder/'manifest.json').read_text())
    if verify and sha(folder/'geometry.json')!=m['geometry_sha256']:raise ValueError('geometry identity mismatch')
    if verify:
        if (folder/'model.mjb.gz').exists():
            if sha(folder/'model.mjb.gz')!=m['model_archive_sha256']:raise ValueError('model identity mismatch')
        elif sha(folder/'model.mjb')!=m['model_sha256']:raise ValueError('model identity mismatch')
    if verify and m.get('initial_snapshot_sha256') and sha(folder/'initial.npz')!=m['initial_snapshot_sha256']:raise ValueError('initial state identity mismatch')
    for entry in m['chunks']:
        p=folder/entry['file']
        if verify and sha(p)!=entry['sha256']:raise ValueError('physics chunk digest mismatch')
        with np.load(p,allow_pickle=False) as raw:
            encoding=json.loads(str(raw['_encoding'])) if '_encoding' in raw else {}
            result={k:raw[k] for k in raw.files if k!='_encoding'}
            for key,shape in encoding.items():
                bits=np.bitwise_xor.accumulate(result[key].T,axis=0)
                result[key]=np.ascontiguousarray(bits).view(np.float64).reshape(shape)
            yield result


def load_model(folder):
    folder=Path(folder)
    if (folder/'model.mjb').exists():return mujoco.MjModel.from_binary_path(str(folder/'model.mjb'))
    with tempfile.TemporaryDirectory(prefix='esrb-measurement-model-') as temporary:
        p=Path(temporary)/'model.mjb'
        with gzip.open(folder/'model.mjb.gz','rb') as source,p.open('wb') as target:shutil.copyfileobj(source,target)
        manifest=json.loads((folder/'manifest.json').read_text())
        if sha(p)!=manifest['model_sha256']:raise ValueError('decompressed model identity mismatch')
        return mujoco.MjModel.from_binary_path(str(p))
