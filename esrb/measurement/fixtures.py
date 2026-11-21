"""Corruption and threshold checks on derivative evidence, never the originals."""
import json
import shutil
from pathlib import Path
import numpy as np
from .recording import chunks,sha
from .rules import analyze


def _copy_first(source,target):
    target.mkdir(parents=True,exist_ok=False);(target/'raw').mkdir()
    m=json.loads((source/'raw/manifest.json').read_text())
    batch=next(chunks(source/'raw'))
    # A short, self-contained derivative is enough for structural fault tests.
    n=min(30,len(batch['time']))
    controls=batch['control'][:n];last_control=controls[-1]
    if len(set(map(int,controls)))>1:
        expected=m.get('physics_steps_per_control')
        if expected and int(np.count_nonzero(controls==last_control))<expected:
            n=int(np.flatnonzero(controls==last_control)[0])
    stop=int(batch['contact_range'][n-1,1])
    batch={k:(v[:stop].copy() if k=='contacts' else v[:n].copy()) for k,v in batch.items()}
    model_file='model.mjb.gz' if (source/'raw/model.mjb.gz').exists() else 'model.mjb'
    shutil.copy2(source/'raw'/model_file,target/'raw'/model_file)
    shutil.copy2(source/'raw/geometry.json',target/'raw/geometry.json')
    if (source/'raw/initial.npz').exists():shutil.copy2(source/'raw/initial.npz',target/'raw/initial.npz')
    actions=json.loads((source/'actions.json').read_text())
    controls=sorted(set(map(int,batch['control'])))
    actions=[a for a in actions if a['step'] in controls]
    # Preserve actual control IDs; segment starts at zero for every source trajectory.
    m.update(physics_steps=n,control_steps=len(controls),complete=True,chunks=[])
    # This derivative ends on a full control boundary when available.
    counts=[int(np.count_nonzero(batch['control']==c)) for c in controls]
    if len(set(counts))==1:m['physics_steps_per_control']=counts[0]
    (target/'actions.json').write_text(json.dumps(actions))
    (target/'provenance.json').write_text(json.dumps(dict(source=str(source),source_manifest_sha256=sha(source/'raw/manifest.json'),
        kind='computational derivative, not a physical episode')))
    return batch,m


def _write(target,batch,manifest):
    file=target/'raw/physics-00000.npz';np.savez_compressed(file,**batch)
    manifest['chunks']=[dict(file=file.name,sha256=sha(file),steps=len(batch['time']))]
    (target/'raw/manifest.json').write_text(json.dumps(manifest))


def validate_copies(root,rows):
    root=Path(root);results=[]
    for scene in sorted({r['scene'] for r in rows}):
        candidates=[r for r in rows if r['scene']==scene and r['case']=='baseline' and r['seed']==0 and r.get('status')=='RECORDED']
        if not candidates:
            results.append(dict(scene=scene,status='BLOCKED',reason='no nominal source trace'));continue
        source=root/candidates[0]['file']
        for kind in ('missing_frame','nonfinite','time_gap','wrong_component','geometry_identity','model_identity'):
            target=root/'fixtures'/scene/kind
            if (target/'verification.json').exists():
                results.append(json.loads((target/'verification.json').read_text()));continue
            batch,m=_copy_first(source,target)
            if kind=='missing_frame':
                batch['start_qpos']=batch['start_qpos'][:-1]
            elif kind=='nonfinite':batch['start_qpos'][0,0]=np.nan
            elif kind=='time_gap':batch['time'][1,0]+=.01
            elif kind=='wrong_component':
                if not len(batch['contacts']):
                    batch['contacts']=np.zeros((1,21));batch['contact_range'][0]=[0,1]
                batch['contacts'][0,0]=1e6
            _write(target,batch,m)
            if kind=='geometry_identity':
                p=target/'raw/geometry.json';p.write_text(p.read_text()+' ')
            elif kind=='model_identity':
                p=target/'raw/model.mjb.gz'
                if not p.exists():p=target/'raw/model.mjb'
                p.write_bytes(p.read_bytes()+b'changed')
            verdict=analyze(target)
            row=dict(scene=scene,case=kind,expected='UNKNOWN',actual=verdict['primary_status'],
                     status='PASS' if verdict['primary_status']=='UNKNOWN' else 'FAIL',file=str(target.relative_to(root)))
            (target/'verification.json').write_text(json.dumps(row,indent=2)+'\n');results.append(row)
        # Numeric threshold boundaries are computational fixtures, never counted
        # as executed physical contacts or inserted into the 648-episode budget.
        from .limits import exceeds
        for value,expected in ((.999999,False),(1.,False),(1.000001,True)):
            result=exceeds(value,1.)
            results.append(dict(scene=scene,case='threshold_boundary',value=value,limit=1.,expected=expected,
                                actual=result,status='PASS' if result==expected else 'FAIL'))
    return results
