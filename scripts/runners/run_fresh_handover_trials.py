"""Host-only bridge between fresh per-round skill workers and instrumented trials.

Workers exchange task-only files in isolated input directories. No conversation
history, prior experiment program or host admission result is supplied to them.
"""
from __future__ import annotations

# Allow direct execution from this subdirectory as well as package imports.
if __package__ in (None, ""):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import argparse,ast
from concurrent.futures import ThreadPoolExecutor,as_completed
from datetime import datetime,timezone
import hashlib,importlib,json,math,os,re,shutil,subprocess,sys,time,traceback
from pathlib import Path

REPO=Path(__file__).resolve().parents[2]
PYTHON=REPO/'.esr-local/aspire/checkout/.venv-robosuite/bin/python'

def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def save(p,x):
 p=Path(p);tmp=p.with_name(p.name+'.tmp');tmp.write_text(json.dumps(x,indent=2,ensure_ascii=False,allow_nan=False)+'\n');tmp.replace(p)
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def array(x,n=None):
 if isinstance(x,dict):x=x.get('$array_inline')
 return x if isinstance(x,list) and (n is None or len(x)==n) and all(finite(v) for v in x) else None

from scripts.feedback.handover_task_feedback import feedback,validate_prompt


def wait_file(path,timeout=3600):
 start=time.monotonic()
 while not path.exists():
  if time.monotonic()-start>timeout:raise TimeoutError(f'waiting for {path.name}')
  time.sleep(.25)


def drive(root,number):
 protocol=read(root/'protocol.json');task=protocol['task']
 prefix='handover'
 state=importlib.import_module('scripts.gates.' + prefix + '_gate_state')
 folder=root/f'round_{number:02d}'
 assignment=folder/'agent_assignment.json';wait_file(assignment)
 assigned=read(assignment);capsule=Path(assigned['capsule'])
 api_hash=sha(capsule/'API.txt')
 for version in range(1,6):
  source=capsule/f'v{version:02d}.py';ready=capsule/f'ready_v{version:02d}.json'
  wait_file(ready);wait_file(source)
  # A program written before its ready marker is an immutable submission.
  original=source.read_bytes();rationale=read(ready).get('rationale','task skill revision')
  validation_code='import ast,sys;from cap_harness.runtime import _ProgramValidator;_ProgramValidator().visit(ast.parse(open(sys.argv[1]).read()))'
  attempt=0
  while True:
   checked=subprocess.run([str(PYTHON),'-B','-c',validation_code,str(source)],capture_output=True,text=True,cwd=REPO)
   previous_programs=list(folder.glob('version_*/program.py'))
   duplicate=any(ast.dump(ast.parse(p.read_text()))==ast.dump(ast.parse(source.read_text())) for p in previous_programs) if checked.returncode==0 else False
   if checked.returncode==0 and not duplicate:break
   attempt+=1
   draft=folder/f'nonexecuted_v{version:02d}_syntax_{attempt:02d}';draft.with_suffix('.py').write_bytes(original)
   reason='Submit a different task program; this program repeats an earlier measured version.' if duplicate else (checked.stderr.splitlines()[-1] if checked.stderr.splitlines() else 'Python validation failed')
   validate_prompt(reason)
   save(draft.with_suffix('.json'),{'status':'NOT_EXECUTED','python_validation':reason})
   ready.unlink();save(capsule/f'validation_v{version:02d}.json',{'status':'CORRECTION_REQUIRED','python_validation':reason})
   wait_file(ready);original=source.read_bytes();rationale=read(ready).get('rationale','task skill revision')
  (capsule/f'validation_v{version:02d}.json').unlink(missing_ok=True)
  validate_prompt(rationale)
  if sha(capsule/'API.txt')!=api_hash:raise ValueError('API input changed during round')
  generation=folder/f'generation_{version:02d}';generation.mkdir(exist_ok=False)
  initial={'task_prompt':assigned['task_prompt'],'api':(capsule/'API.txt').read_text(),'iteration':version}
  if version==1:
   initial.update(starting_program=None,previous_task_feedback=None)
  else:
   initial.update(starting_program=(capsule/f'starting_v{version:02d}.py').read_text(),previous_task_feedback=read(capsule/f'feedback_v{version-1:02d}.json'))
  validate_prompt(json.dumps(initial,ensure_ascii=False))
  save(generation/'agent_input.json',initial)
  (generation/'agent_output.py').write_bytes(original)
  save(generation/'agent_output.json',{'rationale':rationale,'source_sha256':sha(source)})
  state.load_rules(root)
  with state.round_lock(folder):
   chain=state.load_chain(root,number)
   if version!=len(chain['events'])+1:raise ValueError('version ordering mismatch')
   active=chain['active'];candidate=folder/f'version_{version:02d}';candidate.mkdir(exist_ok=False)
   (candidate/'program.py').write_bytes(original)
   (candidate/'candidate_prompt.txt').write_text(initial['task_prompt']+'\n'+initial['api'])
   save(candidate/'task_feedback.json',initial['previous_task_feedback'])
   base={'version':active['version'],'program_sha256':active['program_sha256']} if active else None
   save(candidate/'proposal.json',{'round':number,'version':version,'gate_base':base,'source_sha256':sha(source),
        'rationale':rationale,'kind':'candidate','candidate_prompt_sha256':sha(candidate/'candidate_prompt.txt'),
        'task_feedback_sha256':sha(candidate/'task_feedback.json'),'parent_program':active['program_path'] if active else None})
   save(candidate/'execution_settings.json',{'renderer':'egl','gpu_device_id':(number-1)%4,'camera_sampling_hz':5,'fresh_explicit_observation':True})
  save(candidate/'generation.json',{'agent_id':assigned['agent_id'],'fork_turns':'none',
       'initial_program_provided':False,'api_sha256':api_hash,
       'agent_input_sha256':sha(generation/'agent_input.json'),'agent_input_path':str((generation/'agent_input.json').relative_to(root)),
       'controlled_injection':False,'rationale':rationale})
  env=os.environ.copy();env.update(MUJOCO_GL='egl',PYOPENGL_PLATFORM='egl',EGL_PLATFORM='device',MUJOCO_EGL_DEVICE_ID=str((number-1)%4),
      LP_NUM_THREADS='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',NUMEXPR_NUM_THREADS='1',PYTHONDONTWRITEBYTECODE='1')
  entry=REPO/'scripts'/'runners'/'run_aspire_handover_batch.py'
  with (candidate/'execution.log').open('w') as log:
   result=subprocess.run([str(PYTHON),'-B',str(entry),'episode','--root',str(root),'--round',str(number),'--version',str(version)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode or not (candidate/'gate_decision.json').exists():raise RuntimeError(f'episode process failed {task} {number} {version}; see execution.log')
  if source.read_bytes()!=original:raise ValueError('submitted program changed during execution')
  public=feedback(candidate);validate_prompt(json.dumps(public,ensure_ascii=False));save(generation/'task_feedback.json',public)
  if version<5:
   chain=state.load_chain(root,number);active=chain['active']
   base=root/active['program_path'] if active else candidate/'program.py'
   (capsule/f'starting_v{version+1:02d}.py').write_bytes(base.read_bytes())
   save(generation/'host_provenance.json',{'starting_next_sha256':sha(base),'active_version':(active or {}).get('version')})
  save(capsule/f'feedback_v{version:02d}.json',public)
  print(json.dumps({'task':task,'round':number,'version':version,'task_success':public['task_success'],'measured':True}),flush=True)
 save(folder/'fresh_agent_complete.json',{'status':'COMPLETE','agent_id':assigned['agent_id'],'measured':5,'fork_turns':'none'})
 return number


def main():
 p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--rounds',type=int,nargs='+');p.add_argument('--workers',type=int,default=10);args=p.parse_args();root=args.root.resolve()
 numbers=args.rounds or list(range(1,len(read(root/'protocol.json')['seeds'])+1))
 with ThreadPoolExecutor(max_workers=args.workers) as pool:
  for future in as_completed([pool.submit(drive,root,n) for n in numbers]):
   print(json.dumps({'completed_round':future.result()}),flush=True)
if __name__=='__main__':main()
