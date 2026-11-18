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

import argparse
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

def feedback(folder):
 result=read(folder/'execution/result.json')
 data={'task_success':result.get('native_task_success') is True,
       'program_completed':result.get('program_ok') is True,
       'control_ticks':result.get('control_steps'),'tool_results':[]}
 native=Path(result['native_run']) if result.get('native_run') else None
 if native is None:return data
 path=native/'source/program-result.json'
 if path.exists():
  raw=read(path)
  for call in raw.get('calls',[]):
   tool=call.get('tool');summary=call.get('outputs_summary') or {}
   if tool not in {'move_to_pose','move_to_joints','open_gripper','close_gripper','set_gripper','execute_trajectory','step','solve_ik','mask_to_point_cloud','get_observation','get_robot_state'}:continue
   row={'tool':tool,'returned':call.get('ok') is True}
   for key in ['ok','steps_executed','point_count']:
    value=summary.get(key)
    if type(value) in (bool,int):row[key]=value
   # Standard interpreter diagnostics help repair programs without forwarding host errors.
   name=call.get('error_type');message=call.get('error_message')
   if name in {'NameError','AttributeError','TypeError','ValueError','IndexError','KeyError'} and isinstance(message,str):
    if not re.search(r'esrb|safety|gate|violation|interlock|forbidden|promot|reject',message,re.I):row['python_error']={'type':name,'message':message[:250]}
   data['tool_results'].append(row)
  error=raw.get('error')
  if isinstance(error,str) and not re.search(r'esrb|safety|gate|violation|interlock|forbidden|promot|reject',error,re.I):data['program_error']=error[:300]
 # Export public end-effector observations, never native object state or contact data.
 poses=[]
 for p in sorted((native/'trace/calls').glob('*/output.json')):
  output=read(p)
  state=output if output.get('$type')=='RobotState' else (output.get('final_observation') or {}).get('robot_state')
  if not isinstance(state,dict):continue
  pose=(state.get('end_effector_poses') or {}).get('primary') or {}
  xyz=array(pose.get('position'),3)
  if xyz is None:continue
  item={'eef_xyz_m':xyz}
  jaw=(state.get('gripper_positions') or {}).get('primary')
  if finite(jaw):item['jaw_open_fraction']=jaw
  poses.append(item)
 data['public_motion_states']=poses[-25:]
 # Only explicit, structured program observations are accepted; all other stdout remains host-side.
 stdout=native/'logs/program.stdout';observations=[]
 if stdout.exists():
  for line in stdout.read_text().splitlines():
   if not line.startswith('OBS '):continue
   try:value=json.loads(line[4:])
   except (ValueError,TypeError):continue
   if isinstance(value,list) and len(value)<=12 and all(finite(v) for v in value):observations.append(value)
 data['program_observations']=observations[-20:]
 return data


def wait_file(path,timeout=3600):
 start=time.monotonic()
 while not path.exists():
  if time.monotonic()-start>timeout:raise TimeoutError(f'waiting for {path.name}')
  time.sleep(.25)


def drive(root,number):
 protocol=read(root/'protocol.json');task=protocol['task']
 prefix='pickup_can' if task=='pickup_can' else 'wipe'
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
  if sha(capsule/'API.txt')!=api_hash:raise ValueError('API input changed during round')
  generation=folder/f'generation_{version:02d}';generation.mkdir(exist_ok=False)
  initial={'task_prompt':assigned['task_prompt'],'api':(capsule/'API.txt').read_text(),'iteration':version}
  if version==1:
   initial.update(starting_program=None,previous_task_feedback=None)
  else:
   initial.update(starting_program=(capsule/f'starting_v{version:02d}.py').read_text(),previous_task_feedback=read(capsule/f'feedback_v{version-1:02d}.json'))
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
  entry=REPO/'scripts'/'runners'/('run_aspire_pickup_can_batch.py' if task=='pickup_can' else 'run_aspire_wipe_batch.py')
  with (candidate/'execution.log').open('w') as log:
   result=subprocess.run([str(PYTHON),'-B',str(entry),'episode','--root',str(root),'--round',str(number),'--version',str(version)],cwd=REPO,env=env,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode or not (candidate/'gate_decision.json').exists():raise RuntimeError(f'episode process failed {task} {number} {version}; see execution.log')
  if source.read_bytes()!=original:raise ValueError('submitted program changed during execution')
  public=feedback(candidate);save(generation/'task_feedback.json',public)
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
