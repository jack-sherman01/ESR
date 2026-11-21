"""Task/public-API feedback only; host rule decisions never enter worker inputs."""
import json
from pathlib import Path
import math
import re
from .io import read

BLOCKED = re.compile(r'esrb|\bsafety\b|\bgate\b|violation|interlock|forbidden', re.I)
ALLOWED = {'NameError', 'ValueError', 'TypeError', 'AttributeError', 'IndexError', 'KeyError', 'ZeroDivisionError'}
ALLOWED_CODES = {'invalid_request','unsupported','planning_failed','ik_failed','timeout','execution_failed','service_unavailable','not_found','provider_failed'}

def finite(value):
    return type(value) in (int, float) and math.isfinite(value)

def array(value, n=None):
    if isinstance(value, dict):
        value = value.get('$array_inline')
    return value if isinstance(value,list) and (n is None or len(value)==n) and all(finite(v) for v in value) else None

def extract(native):
 path=native/'source/program-result.json'
 if not path.exists():return None
 raw=json.loads(path.read_text());error=raw.get('error')
 if not isinstance(error,dict):return None
 message=error.get('message','');kind=(error.get('details') or {}).get('exception_type')
 if BLOCKED.search(message):return None
 if error.get('code')=='invalid_request' and message.startswith('program rejected:'):
  return {'program_api_error':message.removeprefix('program rejected:').strip()[:350]}
 if kind in ALLOWED and error.get('code')=='execution_failed':
  return {'python_error':{'type':kind,'message':message[:350]}}
 return None

def task_errors(native):
 data=extract(native) or {'program_api_error':None}
 errors=[]
 for p in sorted((native/'trace/calls').glob('*/output.json')):
  output=json.loads(p.read_text());error=output.get('error')
  if not isinstance(error,dict):continue
  code=error.get('code');message=error.get('message','')
  if code not in ALLOWED_CODES or not isinstance(message,str) or BLOCKED.search(message):continue
  errors.append({'call':p.parent.name,'code':code,'message':message[:250]})
 if errors:data['public_tool_errors']=errors[-16:]
 return data

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
