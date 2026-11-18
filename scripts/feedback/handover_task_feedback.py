"""Task-only public feedback for fresh handover workers."""
from pathlib import Path
import json,hashlib,math,re

def read(p):return json.loads(Path(p).read_text())
def finite(x):return type(x) in (int,float) and math.isfinite(x)
def array(x,n=None):
 if isinstance(x,dict):x=x.get('$array_inline')
 return x if isinstance(x,list) and (n is None or len(x)==n) and all(finite(v) for v in x) else None
TASK_PROMPT='完成 TwoArmHandover，把锤子从 primary 交给 secondary，并尽量减少控制步数。'
PROHIBITED=re.compile(r'(?i)\b(?:gate|esrb?|promot\w*|reject\w*|interlock|violation|rollback|parent)\b|门控|晋升|回退|违规')
def validate_prompt(text):
 if PROHIBITED.search(text):raise ValueError('host admission information in candidate input')
def archive_inputs(folder,base_program,previous=None):
 data=feedback(previous) if previous else None
 (folder/'candidate_prompt.txt').write_text(TASK_PROMPT+'\n'+base_program)
 (folder/'task_feedback.json').write_text(json.dumps(data,indent=2)+'\n')

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
   if tool not in {'move_to_pose','move_to_joints','open_gripper','close_gripper','set_gripper','set_grippers','plan_motion','execute_trajectory','step','solve_ik','mask_to_point_cloud','get_observation','get_robot_state'}:continue
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
  item={'time_s':state.get('timestamp_s'),'arms':{}}
  for arm,pose in (state.get('end_effector_poses') or {}).items():
   xyz=array(pose.get('position'),3)
   if xyz is None:continue
   part={'eef_xyz_m':xyz}
   jaw=(state.get('gripper_positions') or {}).get(arm)
   if finite(jaw):part['jaw_open_fraction']=jaw
   item['arms'][arm]=part
  if item['arms']:poses.append(item)
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
