"""Task-only candidate input boundary. The interactive session is not context-blinded."""
from __future__ import annotations
import json,re,math
from pathlib import Path

TASK_PROMPT='''Write a Python skill for the Robosuite PickPlaceCan task using ASPIRE's public API.
Pick up the red soda can from its starting bin, carry it to the compartment
marked by the can silhouette in the destination bin, release it, and move away.
The public observation.task_context.language supplies a static fixture goal
can-center XYZ in robot0_base, after "Fixture goal can-center robot0_base XYZ: ".
Use public calibrated RGB-D images and robot state. Cameras are agentview and
robot0_eye_in_hand, 128 by 128 pixels. MotionStrategy(ik_solver="pyroki",
trajectory_planner="interpolation") is available. The budget is 1000 control
ticks at 20 Hz, including a terminal hold. Use import math, functions, loops and
public tool calls. Work from the supplied starting program and task feedback.
Return a complete, functionally different Python program that improves task
completion, placement alignment or motion efficiency, and a short task-based
rationale. Retain successful portions where appropriate.
'''
_STAGE=re.compile(r'^(approach|grasp_descent(?:_\d+)?|yaw_\d+|lift(?:_\d+)?|carry(?:_\d+)?|pre_place|place|retreat(?:_\d+)?|close|release) (True|False)(?: |$)')
_LOCAL=re.compile(r'^localized (can|destination) pixels (\d+) top_center (\[[^\]]+\])$')
PROHIBITED=re.compile(r'(?i)\b(?:safety|gate|esrb?|promot\w*|reject\w*|parent|violation|interlock|collision|forbidden)\b|安全|门控|晋升|回退|违规')


def validate_prompt(text):
    if PROHIBITED.search(text):raise ValueError('candidate input contains host-only terms')


def task_feedback(result,stdout):
    stages=[];observations=[]
    for line in stdout.splitlines():
        m=_STAGE.match(line)
        if m:stages.append({'stage':m[1],'completed':m[2]=='True'})
        m=_LOCAL.fullmatch(line)
        if m:
            try:xyz=json.loads(m[3])
            except (ValueError,TypeError):continue
            if isinstance(xyz,list) and len(xyz)==3 and all(type(x) in (float,int) and math.isfinite(x) for x in xyz):
                observations.append({'object':m[1],'rgbd_points':int(m[2]),'top_center_robot_base_m':xyz})
    ticks=result.get('control_steps')
    return {'task_success':result.get('native_task_success') is True,'program_completed':result.get('program_ok') is True,
            'control_ticks':ticks if type(ticks) is int and ticks>=0 else None,'motion_stages':stages,'public_rgbd_observations':observations}


def export_previous(folder):
    result=json.loads((folder/'execution/result.json').read_text())
    logs=list(folder.glob('execution/native/**/logs/program.stdout'))
    if len(logs)>1:raise ValueError('ambiguous task log')
    return task_feedback(result,logs[0].read_text() if logs else '')


def archive_inputs(folder,base_program,previous=None):
    feedback=export_previous(previous) if previous else None
    text=TASK_PROMPT+'\nStarting program:\n'+base_program+'\nPrevious task feedback:\n'+json.dumps(feedback,ensure_ascii=False)
    validate_prompt(text)
    (folder/'task_feedback.json').write_text(json.dumps(feedback,indent=2)+'\n')
    (folder/'candidate_prompt.txt').write_text(text)
    return text
