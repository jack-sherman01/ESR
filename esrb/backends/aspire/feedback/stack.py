"""Export only task observations and motion outcomes for Stack skill generation.

Raw executor errors and host admission data are deliberately excluded. The
export is an input boundary, not a claim that the interactive assistant has
forgotten prior conversation context.
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import re

TASK_PROMPT = '''Write a Python skill for the Robosuite Stack task using ASPIRE's public API.
Pick up the red cube, move it over the green cube, place it on top, release it,
and move the gripper away so the stack remains assembled.
Use public calibrated RGB-D observations and robot state. The available motion
strategy is MotionStrategy(ik_solver="pyroki", trajectory_planner="interpolation").
Keep the program within 1000 control ticks at 20 Hz, including the executor's
terminal hold. Cameras are agentview and robot0_eye_in_hand at 128 by 128 pixels.
Work from the supplied base program and previous task feedback. Produce a
functionally different version that improves completion, alignment or execution
efficiency. Explain the change using task or motion feedback. Keep successful
parts of the program where appropriate. Return executable Python source and a
short rationale. You may use import math, public tool calls, functions and loops.
'''

_STAGE = re.compile(r'^(approach|grasp_descent(?:_\d+)?|yaw_at_clearance_\d+|lift|carry(?:_\d+)?|pre_place|place|clear_stack|retreat|close|release) (True|False)(?: |$)')
_LOCALIZED = re.compile(r'^localized (red|green) pixels (\d+) top_center (\[[^\]]+\])$')


def task_feedback(result: dict, stdout: str) -> dict:
    """Whitelist data; never copy error text or arbitrary program output."""
    stages, observations = [], []
    for line in stdout.splitlines():
        match = _STAGE.match(line)
        if match:
            stages.append({'stage': match[1], 'completed': match[2] == 'True'})
        match = _LOCALIZED.fullmatch(line)
        if match:
            try:
                xyz = json.loads(match[3])
            except (ValueError, TypeError):
                continue
            if (isinstance(xyz, list) and len(xyz) == 3
                    and all(type(x) in (int, float) and math.isfinite(x) for x in xyz)):
                observations.append({'object': match[1], 'rgbd_points': int(match[2]),
                                     'top_center_robot_base_m': xyz})
    ticks = result.get('control_steps')
    if type(ticks) is not int or ticks < 0:
        ticks = None
    return {'task_success': result.get('native_task_success') is True,
            'program_completed': result.get('program_ok') is True,
            'control_ticks': ticks, 'motion_stages': stages,
            'public_rgbd_observations': observations}


def export_previous(folder: Path) -> dict:
    result = json.loads((folder/'execution/result.json').read_text())
    logs = list(folder.glob('execution/native/**/logs/program.stdout'))
    if len(logs) > 1:
        raise ValueError('ambiguous task log')
    stdout = logs[0].read_text() if logs else ''
    return task_feedback(result, stdout)
