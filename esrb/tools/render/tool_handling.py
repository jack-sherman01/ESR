"""Render measured tool-handling and assembly previews for the README (requires Pillow and MuJoCo).

Usage: MUJOCO_GL=egl python -m esrb tools render tool-handling --out-dir assets
Tool previews add matched-time detail views with display-only lighting/texture changes.
"""


import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from esrb.oracles.multiconstraint import scenario_oracle

WIDTH, HEIGHT, STRIDE = 960, 720, 4


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def view_camera(task):
    camera = task._model.camera("overview").id
    if task.name == 'tool_handling':
        position, target = np.array([1.05, -1.10, 0.85]), np.array([0.0, -0.16, 0.28])
    else:
        position, target = np.array([0.24, -0.44, 0.32]), np.array([0.0, 0.0, 0.12])
    backward = position - target
    backward /= np.linalg.norm(backward)
    right = np.cross([0.0, 0.0, 1.0], backward)
    right /= np.linalg.norm(right)
    up = np.cross(backward, right)
    rotation = np.column_stack((right, up, backward))
    task._model.cam_pos[camera] = position
    mujoco.mju_mat2Quat(task._model.cam_quat[camera], rotation.ravel())
    mujoco.mj_forward(task._model, task._data)


def caption(frame, title, detail, color):
    image = Image.fromarray(frame)
    draw = ImageDraw.Draw(image)
    scale = WIDTH / 480
    draw.rectangle((0, 0, WIDTH, round(34*scale)), fill="#141820")
    draw.text((round(12*scale), round(7*scale)), title, font=font(round(17*scale)), fill=color)
    draw.rectangle((0, HEIGHT-round(32*scale), WIDTH, HEIGHT), fill="#141820")
    draw.text((round(12*scale), HEIGHT-round(25*scale)), detail, font=font(round(15*scale)), fill="#eef1f6")
    return image


def render_assembly_animation(out, seed, duration=5.):
    """Measured assembly previews using the shared single-view workcell renderer."""
    from esrb.tools.render.focused_workcells import render_controlled
    render_controlled('long_horizon_assembly', out, seed, WIDTH, HEIGHT, duration)


def render_assembly_witness(witness_dir, out):
    """Render the final measured states of the current three-piece witness."""
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.oracles.production import validate_production
    from esrb.utils.hashing import hash_dict
    provenance=[]
    for role in ('parent','candidate'):
        saved=json.loads((witness_dir/(role+'.json')).read_text())
        trace=saved['trace']; validate_production(trace)
        if hash_dict({k:v for k,v in trace.items() if k!='seed'})!=saved['trace_digest']:
            raise ValueError('assembly witness digest mismatch')
        task=LongHorizonAssemblyTask()
        try:
            task.reset(trace['seed'],disturbance=trace['reset_readback'])
            state=trace['physics_samples'][-1]
            task._data.qpos[:]=state['qpos']; task._data.qvel[:]=state['qvel']
            task._data.time=state['time_s']
            mujoco.mj_forward(task._model,task._data)
            filename='long_horizon_assembly'+('_failure' if role=='candidate' else '')+'.png'
            Image.fromarray(task.render_frame(WIDTH,HEIGHT)).save(out/filename)
            provenance.append({'role':role,'file':filename,'trace_digest':saved['trace_digest'],
                               'model_digest':trace['model_digest'],'step':trace['control_step_count']})
        finally: task.close()
    (out/'long_horizon_assembly_preview.json').write_text(json.dumps({'scope':'measured_three_piece_witness','states':provenance,'inset':False},indent=2)+'\n')


def main(args=None):
    global WIDTH, HEIGHT
    from esrb.cli.options import options_tools_render_tool_handling
    parser = argparse.ArgumentParser()
    options_tools_render_tool_handling(parser)
    args = parser.parse_args() if args is None else args
    if args.width < 320 or args.height < 240 or not np.isfinite(args.duration) or args.duration < 1:
        parser.error('render size must be at least 320x240 and duration must be at least 1 second')
    WIDTH, HEIGHT = args.width, args.height
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.assembly:
        render_assembly_animation(args.out_dir, args.seed, args.duration)
        return
    if args.assembly_witness is not None:
        render_assembly_witness(args.assembly_witness, args.out_dir)
        return
    from esrb.tools.render.tool_workcell import capture_pair, render_pair
    metadata = render_pair(capture_pair(args.seed), args.out_dir, WIDTH, HEIGHT, args.duration)
    print(json.dumps({"out_dir": str(args.out_dir), "synchronized_comparison": True,
                      "demos": [{key: demo[key] for key in
                                 ("file", "task_success", "safe", "frames", "playback_duration_s")}
                                for demo in metadata["demos"]]}, indent=2))
