"""Render measured tool-handling demos for the README (requires Pillow and MuJoCo).

Usage: MUJOCO_GL=egl python scripts/render_tool_handling.py --out-dir assets
Only the viewing camera changes; both demos use the task's physical dynamics.
"""
import argparse
import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from esrb.agent.tool_handling_policy import ToolHandlingPolicy
from esrb.oracles.multiconstraint import scenario_oracle
from esrb.tasks.tool_handling import ToolHandlingTask
from esrb.update_operators.tool_grip import ToolGripRelaxationOperator

WIDTH, HEIGHT, STRIDE = 480, 360, 4


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default()


def view_camera(task):
    camera = task._model.camera("overview").id
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
    draw.rectangle((0, 0, WIDTH, 34), fill="#141820")
    draw.text((12, 7), title, font=font(17), fill=color)
    draw.rectangle((0, HEIGHT-32, WIDTH, HEIGHT), fill="#141820")
    draw.text((12, HEIGHT-25), detail, font=font(15), fill="#eef1f6")
    return image


def render(out, seed, relaxed):
    task, policy = ToolHandlingTask(), ToolHandlingPolicy()
    if relaxed:
        policy = ToolGripRelaxationOperator().apply(policy, seed, 0.5).candidate_policy
    frames, snapshots = [], {}
    try:
        observation = task.reset(seed)
        view_camera(task)
        for step in range(600):
            stage = observation.info["stage"]
            observation, _, done, trace = task.step(policy.act(observation))
            if step % STRIDE == 0 or done:
                frame = task.render_frame(WIDTH, HEIGHT)
                dropped = bool(trace["drop_events"])
                title = "Relaxed transport grip" if relaxed else "Baseline: friction-held tool"
                detail = f"{stage.upper()}  |  t={trace['times_s'][-1]:.2f}s"
                if dropped:
                    detail = f"TOOL DROPPED  |  t={trace['times_s'][-1]:.2f}s"
                if done and trace["success"]:
                    detail = "OPERATION COMPLETE  |  aligned surface contact"
                labeled = caption(frame, title, detail, "#ffb17a" if relaxed else "#8fe4bb")
                frames.append(labeled)
                snapshots["drop" if dropped else stage] = labeled.copy()
            if done or (trace["drop_events"] and step+1 >= trace["drop_events"][0]["step"]+30):
                break
        assessment = scenario_oracle("tool_handling").evaluate(trace, 0)
        assert assessment.safe is (not relaxed)
        assert task.evaluate(trace).success is (not relaxed)
        assert bool(trace["drop_events"]) is relaxed
        path = out / ("tool_handling_drop.gif" if relaxed else "tool_handling.gif")
        indexed = [image.quantize(colors=128, dither=Image.Dither.NONE) for image in frames]
        durations = [80]*len(indexed)
        durations[-1] = 1200
        indexed[0].save(path, save_all=True, append_images=indexed[1:], duration=durations, loop=0, optimize=True)
        return snapshots, {
            "file": path.name, "seed": seed, "frames": len(frames),
            "transport_closure_m": policy.transport_closure,
            "task_success": task.evaluate(trace).success, "safe": bool(assessment.safe),
            "signals": assessment.signal_values, "drop_events": trace["drop_events"],
            "stage_events": trace["stage_events"], "model_digest": trace["model_digest"],
        }
    finally:
        task.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "assets")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    baseline, base_info = render(args.out_dir, args.seed, False)
    candidate, child_info = render(args.out_dir, args.seed, True)
    panels = [(baseline["grasp"], "1. GRASP"), (baseline["transport"], "2. TRANSPORT"),
              (baseline["operate"], "3. ORIENT + OPERATE"), (candidate["drop"], "4. RELAXED GRIP: DROP")]
    overview = Image.new("RGB", (WIDTH*2, (HEIGHT+36)*2), "#141820")
    for index, (panel, label) in enumerate(panels):
        x, y = index % 2 * WIDTH, index // 2 * (HEIGHT+36)
        overview.paste(panel, (x, y+36))
        ImageDraw.Draw(overview).text((x+12, y+8), label, font=font(18), fill="#eef1f6")
    overview.save(args.out_dir / "tool_handling_stages.png")
    metadata = {"scope": "controlled_engineering_visualization", "camera_only_override": True,
                "demos": [base_info, child_info]}
    (args.out_dir / "tool_handling_preview.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
