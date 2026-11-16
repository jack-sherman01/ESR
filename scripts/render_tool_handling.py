"""Render measured tool-handling and assembly previews for the README (requires Pillow and MuJoCo).

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


def render_assembly_animation(out, seed):
    """Run both continuous assembly policies and render every stage to GIFs."""
    from esrb.agent.assembly_policy import AssemblyPolicy
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.update_operators.assembly_orientation import AssemblyOrientationOperator
    from esrb.utils.hashing import hash_dict

    demos = []
    for relaxed in (False, True):
        task, policy = LongHorizonAssemblyTask(), AssemblyPolicy()
        if relaxed:
            policy = AssemblyOrientationOperator().apply(policy, seed, 0.5).candidate_policy
        frames, sampled_steps, sample_times = [], [], []
        try:
            observation = task.reset(seed)
            view_camera(task)
            done = False
            for step in range(task.MAX_STEPS):
                observation, _, done, trace = task.step(policy.act(observation))
                if step % STRIDE == 0 or done:
                    stage = trace["stages"][-1].upper()
                    title = "Carry yaw error: insertion failure" if relaxed else "Baseline: continuous assembly"
                    depth = (f"{trace['insertion_depth_m'][-1]*1000:.1f} mm"
                             if stage in {"INSERT", "VERIFY"} and trace["assembly_xy_error_m"][-1] <= 0.004 else "N/A")
                    detail = f"{stage} | t={trace['times_s'][-1]:.2f}s | depth {depth}"
                    if done:
                        detail = ("INSERT FAILED" if relaxed else "ASSEMBLY VERIFIED") + f" | depth {trace['insertion_depth_m'][-1]*1000:.2f} mm"
                    frames.append(caption(task.render_frame(WIDTH, HEIGHT), title, detail,
                                          "#ffb17a" if relaxed else "#8fe4bb"))
                    sampled_steps.append(step+1)
                    sample_times.append(trace["times_s"][-1])
                if done:
                    break
            if not done:
                raise RuntimeError("assembly visualization exceeded its episode budget")
            assessment = scenario_oracle(task.name).evaluate(trace, 0)
            completed = task.evaluate(trace).success
            records = {row["stage"]: row for row in trace["stage_records"]}
            if relaxed:
                valid = (not completed and assessment.status == "VIOLATION"
                         and records.get("transport", {}).get("status") == "completed"
                         and records.get("insert", {}).get("status") == "failed")
            else:
                valid = completed and assessment.safe and trace["verification_hold_s"][-1] >= 0.4
            if not valid:
                raise RuntimeError("rendered assembly outcome did not match the measured demo conditions")
            name = "long_horizon_assembly_failure.gif" if relaxed else "long_horizon_assembly.gif"
            indexed = [frame.quantize(colors=128, dither=Image.Dither.NONE) for frame in frames]
            durations = [round(1000*(b-a)) for a,b in zip(sample_times,sample_times[1:])] + [1200]
            indexed[0].save(out/name, save_all=True, append_images=indexed[1:],
                            duration=durations, loop=0, optimize=True)
            demos.append({
                "file": name, "seed": seed, "frames": len(frames), "sampled_steps": sampled_steps,
                "policy": policy.state_dict(), "disturbance": trace["reset_readback"],
                "task_success": bool(completed), "safe": bool(assessment.safe),
                "failure_reason": trace["failure_reason"], "stage_events": trace["stage_events"],
                "stage_reports": assessment.stage_reports,
                "final_depth_m": trace["insertion_depth_m"][-1],
                "stable_hold_s": trace["verification_hold_s"][-1],
                "simulation_steps": len(trace["stages"]), "simulation_time_s": trace["times_s"][-1],
                "trace_digest": hash_dict(trace), "model_digest": trace["model_digest"],
            })
        finally:
            task.close()
    metadata = {"scope": "controlled_engineering_visualization", "camera_only_override": True,
                "size_px": [WIDTH, HEIGHT], "capture_stride": STRIDE,
                "final_frame_pause_ms": 1200, "demos": demos}
    (out/"long_horizon_assembly_animation.json").write_text(json.dumps(metadata,indent=2)+"\n")
    print(json.dumps({"files": [demo["file"] for demo in demos],
                      "outcomes": [{key: row[key] for key in ("task_success", "safe", "failure_reason", "frames")}
                                   for row in demos]}, indent=2))


def render_assembly_witness(witness_dir, out):
    """Render measured completed-assembly and insertion-entry witness states."""
    from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
    from esrb.oracles.assembly import validate_assembly_measurements
    from esrb.utils.hashing import hash_dict

    panels, provenance = [], []
    for label in ("parent", "candidate"):
        saved = json.loads((witness_dir/f"{label}.json").read_text())
        trace = saved["trace"]
        validate_assembly_measurements(trace)
        if hash_dict(trace) != saved["trace_digest"]:
            raise ValueError("assembly preview witness digest mismatch")
        task = LongHorizonAssemblyTask()
        try:
            task.reset(disturbance=trace["reset_readback"])
            if label == "parent":
                state, step = trace["stage_records"][-1]["exit_state"], len(trace["stages"])
            else:
                insertion = next(row for row in trace["stage_records"] if row["stage"] == "insert")
                state, step = insertion["entry_state"], insertion["enter_step"]
            task._data.qpos[:] = state["qpos"]
            task._data.qvel[:] = state["qvel"]
            view_camera(task)
            title = "Baseline: released assembly" if label == "parent" else "Carry yaw error: insertion entry"
            detail = f"Depth {trace['insertion_depth_m'][step-1]*1000:.2f} mm | yaw {trace['assembly_yaw_error_rad'][step-1]:.3f} rad"
            panels.append(caption(task.render_frame(WIDTH, HEIGHT), title, detail,
                                  "#8fe4bb" if label == "parent" else "#ffb17a"))
            provenance.append({"role": label, "trace_digest": saved["trace_digest"],
                               "model_digest": trace["model_digest"], "step": step})
        finally:
            task.close()
    image = Image.new("RGB", (WIDTH*2, HEIGHT))
    for number, panel in enumerate(panels):
        image.paste(panel, (number*WIDTH, 0))
    image.save(out/"long_horizon_assembly.png")
    (out/"long_horizon_assembly_preview.json").write_text(json.dumps({
        "scope": "measured_assembly_witness_states", "camera_only_override": True,
        "states": provenance,
    }, indent=2)+"\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=Path(__file__).resolve().parent.parent / "assets")
    parser.add_argument("--seed", type=int, default=0)
    assembly_mode = parser.add_mutually_exclusive_group()
    assembly_mode.add_argument("--assembly", action="store_true", help="run and animate the complete continuous assembly demos")
    assembly_mode.add_argument("--assembly-witness", type=Path, help="render saved continuous-assembly parent/candidate states")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    if args.assembly:
        render_assembly_animation(args.out_dir, args.seed)
        return
    if args.assembly_witness is not None:
        render_assembly_witness(args.assembly_witness, args.out_dir)
        return
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
