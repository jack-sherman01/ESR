"""Check ASPIRE's real simulator; this is NOT an agent or task-success result.

Run with the ASPIRE LIBERO interpreter; see scripts/setup_aspire.sh.
No LLM calls, perception services, policy training, or ESRB safety claims.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import traceback

import numpy as np

from cap_harness.contracts import ArmCommand, RobotAction
from cap_harness.libero.adapter import LiberoAdapter


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--checkout", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()
    # Never overwrite evidence from an earlier attempt.
    args.out.mkdir(parents=True, exist_ok=False)
    root = Path(args.checkout).resolve()
    result = {
        "kind": "simulator_structural_smoke",
        "agent_called": False,
        "task_success_evaluated": False,
        "esrb_safety_evaluated": False,
        "suite": "libero_object_swap", "task_id": 0, "seed": args.seed,
        "init_mode": "saved", "status": "failed",
        "source_commit": subprocess.check_output(
            ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
        ).strip(),
        "uv_lock_sha256": hashlib.sha256((root / "uv.lock").read_bytes()).hexdigest(),
        "source_diff_sha256": hashlib.sha256(subprocess.check_output(
            ["git", "-C", str(root), "diff", "--binary", "HEAD"]
        )).hexdigest(),
        "local_patch": "integrations/aspire/aspire-local.patch",
        "versions": {name: importlib.metadata.version(name)
                     for name in ("mujoco", "numpy", "torch", "cap-harness")},
    }
    adapter = LiberoAdapter(camera_height=128, camera_width=128)
    try:
        obs = adapter.reset("libero_object_swap:0", args.seed)
        from OpenGL.GL import GL_RENDERER, glGetString
        renderer = glGetString(GL_RENDERER)
        result["gl_renderer"] = renderer.decode() if renderer else None
        q0 = obs.robot_state.joint_positions["primary"].copy()
        result["camera_names"] = list(obs.cameras)
        for name, camera in obs.cameras.items():
            from PIL import Image
            Image.fromarray(camera.rgb).save(args.out / f"{name}.png")
            if not np.all(np.isfinite(camera.depth_m)):
                raise RuntimeError(f"nonfinite depth: {name}")
        records = []
        for i in range(3):
            target = q0.copy()
            if i > 0:
                target[0] += 0.02
            before = obs.timestamp_s
            step = adapter.step(RobotAction(arms={"primary": ArmCommand(
                mode="joint_position", target=target,
                gripper_position=obs.robot_state.gripper_positions["primary"],
                embodiment="libero",
            )}))
            if not step.ok or step.observation is None:
                raise RuntimeError(f"step failed: {step.error}")
            obs = step.observation
            if before is None or obs.timestamp_s is None:
                raise RuntimeError("missing simulator timestamp")
            dt = obs.timestamp_s - before
            if not np.isclose(dt, 0.05, atol=1e-6, rtol=0):
                raise RuntimeError(f"wrong simulator timestep: {dt}")
            if step.terminated or step.truncated:
                raise RuntimeError("episode ended during structural smoke")
            records.append({"step": i, "dt_s": dt,
                            "joint_positions": obs.robot_state.joint_positions["primary"].tolist()})
        delta = float(np.max(np.abs(obs.robot_state.joint_positions["primary"] - q0)))
        if delta <= 1e-6:
            raise RuntimeError("command did not move the robot")
        result.update(status="passed", steps=records, max_joint_delta_rad=delta)
    except Exception:
        result["error"] = traceback.format_exc()
    finally:
        try:
            adapter.close()
        except Exception:
            result["close_error"] = traceback.format_exc()
            result["status"] = "failed"
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
