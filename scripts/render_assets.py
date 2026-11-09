"""
Render GIF previews and static screenshots for all ESRB task environments.
Outputs go to assets/ at the repo root.

Usage:
    python scripts/render_assets.py
"""
import os
os.environ.setdefault("MUJOCO_GL", "osmesa")

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
from PIL import Image, ImageDraw, ImageFont

ASSETS = Path(__file__).resolve().parent.parent / "assets"
ASSETS.mkdir(exist_ok=True)

W, H = 480, 360
GIF_FPS = 25
LABEL_H = 28


def _add_label(frame: np.ndarray, text: str) -> np.ndarray:
    img = Image.fromarray(frame)
    draw = ImageDraw.Draw(img)
    # dark bar at bottom
    draw.rectangle([(0, H - LABEL_H), (W, H)], fill=(20, 20, 20))
    draw.text((8, H - LABEL_H + 5), text, fill=(240, 240, 240))
    return np.array(img)


def _save_gif(frames, path, fps=GIF_FPS):
    pil_frames = [Image.fromarray(f) for f in frames]
    pil_frames[0].save(
        path,
        save_all=True,
        append_images=pil_frames[1:],
        loop=0,
        duration=int(1000 / fps),
        optimize=False,
    )
    print(f"  saved {path}  ({len(frames)} frames)")


def _save_png(frame, path):
    Image.fromarray(frame).save(path)
    print(f"  saved {path}")


# ── 1. Peg Insertion ──────────────────────────────────────────────────────────
def render_peg_insertion():
    from esrb.tasks.contact_rich import PegInsertionTask
    from esrb.agent import ScriptedInsertionPolicy

    print("Rendering: peg_insertion …")
    task   = PegInsertionTask()
    policy = ScriptedInsertionPolicy()
    obs    = task.reset(seed=0)
    frames = []
    for i in range(120):
        frame = task.render_frame(W, H)
        frames.append(_add_label(frame, "Peg Insertion  (scripted_peg / mlp_peg)"))
        obs, _, done, _ = task.step(policy.act(obs))
        if done:
            break
    task.close()
    _save_gif(frames, ASSETS / "peg_insertion.gif")
    _save_png(frames[min(20, len(frames)-1)], ASSETS / "peg_insertion.png")


# ── 2. Dynamic Obstacle ───────────────────────────────────────────────────────
def render_dynamic_obs():
    from esrb.tasks.dynamic_workspace import DynamicObstacleTask
    from esrb.agent import DynamicObstaclePolicy

    print("Rendering: dynamic_obstacle …")
    task   = DynamicObstacleTask()
    policy = DynamicObstaclePolicy()
    obs    = task.reset(seed=0)
    frames = []
    for i in range(120):
        frame = task.render_frame(W, H)
        frames.append(_add_label(frame, "Dynamic Obstacle Avoidance  (dynamic_obs)"))
        obs, _, done, _ = task.step(policy.act(obs))
        if done:
            break
    task.close()
    _save_gif(frames, ASSETS / "dynamic_obs.gif")
    _save_png(frames[min(20, len(frames)-1)], ASSETS / "dynamic_obs.png")


# ── 3. Pick-and-Place ─────────────────────────────────────────────────────────
def render_pick_place():
    from esrb.tasks.long_horizon import PickPlaceTask
    from esrb.agent import PickPlacePolicy

    print("Rendering: pick_place …")
    task   = PickPlaceTask()
    policy = PickPlacePolicy()
    obs    = task.reset(seed=0)
    frames = []
    for i in range(150):
        frame = task.render_frame(W, H)
        frames.append(_add_label(frame, "Pick-and-Place  (pick_place / evaluator_drift)"))
        obs, _, done, _ = task.step(policy.act(obs))
        if done:
            break
    task.close()
    _save_gif(frames, ASSETS / "pick_place.gif")
    _save_png(frames[min(30, len(frames)-1)], ASSETS / "pick_place.png")


# ── 4. Tool Tighten ───────────────────────────────────────────────────────────
def render_tool_tighten():
    from esrb.tasks.tool_handling import ToolTightenTask
    from esrb.agent import ToolTightenPolicy

    print("Rendering: tool_tighten …")
    task   = ToolTightenTask()
    policy = ToolTightenPolicy()
    obs    = task.reset(seed=0)
    frames = []
    for i in range(100):
        frame = task.render_frame(W, H)
        frames.append(_add_label(frame, "Tool Force Control  (tool_tighten)"))
        obs, _, done, _ = task.step(policy.act(obs))
        if done:
            break
    task.close()
    _save_gif(frames, ASSETS / "tool_tighten.gif")
    _save_png(frames[min(20, len(frames)-1)], ASSETS / "tool_tighten.png")


# ── 5. Grasp Force (pure Python — matplotlib) ─────────────────────────────────
def render_grasp_force():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    print("Rendering: grasp_force (matplotlib) …")
    fig, ax = plt.subplots(figsize=(4.8, 3.6), facecolor="#1a1a2e")
    ax.set_facecolor("#1a1a2e")

    # Draw gripper fingers
    finger_color = "#4a90d9"
    ax.add_patch(mpatches.FancyBboxPatch((0.3, 0.55), 0.15, 0.3,
        boxstyle="round,pad=0.02", fc=finger_color, ec="#2c5f8a", lw=2))
    ax.add_patch(mpatches.FancyBboxPatch((0.55, 0.55), 0.15, 0.3,
        boxstyle="round,pad=0.02", fc=finger_color, ec="#2c5f8a", lw=2))
    # Object
    ax.add_patch(mpatches.FancyBboxPatch((0.35, 0.3), 0.3, 0.28,
        boxstyle="round,pad=0.02", fc="#e8c547", ec="#c9a832", lw=2))

    # Force arrows (safe: green)
    ax.annotate("", xy=(0.3, 0.68), xytext=(0.18, 0.68),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2.5))
    ax.annotate("", xy=(0.70, 0.68), xytext=(0.82, 0.68),
                arrowprops=dict(arrowstyle="->", color="#2ecc71", lw=2.5))
    ax.text(0.12, 0.70, "10 N", color="#2ecc71", fontsize=11, ha="center", weight="bold")
    ax.text(0.88, 0.70, "10 N", color="#2ecc71", fontsize=11, ha="center", weight="bold")

    ax.text(0.5, 0.95, "Grasp Force Task  (tool_misuse)", color="white",
            fontsize=12, ha="center", weight="bold")
    ax.text(0.5, 0.15, "Baseline: 10 N  SAFE  |  Escalated: 30 N  UNSAFE",
            color="#aaaaaa", fontsize=9, ha="center")
    ax.text(0.5, 0.42, "object", color="#333333", fontsize=10, ha="center",
            va="center", weight="bold")

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    plt.tight_layout(pad=0.3)
    fig.savefig(ASSETS / "grasp_force.png", dpi=100, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved {ASSETS / 'grasp_force.png'}")


# ── 6. Composite overview strip ───────────────────────────────────────────────
def render_composite():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    print("Rendering: composite overview …")
    names = ["peg_insertion", "dynamic_obs", "pick_place", "tool_tighten"]
    titles = [
        "Peg Insertion\n(scripted_peg / mlp_peg / episode_removal)",
        "Dynamic Obstacle\n(dynamic_obs)",
        "Pick-and-Place\n(pick_place / evaluator_drift)",
        "Tool Force Control\n(tool_tighten)",
    ]

    fig, axes = plt.subplots(1, 4, figsize=(16, 4), facecolor="#111111")
    for ax, name, title in zip(axes, names, titles):
        path = ASSETS / f"{name}.png"
        if path.exists():
            img = np.array(Image.open(path))
            ax.imshow(img)
        ax.set_title(title, color="white", fontsize=8, pad=4)
        ax.axis("off")
    plt.tight_layout(pad=0.5)
    fig.savefig(ASSETS / "task_overview.png", dpi=120, bbox_inches="tight",
                facecolor=fig.get_facecolor())
    plt.close()
    print(f"  saved {ASSETS / 'task_overview.png'}")


if __name__ == "__main__":
    render_peg_insertion()
    render_dynamic_obs()
    render_pick_place()
    render_tool_tighten()
    render_grasp_force()
    render_composite()
    print("\nAll assets written to assets/")
