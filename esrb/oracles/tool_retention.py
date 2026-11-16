"""Tool retention metrics from world poses and pad contact measurements."""

import numpy as np


def relative_positions(tool, gripper, rotations):
    w, x, y, z = rotations.T
    matrices = np.stack([
        1 - 2 * (y*y + z*z), 2 * (x*y - z*w), 2 * (x*z + y*w),
        2 * (x*y + z*w), 1 - 2 * (x*x + z*z), 2 * (y*z - x*w),
        2 * (x*z - y*w), 2 * (y*z + x*w), 1 - 2 * (x*x + y*y),
    ], axis=1).reshape(-1, 3, 3)
    return np.einsum("nji,nj->ni", matrices, tool - gripper)


def relative_quaternions(tool, gripper):
    # inv(gripper) * tool, with quaternion components ordered w, x, y, z.
    a, b = gripper[:, 0], -gripper[:, 1:]
    c, d = tool[:, 0], tool[:, 1:]
    return np.column_stack((a*c - np.sum(b*d, axis=1), a[:, None]*d + c[:, None]*b + np.cross(b, d)))


def retention_metric(metric, values):
    if metric == "peak_grip_force_n":
        return float(np.max(np.sum(values[0], axis=1)))
    if metric == "peak_tool_rotation_rad":
        tool, gripper, active = values
        relative = relative_quaternions(tool, gripper)[active > 0.5]
        if not len(relative):
            return 0.0
        angles = 2 * np.arccos(np.clip(np.abs(relative @ relative[0]), 0, 1))
        return float(np.max(angles))
    tool, gripper, rotations, active = values[:4]
    relative = relative_positions(tool, gripper, rotations)[active > 0.5]
    if not len(relative):
        return 0.0
    displacement = np.linalg.norm(relative - relative[0], axis=1)
    if metric == "peak_tool_slip_m":
        return float(np.max(displacement))
    grip = values[4][active > 0.5]
    return float(np.any((np.min(grip, axis=1) < 0.05) & (displacement > 0.035)))


def validate_tool_measurements(trace, *, retention_stages=None):
    """Check reductions against ten recorded contact substeps per control step."""
    samples = np.asarray(trace["contact_samples"], dtype=float)
    count = len(trace["ee_positions"])
    if samples.shape != (count, 10, 10) or not np.isfinite(samples).all():
        raise ValueError("invalid tool contact samples")
    rows = np.arange(count)
    forces, torques = samples[:, :, 1:4], samples[:, :, 4:7]
    force_peaks = forces[rows, np.argmax(np.linalg.norm(forces, axis=2), axis=1)]
    torque_peaks = torques[rows, np.argmax(np.linalg.norm(torques, axis=2), axis=1)]
    checks = (
        (trace["contact_forces"], force_peaks),
        (trace["contact_torques"], torque_peaks),
        (trace["grip_normal_forces"], samples[:, :, 7:9].max(axis=1)),
        (trace["surface_forces"], samples[:, :, 9].max(axis=1)),
        (trace["times_s"], samples[:, -1, 0]),
    )
    if any(np.shape(actual) != np.shape(expected) or not np.allclose(actual, expected, atol=1e-10, rtol=0)
           for actual, expected in checks):
        raise ValueError("tool contact reduction mismatch")
    if not np.allclose(np.diff(samples[:, :, 0].ravel()), 0.002, atol=1e-10, rtol=0):
        raise ValueError("tool contact timebase mismatch")
    active = [int(stage != "grasp" if retention_stages is None else stage in retention_stages)
              for stage in trace["stages"]]
    if trace["retention_active"] != active:
        raise ValueError("tool retention stage mismatch")

    tool = np.asarray(trace["tool_positions"])
    gripper = np.asarray(trace["ee_positions"])
    rotations = np.asarray(trace["gripper_quaternions"])
    relative = relative_positions(tool, gripper, rotations)
    quaternions = relative_quaternions(np.asarray(trace["tool_quaternions"]), rotations)
    if not np.allclose(trace["relative_positions"], relative, atol=1e-10, rtol=0):
        raise ValueError("tool relative position mismatch")
    if not np.allclose(trace["relative_quaternions"], quaternions, atol=1e-10, rtol=0):
        raise ValueError("tool relative orientation mismatch")
    indices = np.flatnonzero(active)
    expected_events = []
    if len(indices):
        slip = np.linalg.norm(relative[indices] - relative[indices[0]], axis=1)
        grip = np.asarray(trace["grip_normal_forces"])[indices]
        dropped = (np.min(grip, axis=1) < 0.05) & (slip > 0.035)
        starts = indices[dropped & ~np.r_[False, dropped[:-1]]]
        expected_events = [
            {"step": int(index) + 1, "time_s": trace["times_s"][index], "stage": trace["stages"][index]}
            for index in starts
        ]
    if trace["drop_events"] != expected_events:
        raise ValueError("tool drop event mismatch")
