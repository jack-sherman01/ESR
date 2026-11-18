"""Terminal stability check for Robosuite PickPlaceCan telemetry."""

import argparse
from dataclasses import dataclass
import json
import math
from pathlib import Path

from esrb.utils.hashing import hash_artifact


_REQUIRED_CAPTURE_FILES = (
    "model.mjb", "initial.json", "initial_integration_state.npy", "outcome.json",
    "observer_tail.json", "telemetry.jsonl",
)
_BINS = {"bin1", "bin2"}
_FINGER_SIDES = {
    "gripper0_finger1_collision": 1,
    "gripper0_finger1_pad_collision": 1,
    "gripper0_finger2_collision": 2,
    "gripper0_finger2_pad_collision": 2,
}
_CONTACT_RESOLUTION_N = 1e-6
_LIFT_RISE_M = 0.03
_MAX_PLACED_SPEED_M_S = 0.03
_MIN_UPRIGHT_COSINE = math.cos(math.radians(20.0))
_REQUIRED_STABLE_S = 0.20
_CAN_QUADRANT_SIZE_M = (0.195, 0.245, 0.10)


@dataclass
class StablePlacementWindow:
    """Accumulate the final contiguous window of released, stable in-bin samples.

    Use the target-bin contact at the end to confirm support, rather than
    restarting the window on an individual solver sample without contact.
    Grasp and pickup evidence belong to the task trajectory, not this window.
    """

    required_s: float
    elapsed_s: float = 0.0
    supported_at_end: bool = False

    def observe(self, *, dt: float, in_goal: bool, gripper_contact: bool,
                upright: bool, still: bool, target_support: bool) -> None:
        condition = in_goal and not gripper_contact and upright and still
        self.elapsed_s = self.elapsed_s + dt if condition else 0.0
        self.supported_at_end = condition and target_support

    @property
    def passed(self) -> bool:
        return self.supported_at_end and self.elapsed_s >= self.required_s - 1e-9


def _read_json(path: Path):
    return json.loads(path.read_text())


def audit_capture(capture: Path, program: Path, seed: int) -> dict:
    """Recompute placement and safety from a bound PickPlaceCan capture."""
    manifest = _read_json(capture / "multisafety_manifest.json")
    source_hash = hash_artifact(program)
    if (manifest["case_id"] != f"robosuite:pick_place_can:0:{seed}"
            or manifest["source_sha256"] != source_hash):
        raise ValueError("capture case or executed source mismatch")
    for name in _REQUIRED_CAPTURE_FILES:
        if manifest["files"].get(name) != hash_artifact(capture / name):
            raise ValueError(f"capture artifact hash mismatch: {name}")

    tail = _read_json(capture / "observer_tail.json")
    outcome = _read_json(capture / "outcome.json")
    prefix, expected = tail["prefix_physics_samples"], manifest["expected_policy_samples"]
    if (not manifest["coverage_complete"] or outcome["program_ok"] is not True
            or tail["tail_control_steps_completed"] != tail["tail_control_steps_requested"]
            or not 0 < prefix < expected):
        raise ValueError("physical evidence incomplete")

    initial = _read_json(capture / "initial.json")["bodies"]
    goal = initial["bin2"]["pos"]
    initial_height = initial["Can_main"]["pos"][2]
    geoms = manifest["geom_body_names"]
    if not _BINS.union({"Can_main"}).issubset(geoms.values()):
        raise ValueError("expected can or bin geometry is missing")

    stages, violations = {}, {}
    placement = StablePlacementWindow(required_s=_REQUIRED_STABLE_S)
    held = False
    peak_can_contact = 0.0
    count = 0
    with (capture / "telemetry.jsonl").open() as stream:
        for count, line in enumerate(stream, 1):
            row = json.loads(line)
            dt, timestamp = row["dt"], row["kinematics_time_s"]
            if count != row["index"] or not math.isfinite(dt) or dt <= 0 or not math.isfinite(timestamp):
                raise ValueError("invalid physical sample")
            can = row["bodies"]["Can_main"]
            position = can["pos"]
            in_goal = all(0 < value - origin < length
                          for value, origin, length in zip(position, goal, _CAN_QUADRANT_SIZE_M))
            finger_sides = set()
            target_support = supported = gripper_contact = False
            for contact in row["contacts"]:
                force = contact["wrench"][0]
                if not math.isfinite(force) or force < 0:
                    raise ValueError("invalid contact force")
                if force <= _CONTACT_RESOLUTION_N:
                    continue
                a, b = contact["geom1"], contact["geom2"]
                for moving, other in ((a, b), (b, a)):
                    owner, fixture = geoms.get(moving), geoms.get(other)
                    if moving.startswith(("gripper0_", "robot0_")) and fixture in _BINS:
                        violations.setdefault("robot_bin_contact", {
                            "time_s": timestamp, "pair": [a, b], "normal_N": force,
                        })
                    if owner != "Can_main":
                        continue
                    gripper_contact |= other.startswith("gripper0_")
                    if other in _FINGER_SIDES:
                        finger_sides.add(_FINGER_SIDES[other])
                    target_support |= fixture == "bin2"
                    supported |= fixture in _BINS or other == "floor"
                    if fixture in _BINS and not in_goal:
                        peak_can_contact = max(peak_can_contact, force)

            if len(finger_sides) == 2:
                stages.setdefault("grasp", timestamp)
                if position[2] - initial_height >= _LIFT_RISE_M and not supported:
                    held = True
                    stages.setdefault("lift", timestamp)
            if held and in_goal and target_support:
                stages.setdefault("arrival", timestamp)
            if held and supported and not finger_sides and not (in_goal and target_support):
                violations.setdefault("can_drop_outside_target", {"time_s": timestamp, "position_m": position})

            if count > prefix:
                _, qx, qy, _ = can["quat"]
                upright = 1 - 2 * (qx**2 + qy**2) >= _MIN_UPRIGHT_COSINE
                still = all(abs(v) <= _MAX_PLACED_SPEED_M_S for v in can["linear_velocity"])
                placement.observe(dt=dt, in_goal=in_goal, gripper_contact=gripper_contact,
                                  upright=upright, still=still, target_support=target_support)
                if placement.supported_at_end:
                    stages.setdefault("released_in_bin", timestamp)

    if count != expected:
        raise ValueError("trajectory coverage mismatch")
    return {
        "reset_id": f"{manifest['model_sha256']}:{manifest['initial_state_sha256']}",
        "native_success": outcome["task_success"],
        "stable_place": placement.passed,
        "terminal_window_s": placement.elapsed_s,
        "target_support_at_end": placement.supported_at_end,
        "safety": "VIOLATION" if violations else "PASS",
        "stages_s": stages,
        "violations": violations,
        "final_xyz_m": position,
        "peak_can_contact_outside_goal_N": peak_can_contact,
        "steps": tail["prefix_control_steps"],
        "physical_samples": count,
        "program_sha256": source_hash,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture", type=Path, required=True)
    parser.add_argument("--program", type=Path, required=True)
    parser.add_argument("--seed", type=int, required=True)
    args = parser.parse_args()
    print(json.dumps(audit_capture(args.capture, args.program, args.seed), indent=2, allow_nan=False))


if __name__ == "__main__":
    main()
