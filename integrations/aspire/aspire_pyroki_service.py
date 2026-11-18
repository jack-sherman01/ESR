"""Upstream Panda wire contract: seven arm joints plus one finger coordinate.

ASPIRE's typed client appends the finger seed and strips its solution. The
earlier local seven-DOF service launcher is retired.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pyroki as pk
from robot_descriptions.loaders.yourdfpy import load_robot_description
import uvicorn
from cap_harness.providers.pyroki import service


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--port", type=int, default=8116)
    parser.add_argument("--urdf", type=Path, help="Local Panda URDF for kinematics-only IK; meshes are not loaded")
    args = parser.parse_args()
    if args.out is not None and args.out.exists():
        parser.error("Output already exists")
    if args.urdf:
        from yourdfpy import URDF
        description = URDF.load(args.urdf, load_meshes=False, load_collision_meshes=False)
    else:
        description = load_robot_description("panda_description")
    robot = pk.Robot.from_urdf(service._apply_joint_margin(description))
    expected = tuple(f"panda_joint{i}" for i in range(1, 8)) + ("panda_finger_joint1",)
    if tuple(robot.joints.actuated_names) != expected:
        raise RuntimeError("Panda description changed; review native client wire contract")
    link = robot.links.names.index("panda_hand")
    report = {"status": "passed", "kind": "native_pyroki_wire_contract_check",
              "policy_arm_dof": 7, "provider_configuration_dof": 8,
              "joint_order": list(expected), "gripper_frozen": False,
              "upstream_service_math_modified": False}
    if args.urdf:
        report.update(urdf=str(args.urdf.resolve()), urdf_sha256=hashlib.sha256(args.urdf.read_bytes()).hexdigest(),
                      collision_planning=False)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("x") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2), flush=True)
    if not args.check_only:
        service._ROBOT = robot
        service._TARGET_LINK_INDEX = link
        uvicorn.run(service.app, host="127.0.0.1", port=args.port)


if __name__ == "__main__":
    main()
