"""Shared simulator-state and NutAssembly contact evidence helpers.

Used by trusted capture hosts; simulator dependencies load only when called.
"""
import hashlib
import json
from pathlib import Path


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def read(path):
    return json.loads(Path(path).read_text())


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def pair_category(group1, group2):
    return "--".join(sorted((group1, group2)))


def geom_group(name, ancestors):
    if (name or "").startswith("SquareNut_"):
        return "square_nut"
    if (name or "").startswith("RoundNut_"):
        return "inactive_round_nut"
    if any((x or "").startswith("gripper0_") for x in ancestors):
        return "gripper"
    if "peg1" in ancestors:
        return "square_peg"
    if "peg2" in ancestors:
        return "round_peg"
    if "table" in ancestors:
        return "table"
    if any((x or "").startswith("robot0_") for x in ancestors):
        return "robot_link"
    return "floor" if name == "floor" else "other"


def geometry_inventory(env):
    """Resolve exact collision geoms, including unnamed arena peg geometry."""
    model = env.sim.model
    rows = []
    for index in range(model.ngeom):
        name = model.geom_id2name(index)
        body_id = int(model.geom_bodyid[index])
        ancestors = []
        cursor = body_id
        while cursor:
            ancestors.append(model.body_id2name(cursor))
            cursor = int(model.body_parentid[cursor])
        body_name = model.body_id2name(body_id)
        group = geom_group(name, ancestors)
        rows.append({"geom_id": index, "geom_name": name, "body_id": body_id,
                     "body_name": body_name, "body_ancestors": ancestors, "group": group,
                     "contype": int(model.geom_contype[index]),
                     "conaffinity": int(model.geom_conaffinity[index]),
                     "type": int(model.geom_type[index]), "size_m": model.geom_size[index].tolist()})
    for group in ("square_nut", "square_peg", "table", "gripper", "robot_link"):
        if not any(row["group"] == group and (row["contype"] or row["conaffinity"]) for row in rows):
            raise ValueError(f"missing collision geometry identity: {group}")
    return rows


def contact_rows(model, data, inventory):
    import mujoco
    import numpy as np
    result = []
    for index in range(data.ncon):
        contact = data.contact[index]
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, index, wrench)
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        frame = np.asarray(contact.frame).reshape(3, 3)
        if not np.all(np.isfinite(wrench)):
            raise ValueError("nonfinite contact wrench")
        result.append({"contact_index": index, "geom1": geom1, "geom2": geom2,
                       "group1": inventory[geom1]["group"], "group2": inventory[geom2]["group"],
                       "pair": pair_category(inventory[geom1]["group"], inventory[geom2]["group"]),
                       "position_world_m": contact.pos.tolist(), "distance_m": float(contact.dist),
                       "frame_world_to_contact": frame.tolist(),
                       "wrench_contact_raw": wrench.tolist(),
                       "force_world_n": (frame.T @ wrench[:3]).tolist(),
                       "torque_qualification": "NOT_ENABLED"})
    return result


def native_state(sim):
    import mujoco
    import numpy as np
    spec = mujoco.mjtState.mjSTATE_INTEGRATION
    state = np.empty(mujoco.mj_stateSize(sim.model._model, spec))
    mujoco.mj_getState(sim.model._model, sim.data._data, state, spec)
    return state
