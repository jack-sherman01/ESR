"""Benchmark-owned applied physical reset state, independent of requested seed."""


def reset_snapshot(model,data):
    return {key:getattr(data,key).tolist() for key in ('qpos','qvel','ctrl','mocap_pos')} | {
        'body_mass_kg':model.body_mass.tolist(),'geom_friction':model.geom_friction.tolist(),
        'geom_size_m':model.geom_size.tolist(),'geom_position_m':model.geom_pos.tolist()}
