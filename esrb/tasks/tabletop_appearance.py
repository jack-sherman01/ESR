"""Close-up appearance for the retained compliant single-insertion workcell.

Articulated production workcells own their robot, geometry and camera setup.
"""
import mujoco
import numpy as np

from esrb.tasks.tool_handling.appearance import decorate_scene as decorate_room

SCENARIOS = ('long_horizon_assembly', 'moving_transport', 'moving_assembly')
STYLE = 'robosuite-inspired-focused-workcell/1'


def apply_appearance(model, name):
    floor = model.geom('floor').id
    model.geom_group[floor] = 5
    for i in range(model.ngeom):
        geom = model.geom(i)
        color = None
        if geom.name == 'rack':
            color = [.17, .32, .43, 1.]
        elif geom.name == 'work_surface':
            color = [.24, .40, .50, 1.]
        elif geom.name and geom.name.startswith('socket_'):
            color = [.38, .54, .63, 1.]
        elif geom.name in ('left_pad', 'right_pad'):
            color = [.12, .14, .16, 1.]
        elif geom.name == 'tool_handle':
            color = [.96, .43, .10, 1.]
        elif geom.name == 'moving_proxy':
            color = [.85, .10, .13, 1.]
        elif model.geom_bodyid[i] == model.body('gantry').id:
            color = [.88, .90, .92, 1.]
        if color is not None:
            model.geom_rgba[i] = color
    model.site_rgba[:, 3] = 0.  # Remove debug dots from the working surfaces.
    model.light_castshadow[:] = False
    if hasattr(model, 'light_type'):
        model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        model.light_directional[:] = True
    model.light_pos[:] = [1., -1., 1.5]
    model.light_dir[:] = [-.19, .19, -.963]
    model.light_diffuse[:] = [.65, .65, .65]
    model.light_ambient[:] = [.10, .10, .10]
    model.vis.headlight.ambient[:] = [.25, .25, .25]
    model.vis.headlight.diffuse[:] = [.35, .35, .35]
    model.vis.headlight.specular[:] = [.1, .1, .1]
    # A single fixed oblique view shows the slot opening and actual insertion.
    moving = name.startswith('moving_')
    position = np.array([.34, -.48, .44] if moving else [.29, -.37, .32])
    target = np.array([.0, .0, .125] if moving else [.015, 0., .115])
    backward = position-target
    backward /= np.linalg.norm(backward)
    right = np.cross([0., 0., 1.], backward)
    right /= np.linalg.norm(right)
    camera = model.camera('overview').id
    model.cam_pos[camera] = position
    mujoco.mju_mat2Quat(model.cam_quat[camera],
                       np.column_stack((right, np.cross(backward, right), backward)).ravel())
    model.cam_fovy[camera] = 42. if moving else 38.


def decorate_scene(scene):
    decorate_room(scene)
    # Top z=0 matches the original physical support plane exactly.
    geom = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX,
                       np.array([.45, .42, .018]), np.array([0., 0., -.018]),
                       np.eye(3).ravel(), np.array([.78, .78, .76, 1.], dtype=np.float32))
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.objtype, geom.objid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1
    scene.ngeom += 1


class TabletopVisuals:
    """Render hooks shared by assembly and the moving-workspace tasks."""

    def render_frame(self, width=640, height=480):
        if self._renderer is None:
            apply_appearance(self._model, self.name)
            mujoco.mj_camlight(self._model, self._data)
        return super().render_frame(width, height)

    def _decorate_render_scene(self, scene):
        decorate_scene(scene)
