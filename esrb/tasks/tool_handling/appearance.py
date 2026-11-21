"""Robosuite-inspired TableArena visuals for the existing Panda physics model.

The texture/material edits and extra MjvScene geometry affect rendering only.
No bodies, collision shapes, joint dynamics or contact parameters are added.
"""
from pathlib import Path

import mujoco
import numpy as np

ASSETS = Path(__file__).parent/'assets/robosuite'
STYLE = 'robosuite-inspired-table-arena/1'


def apply_appearance(model):
    from PIL import Image

    texture = model.texture('bench_grid').id
    channels = int(model.tex_nchannel[texture]) if hasattr(model, 'tex_nchannel') else 3
    width, height = int(model.tex_width[texture]), int(model.tex_height[texture])
    pixels = model.tex_data if hasattr(model, 'tex_data') else model.tex_rgb
    start = int(model.tex_adr[texture])
    with Image.open(ASSETS/'ceramic.png') as image:
        rgb = np.asarray(image.convert('RGB').resize((width, height), Image.Resampling.LANCZOS))
    pixels[start:start+width*height*channels].reshape(height, width, channels)[:, :, :3] = rgb
    bench = model.material('bench').id
    model.mat_rgba[bench] = [.86, .86, .84, 1.]
    model.mat_texrepeat[bench] = [1., 1.]
    model.mat_reflectance[bench], model.mat_shininess[bench], model.mat_specular[bench] = 0., 0., .2
    floor = model.material('white').id
    model.mat_rgba[floor] = [.84, .84, .82, 1.]
    model.mat_specular[floor], model.mat_shininess[floor], model.mat_reflectance[floor] = 0., 0., .01
    model.geom_matid[model.geom('floor').id] = floor
    # Its finite display extent leaves the old sky visible behind the table.
    # Render a room-wide floor below instead; the physical plane stays active.
    model.geom_group[model.geom('floor').id] = 5
    metal = model.material('aluminium').id
    model.mat_rgba[metal] = [.58, .62, .65, 1.]
    model.mat_specular[metal], model.mat_shininess[metal] = .25, .15
    # Hide non-task decorations using an invisible display group. The geometry
    # and collision masks remain present and unchanged in the physical model.
    for index in range(model.ngeom):
        if (model.geom_bodyid[index] != 0 or model.geom_contype[index]
                or model.geom_conaffinity[index] or model.geom_pos[index, 2] < -.04):
            continue
        x, y = model.geom_pos[index, :2]
        near_fixture = (abs(x+.10) <= .056 and abs(y) <= .056) or (abs(x-.10) <= .061 and abs(y) <= .061)
        if not near_fixture:
            model.geom_group[index] = 5
    model.light_castshadow[:] = False
    if hasattr(model, 'light_type'):
        model.light_type[:] = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
    else:
        model.light_directional[:] = True
    model.light_pos[:] = [[1., -1., 1.5], [-.5, -.5, 1.], [0., 1., 1.5]]
    directions = np.array([[-.2, .2, -1.], [.3, .2, -1.], [0., -.2, -1.]])
    model.light_dir[:] = directions / np.linalg.norm(directions, axis=1)[:, None]
    model.light_diffuse[:] = [[.65, .65, .65], [.35, .35, .35], [.2, .2, .2]]
    model.light_ambient[:] = 0.
    model.vis.headlight.ambient[:] = [.25, .25, .25]
    model.vis.headlight.diffuse[:] = [.35, .35, .35]
    model.vis.headlight.specular[:] = [.1, .1, .1]
    model.vis.rgba.haze[:] = [.88, .88, .87, 1.]


def decorate_scene(scene):
    """Add the room and subtle floor grout to the renderer, outside MjModel."""
    def box(position, size, color):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError('Panda arena exceeds the renderer geometry capacity')
        geom = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(geom, mujoco.mjtGeom.mjGEOM_BOX, np.asarray(size, dtype=float),
                           np.asarray(position, dtype=float), np.eye(3).ravel(),
                           np.asarray(color, dtype=np.float32))
        geom.category = mujoco.mjtCatBit.mjCAT_DECOR
        geom.objtype, geom.objid = mujoco.mjtObj.mjOBJ_UNKNOWN, -1
        scene.ngeom += 1

    box([.5, 0., -.325], [2.5, 3., .005], [.53, .53, .52, 1.])
    wall = [.88, .87, .85, 1.]
    for position, size in [([-2., 0., 1.18], [.01, 3., 1.5]),
                           ([3., 0., 1.18], [.01, 3., 1.5]),
                           ([.5, 3., 1.18], [2.5, .01, 1.5]),
                           ([.5, -3., 1.18], [2.5, .01, 1.5])]:
        box(position, size, wall)
    for offset in np.arange(-3., 3.01, .5):
        box([offset, 0., -.3198], [.001, 3., .0001], [.46, .46, .45, 1.])
        box([0., offset, -.3198], [3., .001, .0001], [.46, .46, .45, 1.])
