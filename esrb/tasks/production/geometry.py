"""Three-piece geometry adapted from NVlabs/mimicgen, revision c0fbff22.

The occupancy patterns and unit sizes derive from ThreePieceAssembly and
BoxPatternObject. Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
See assets/MIMICGEN_LICENSE. Adjacent cells are merged without filling holes.
ESRB additions: a real base grasp tab, chamfered visual meshes, fixtures, and
bounded mass densities suitable for friction grasping with the native Panda.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from scipy.spatial import ConvexHull

from esrb.tasks.robots.panda import robot_tree

PATTERNS = {
    'base': [[[1,1,1,1,1],[1,0,0,0,1],[1,0,0,0,1],[1,0,0,0,1],[1,1,1,1,1]]],
    'connector': [[[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[1,1,1],[0,0,0]], [[0,0,0],[1,1,1],[0,0,0]]],
    'cap': [[[1,1,1],[0,0,0],[1,1,1]], [[1,1,1],[0,0,0],[1,1,1]],
            [[1,1,1],[1,1,1],[1,1,1]], [[0,0,0],[0,1,0],[0,0,0]]],
}
UNIT = {'base': .019, 'connector': .017, 'cap': .02}
HALF_HEIGHT = {'base': .019, 'connector': .051, 'cap': .08}
GRASP = {'base': np.array([-.076,0.,.045]), 'connector': np.array([0.,0.,.027]), 'cap': np.array([0.,0.,.06])}
GRASP_YAW = {'base': 0., 'connector': np.pi/2, 'cap': 0.}
ASSEMBLED_Z = {'base': .019, 'connector': .051, 'cap': .118}
PARTS = tuple(PATTERNS)
ASSETS = Path(__file__).parent/'assets'


def merged_boxes(name):
    """Merge face-adjacent occupied cells into collision-exact cuboids."""
    a = np.array(PATTERNS[name], dtype=bool)
    size = np.array(a.shape)
    unit = UNIT[name]
    boxes = []
    for index in np.ndindex(a.shape):
        if not a[index]:
            continue
        low = np.array(index)
        high = low+1
        for axis in (2,1,0):
            while high[axis] < size[axis]:
                trial = high.copy(); trial[axis] += 1
                if not a[tuple(slice(l,h) for l,h in zip(low,trial))].all():
                    break
                high = trial
        a[tuple(slice(l,h) for l,h in zip(low,high))] = False
        center = ((low+high-size)*unit)[[1,2,0]]
        half = ((high-low)*unit)[[1,2,0]]
        boxes.append((center,half))
    if name == 'base':
        boxes.append((np.array([-.076,0.,.04]), np.array([.011,.018,.021])))
    return boxes


def chamfer_box(center, half, bevel=.0005):
    points = []
    for corner in np.ndindex(2,2,2):
        signs = 2*np.array(corner)-1
        for axis in range(3):
            q = half.copy()-bevel
            q[axis] += bevel
            points.append(center+signs*q)
    points = np.array(points)
    hull=ConvexHull(points)
    faces=hull.simplices.copy()
    for face,plane in zip(faces,hull.equations):
        normal=np.cross(points[face[1]]-points[face[0]],points[face[2]]-points[face[0]])
        if normal @ plane[:3] < 0: face[1],face[2]=face[2],face[1]
    return points,faces


def build_meshes():
    """Explicit asset build used by development, never run during task reset."""
    for name in PARTS:
        vertices, faces = [], []
        for center, half in merged_boxes(name):
            v, f = chamfer_box(center,half)
            faces.extend((f+len(vertices)+1).tolist()); vertices.extend(v.tolist())
        text = ['# ESRB chamfered display mesh; collision topology uses native pattern boxes']
        text += ['v '+' '.join(f'{x:.7f}' for x in v) for v in vertices]
        text += ['f '+' '.join(str(i) for i in f) for f in faces]
        (ASSETS/(name+'.obj')).write_text('\n'.join(text)+'\n')


def add_geom(parent, name, pos, half, *, material='steel', **kwargs):
    return ET.SubElement(parent,'geom',name=name,type='box',pos=' '.join(map(str,pos)),
                         size=' '.join(map(str,half)),material=material,**{k:str(v) for k,v in kwargs.items()})


def frame(parent, name, center, half, thickness=.006, height=.012, material='blue'):
    x,y,z = center; a,b = half
    add_geom(parent,name+'_front',[x,y-b,z+height/2],[a+thickness,thickness,height/2],material=material)
    add_geom(parent,name+'_back',[x,y+b,z+height/2],[a+thickness,thickness,height/2],material=material)
    add_geom(parent,name+'_left',[x-a,y,z+height/2],[thickness,b,height/2],material=material)
    add_geom(parent,name+'_right',[x+a,y,z+height/2],[thickness,b,height/2],material=material)


def make_scene(mode):
    root = robot_tree()
    root.set('model','panda_production_'+mode)
    asset, world = root.find('asset'),root.find('worldbody')
    for name, color in {'steel':'.64 .69 .73 1','blue':'.10 .30 .46 1','orange':'.94 .43 .11 1',
                        'bench':'.78 .79 .78 1','belt':'.16 .19 .21 1','rubber':'.08 .09 .1 1'}.items():
        old=asset.find(f"material[@name='{name}']")
        if old is None: ET.SubElement(asset,'material',name=name,rgba=color,specular='.3',shininess='.2')
        else:
            old.set('rgba',color)
            if name=='bench':
                for attr in ('texture','texrepeat','reflectance'): old.attrib.pop(attr,None)
    for name in PARTS:
        ET.SubElement(asset,'mesh',name=name+'_visual',file=str(ASSETS/(name+'.obj')))
    add_geom(world,'workbench',[0.,-.05,-.025],[1.1,.62,.025],material='bench')
    ET.SubElement(world,'geom',name='floor',type='plane',pos='0 0 -.75',size='3 3 .02',rgba='.85 .85 .84 1')
    source_y=-.08
    if mode=='transport':
        add_geom(world,'conveyor_surface',[-.6,.13,.01],[1.1,.125,.01],material='belt',surfacevel='.04 0 0 0 0 0')
        for side in (-1,1):
            add_geom(world,'conveyor_guard_'+str(side),[-.6,.13+side*.139,.025],[1.1,.008,.025])
        positions={name:np.array([-.2-.64*i,.13,.02+HALF_HEIGHT[name]]) for i,name in enumerate(PARTS)}
        destinations={name:np.array([-.25+.25*i,-.16,.02+HALF_HEIGHT[name]]) for i,name in enumerate(PARTS)}
        for name, target in destinations.items():
            h={'base':.104,'connector':.066,'cap':.074}[name]
            add_geom(world,'kit_'+name+'_floor',[*target[:2],.01],[h+.012,h+.012,.01],material='blue')
            frame(world,'kit_'+name,[*target[:2],.02],[h,h],height=.012)
    else:
        positions={'base':np.array([-.24,source_y,.039]),'connector':np.array([0.,source_y,.071]),
                   'cap':np.array([.24,source_y,.10])}
        for name,pos in positions.items():
            if mode=='moving_assembly' and name=='base': continue
            h=.105 if name=='base' else .065
            add_geom(world,'rack_'+name,[*pos[:2],.01],[h,h,.01],material='blue')
        if mode=='assembly':
            add_geom(world,'fixture_floor',[0.,.17,.01],[.114,.114,.01],material='blue')
            frame(world,'fixture',[0.,.17,.02],[.103,.103],height=.012)
        else:
            add_geom(world,'conveyor_surface',[.05,.17,.01],[.50,.13,.01],material='belt',surfacevel='.005 0 0 0 0 0')
            pallet=ET.SubElement(world,'body',name='pallet',pos='-.12 .17 .027')
            ET.SubElement(pallet,'freejoint',name='pallet_free')
            add_geom(pallet,'pallet_floor',[0,0,0],[.114,.114,.007],material='blue',mass='.25')
            frame(pallet,'pallet',[0.,0.,.007],[.103,.103],height=.012)
            positions['base']=np.array([-.12,.17,.034+HALF_HEIGHT['base']])
        destinations=None
    for name, pos in positions.items():
        body=ET.SubElement(world,'body',name=name,pos=' '.join(map(str,pos)))
        ET.SubElement(body,'freejoint',name=name+'_free')
        for i,(center,half) in enumerate(merged_boxes(name)):
            add_geom(body,name+'_collision_'+str(i),center,half,material={'base':'steel','connector':'orange','cap':'blue'}[name],
                     density={'base':100,'connector':250,'cap':150}[name],group='3')
        ET.SubElement(body,'geom',name=name+'_visual',mesh=name+'_visual',type='mesh',
                      material={'base':'steel','connector':'orange','cap':'blue'}[name],contype='0',conaffinity='0',mass='0',group='2')
        ET.SubElement(body,'site',name=name+'_grasp',pos=' '.join(map(str,GRASP[name])),size='.001',rgba='0 0 0 0')
    ET.SubElement(world,'light',name='key',pos='1 1 2',dir='-.3 -.3 -1',diffuse='.7 .7 .7',castshadow='false')
    ET.SubElement(world,'light',name='fill',pos='-1 0 1',dir='.4 0 -1',diffuse='.4 .4 .4',castshadow='false')
    ET.SubElement(world,'camera',name='overview',pos='.65 -.95 .8',fovy='43')
    root.find('visual/global').set('offwidth','1280'); root.find('visual/global').set('offheight','960')
    return ET.tostring(root,encoding='unicode'), positions, destinations
