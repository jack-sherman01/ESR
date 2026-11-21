"""Original ESR fixtures; task ideas reference ManiSkill, no upstream assets copied."""
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np
from esrb.tasks.robots.panda import robot_tree
from esrb.tasks.production.geometry import add_geom,chamfer_box
ASSETS=Path(__file__).parent/'assets'
PARTS={
 'constrained_push':[((0,0,0),(.045,.032,.016),'steel'),((0,0,.018),(.033,.023,.002),'blue')],
 'force_alignment':[((0,0,0),(.022,.05,.012),'orange'),((0,0,.055),(.018,.014,.035),'steel')],
 'compliant_assembly':[((0,0,0),(.07,.009,.009),'steel'),((-.05,0,.016),(.018,.014,.024),'orange')],
}
GRASP={'force_alignment':np.array([0.,0.,.065]),'compliant_assembly':np.array([-.05,0.,.032])}
START={'constrained_push':np.array([-.23,.10,.036]),'force_alignment':np.array([-.22,-.10,.035]),'compliant_assembly':np.array([-.22,-.10,.049])}
GOAL={'constrained_push':np.array([.20,.10,.036]),'force_alignment':np.array([.16,.12,.034]),'compliant_assembly':np.array([.14,.12,.075])}

def build_assets():
 for name,boxes in PARTS.items():
  for i,(pos,size,_) in enumerate(boxes):
   v,f=chamfer_box(np.array(pos),np.array(size),bevel=.0007)
   text=['# ESR original chamfered contact workpiece display mesh']
   text+=['v '+' '.join(map(str,p)) for p in v]
   text+=['f '+' '.join(map(str,p+1)) for p in f]
   (ASSETS/f'{name}_{i}.obj').write_text('\n'.join(text)+'\n')

def bolt(world,name,position):
 ET.SubElement(world,'geom',name=name,type='cylinder',pos=' '.join(map(str,position)),size='.004 .002',material='steel',contype='0',conaffinity='0',mass='0')
 ET.SubElement(world,'geom',name=name+'_socket',type='cylinder',pos=' '.join(map(str,np.array(position)+[0,0,.0021])),size='.0018 .0002',rgba='.09 .11 .13 1',contype='0',conaffinity='0',mass='0')

def make_scene(name):
 root=robot_tree();root.set('model',name+'_panda_v2');asset=root.find('asset');world=root.find('worldbody')
 colors={'steel':'.66 .71 .76 1','blue':'.08 .26 .43 1','orange':'.95 .40 .08 1','bench':'.78 .80 .80 1','rubber':'.12 .15 .18 1','green':'.13 .63 .43 1'}
 for key,color in colors.items():
  old=asset.find(f"material[@name='{key}']")
  if old is None:old=ET.SubElement(asset,'material',name=key)
  old.set('rgba',color);old.set('specular','.35');old.set('shininess','.3')
  for attribute in ('texture','texrepeat','reflectance'):old.attrib.pop(attribute,None)
 add_geom(world,'workbench',[0,0,-.025],[.85,.58,.025],material='bench')
 ET.SubElement(world,'geom',name='floor',type='plane',pos='0 0 -.75',size='3 3 .02',rgba='.84 .85 .85 1')
 if name=='constrained_push':
  add_geom(world,'track_floor',[0,.10,.01],[.34,.075,.01],material='steel')
  for s in (-1,1):
   add_geom(world,'guide_'+str(s),[-.12,.10+s*.048,.028],[.22,.010,.028],material='blue')
   add_geom(world,'guide_dock_'+str(s),[.22,.10+s*.044,.028],[.06,.010,.028],material='blue')
   add_geom(world,'guide_taper_'+str(s),[.13,.10+s*.046,.028],[.032,.010,.028],material='blue',euler=f'0 0 {-s*.067}')
   for x in (-.30,0,.30):bolt(world,f'rail_bolt_{s}_{x}',[x,.10+s*.048,.058])
  for y in (.065,.135):add_geom(world,'dock_stripe_'+str(y),[.20,y,.0205],[.049,.002,.0005],material='green',contype='0',conaffinity='0')
  add_geom(world,'dock_stop',[.258,.10,.028],[.006,.038,.008],material='blue')
 elif name=='force_alignment':
  add_geom(world,'fixture_base',[.16,.07,.01],[.09,.16,.01],material='blue')
  add_geom(world,'datum_bed',[.16,.07,.021],[.026,.15,.001],material='steel')
  for s in (-1,1):
   add_geom(world,'guide_'+str(s),[.16+s*.033,.12,.032],[.009,.10,.012],material='steel')
   add_geom(world,'guide_funnel_'+str(s),[.16+s*.044,-.006,.032],[.009,.037,.012],material='steel',euler=f'0 0 {s*.24}')
   for y in (.04,.20):bolt(world,f'fixture_bolt_{s}_{y}',[.16+s*.066,y,.022])
  add_geom(world,'datum_stop',[.16,.180,.036],[.026,.006,.016],material='blue')
  add_geom(world,'part_rack',[-.22,-.10,.0115],[.065,.078,.0115],material='blue')
 else:
  add_geom(world,'receiver_base',[.20,.12,.015],[.10,.085,.015],material='blue')
  add_geom(world,'flange_cradle',[.09,.12,.0475],[.023,.025,.0175],material='blue')
  # Four independent solids form an actual through-hole along world X.
  for s in (-1,1):
   add_geom(world,'guide_side_'+str(s),[.20,.12+s*.027,.075],[.06,.016,.045],material='steel')
   add_geom(world,'guide_cap_'+str(s),[.20,.12,.075+s*.028],[.06,.011,.017],material='steel')
   for x in (.13,.27):bolt(world,f'receiver_bolt_{s}_{x}',[x,.12+s*.065,.032])
  add_geom(world,'part_rack',[-.22,-.10,.020],[.09,.045,.020],material='blue')
 body=ET.SubElement(world,'body',name='workpiece',pos=' '.join(map(str,START[name])))
 ET.SubElement(body,'freejoint',name='workpiece_free')
 for i,(p,h,material) in enumerate(PARTS[name]):
  add_geom(body,f'part_collision_{i}',p,h,material=material,density='450',group='3',friction='.9 .005 .0001')
  mesh=f'{name}_{i}';ET.SubElement(asset,'mesh',name=mesh,file=str(ASSETS/(mesh+'.obj')))
  ET.SubElement(body,'geom',name=f'part_visual_{i}',type='mesh',mesh=mesh,material=material,contype='0',conaffinity='0',mass='0',group='2')
 ET.SubElement(body,'site',name='part_grasp',pos=' '.join(map(str,GRASP.get(name,np.zeros(3)))),rgba='0 0 0 0',size='.001')
 ET.SubElement(world,'light',name='key',pos='1 1 2',dir='-.3 -.3 -1',diffuse='.8 .8 .8',castshadow='false')
 ET.SubElement(world,'light',name='fill',pos='-1 0 1',dir='.4 0 -1',diffuse='.5 .5 .5',castshadow='false')
 ET.SubElement(world,'camera',name='overview',pos='.7 .9 .8',fovy='38')
 return ET.tostring(root,encoding='unicode')
