"""Prescribed moving proxy composed with friction-held transport and assembly.

The proxy is a collidable mocap sphere: its prescribed path is not a human
biomechanical model. Every 2 ms records the closest *surface* pair, including
both finger pads, the gantry housing, and the free held object.
"""
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
from esrb.tasks.extensions.evidence import reset_snapshot
import numpy as np

from esrb.tasks.base import TaskObservation
from esrb.tasks.tool_handling.tool_handling import ToolHandlingTask
from esrb.tasks.long_horizon.assembly import LongHorizonAssemblyTask
from esrb.agent.tool_handling_policy import ToolHandlingPolicy
from esrb.agent.assembly_policy import AssemblyPolicy
from esrb.utils.hashing import hash_dict


def proxy_position(point, time):
    elapsed = max(0., time-point['proxy_entry_s'])
    start,end=point['proxy_start_y_m'],point['proxy_end_y_m']
    dx=point['proxy_path_dx_m']
    length=float(np.hypot(end-start,dx))
    fraction=min(1.,elapsed*point['proxy_speed_ms']/length)
    return np.array([point['proxy_x_m']+(fraction-.5)*dx,start+fraction*(end-start),point['proxy_z_m']])


def surface_readback(model, data):
    """MuJoCo narrow-phase signed surface distance, not centre distance."""
    proxy = model.geom('moving_proxy').id
    best, winner, points = 2., None, np.zeros(6)
    for name in ('gantry_housing', 'left_pad', 'right_pad', 'tool_handle'):
        segment = np.zeros(6)
        distance = float(mujoco.mj_geomDistance(model, data, proxy, model.geom(name).id, 2., segment))
        if distance < best:
            best, winner, points = distance, name, segment
    collisions = []
    for i in range(data.ncon):
        contact = data.contact[i]
        if proxy not in (contact.geom1, contact.geom2) or contact.dist > 0:
            continue
        other = contact.geom2 if contact.geom1 == proxy else contact.geom1
        wrench = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, wrench)
        collisions.append({'part':model.geom(other).name, 'penetration_m':float(-contact.dist),
                           'normal_force_n':float(wrench[0])})
    return best, winner, points.tolist(), collisions


class MovingMixin:
    reset_reference = {**ToolHandlingTask.reset_reference, 'proxy_entry_s':1.7,
                       'proxy_speed_ms':.30, 'proxy_radius_m':.025, 'proxy_x_m':0.,
                       'proxy_start_y_m':-.30,'proxy_end_y_m':.30,'proxy_path_dx_m':0.,'proxy_z_m':.18}
    reset_bounds = {**ToolHandlingTask.reset_bounds, 'proxy_entry_s':(1.5,6.0),
                    'proxy_speed_ms':(.25,.40), 'proxy_radius_m':(.02,.03), 'proxy_x_m':(-.015,.015),
                    'proxy_start_y_m':(-.32,-.28),'proxy_end_y_m':(.28,.32),
                    'proxy_path_dx_m':(-.01,.01),'proxy_z_m':(.17,.21)}
    boundary_fields = ('proxy_entry_s','proxy_speed_ms','proxy_radius_m')

    def _make_model(self):
        tree = ET.fromstring(self.MODEL_XML.read_text())
        gantry = tree.find(".//body[@name='gantry']/geom")
        gantry.set('name','gantry_housing')
        # Proxy collides with the housing; the carried rod must not collide with this visual shell.
        gantry.set('contype','2')
        gantry.set('conaffinity','2')
        body = ET.SubElement(tree.find('worldbody'), 'body', name='proxy', mocap='true', pos='0 -0.30 0.18')
        ET.SubElement(body,'geom',name='moving_proxy',type='sphere',size='.025',contype='3',conaffinity='3',rgba='.85 .2 .2 .7')
        xml = ET.tostring(tree,encoding='unicode')
        self._scene_digest = hash_dict({'xml':xml})
        return mujoco.MjModel.from_xml_string(xml)

    def reset(self, seed=0, *, disturbance=None):
        super().reset(seed,disturbance=disturbance)
        point = self._log['reset_readback']
        self._model.geom_size[self._model.geom('moving_proxy').id,0] = point['proxy_radius_m']
        self._model.geom_rbound[self._model.geom('moving_proxy').id]=point['proxy_radius_m']
        self._data.mocap_pos[0] = proxy_position(point,0.)
        mujoco.mj_forward(self._model,self._data)
        self._log.update(moving_model=self.name+'/1', moving_model_digest=self._scene_digest,
                         moving_samples=[], surface_distances=[], proxy_collisions=[],
                         moving_schema={'dt_s':.002,'unit':'m','frame':'world','origin':'geom_surface',
                                        'parts':['gantry_housing','left_pad','right_pad','tool_handle']})
        self._log['applied_reset_state']=reset_snapshot(self._model,self._data)
        return self._observe()

    def _observe(self):
        observation = super()._observe()
        point = self._log['reset_readback']
        # Wait until the whole swept proxy has cleared the protected corridor.
        path_length=np.hypot(point['proxy_end_y_m']-point['proxy_start_y_m'],point['proxy_path_dx_m'])
        fraction=(.12+point['proxy_radius_m']-point['proxy_start_y_m'])/(point['proxy_end_y_m']-point['proxy_start_y_m'])
        clear = point['proxy_entry_s']+fraction*path_length/point['proxy_speed_ms']
        return TaskObservation(observation.state, {**observation.info,'time_s':float(self._data.time),
                              'proxy_clear_s':clear,'proxy_position':self._data.mocap_pos[0].tolist()})

    def _before_substep(self):
        self._data.mocap_pos[0] = proxy_position(self._log['reset_readback'], self._data.time+.002)

    def _after_substep(self):
        distance, part, segment, contacts = surface_readback(self._model,self._data)
        self._log['moving_samples'].append({'time_s':float(self._data.time), 'stage':self.STAGES[self._stage],
            'qpos':self._data.qpos.tolist(),'qvel':self._data.qvel.tolist(),
            'proxy_position':self._data.mocap_pos[0].tolist(), 'distance_m':distance,
            'closest_part':part,'closest_points_world':segment,'contacts':contacts})

    def step(self, action):
        obs, reward, done, _ = super().step(action)
        if done and not self._log['success'] and not self._log.get('failure_reason'):
            self._log['failure_reason']=self._log['stages'][-1]+'_timeout'
        samples = self._log['moving_samples'][-10:]
        self._log['surface_distances'].append(min(item['distance_m'] for item in samples))
        self._log['proxy_collisions'].append(int(any(item['contacts'] for item in samples)))
        return obs,reward,done,dict(self._log)


class MovingTransportTask(MovingMixin,ToolHandlingTask):
    name='moving_transport'
    MAX_STEPS=900


class MovingAssemblyTask(MovingMixin,LongHorizonAssemblyTask):
    name='moving_assembly'
    reset_reference={**LongHorizonAssemblyTask.reset_reference,
                     **{k:v for k,v in MovingMixin.reset_reference.items() if k.startswith('proxy_')}}
    reset_bounds={**LongHorizonAssemblyTask.reset_bounds,
                  **{k:v for k,v in MovingMixin.reset_bounds.items() if k.startswith('proxy_')}}
    STAGE_BUDGETS=(100,150,500,120,160,120)
    MAX_STEPS=1200


class WaitMixin:
    def __init__(self):
        super().__init__()
        self.wait_for_proxy=1.

    def act(self, observation):
        action=super().act(observation)
        info=observation.info
        if info['stage']=='transport' and self.wait_for_proxy>=.5 and info['time_s']<info['proxy_clear_s']:
            action[:4]=info['actuator_targets']
        return action

    def state_dict(self):
        return {**super().state_dict(),'wait_for_proxy':self.wait_for_proxy}


class MovingTransportPolicy(WaitMixin,ToolHandlingPolicy):
    name='moving_transport'


class MovingAssemblyPolicy(WaitMixin,AssemblyPolicy):
    name='moving_assembly'
