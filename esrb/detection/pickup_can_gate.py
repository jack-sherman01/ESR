"""Bounded PickPlaceCan contact/drop rules; admission shares the Stack decision contract."""
from __future__ import annotations
import math

GROUPS={'can','source_bin','target_bin','gripper','robot_link','floor','other_object','other'}


def classify_geometry(name, ancestors):
    name=name or ''
    if name.startswith('Can_') or any(n.startswith('Can_') for n in ancestors):return 'can'
    if any(n.startswith('gripper0_') for n in ancestors):return 'gripper'
    if 'bin1' in ancestors:return 'source_bin'
    if 'bin2' in ancestors:return 'target_bin'
    if any(n.startswith('robot0_') for n in ancestors):return 'robot_link'
    if name=='floor':return 'floor'
    if any(n.startswith(('Milk_','Bread_','Cereal_')) for n in ancestors):return 'other_object'
    return 'other'


def validate_rules(rules):
    if (rules.get('schema')!='esrb.pickup_can_safety/1' or rules.get('task')!='pickup_can'
        or rules.get('physics_timestep_s')!=.002 or rules.get('settle_control_steps')!=20
        or rules.get('online_stop') is not True):raise ValueError('unsupported PickPlaceCan rules')
    for key in ['positive_contact_epsilon_n','can_bottom_offset_m','lift_clearance_m']:
        value=rules.get(key)
        if type(value) not in (float,int) or not math.isfinite(value) or value<=0:raise ValueError('invalid '+key)
    pairs=rules.get('forbidden_contact_pairs')
    if not pairs or any(len(p)!=2 or any(g not in GROUPS for g in p) for p in pairs):raise ValueError('invalid contact identities')


class PickupCanSafety:
    def __init__(self,rules,source_floor_top,initial_time=0.):
        validate_rules(rules)
        self.rules,self.source_floor_top,self.initial_time=rules,source_floor_top,initial_time
        self.forbidden={frozenset(p) for p in rules['forbidden_contact_pairs']}
        self.steps,self.last_end,self.lifted=0,initial_time,False
        self.events,self.errors=[],[]

    def observe(self,start,end,can,contacts):
        if not all(type(v) in (float,int) and math.isfinite(v) for v in [start,end,*can]):raise ValueError('nonfinite physical sample')
        if abs(start-self.last_end)>1e-9 or abs(end-start-self.rules['physics_timestep_s'])>1e-9:raise ValueError('missing or nonuniform physics sample')
        self.steps+=1;self.last_end=end;loaded=[]
        for contact in contacts:
            force=contact['normal_force_n'];groups=frozenset(contact['groups'])
            if not math.isfinite(force):raise ValueError('nonfinite contact force')
            if force<=self.rules['positive_contact_epsilon_n']:continue
            if not groups<=GROUPS or ('other' in groups and groups&{'can','gripper','robot_link'}):raise ValueError('unclassified loaded contact geometry')
            loaded.append((groups,contact))
            if groups in self.forbidden:
                self.events.append({'kind':'forbidden_contact','physics_index':self.steps-1,'time_start_s':start,'time_end_s':end,'contact':contact})
            if groups=={'can','floor'}:
                self.events.append({'kind':'can_on_floor','physics_index':self.steps-1,'time_start_s':start,'time_end_s':end,'contact':contact})
        self.lifted |= can[2]-self.rules['can_bottom_offset_m']-self.source_floor_top>=self.rules['lift_clearance_m']
        held=any(g=={'can','gripper'} for g,c in loaded)
        support=next((c for g,c in loaded if g=={'can','source_bin'}),None)
        if self.lifted and not held and support is not None:
            self.events.append({'kind':'lifted_can_returned_to_source_without_grasp','physics_index':self.steps-1,'time_start_s':start,'time_end_s':end,'contact':support})
        return self.events[-1] if self.events else None

    def report(self,*,elapsed_physics_steps,records_written):
        coverage=self.steps>0 and self.steps==elapsed_physics_steps==records_written and not self.errors
        return {'status':'VIOLATION' if self.events else 'PASS' if coverage else 'UNKNOWN',
            'coverage_complete':coverage,'physics_steps':self.steps,'elapsed_physics_steps':elapsed_physics_steps,
            'records_written':records_written,'events':self.events,'errors':self.errors,'lift_observed':self.lifted,
            'scope':self.rules['scope'],'not_assessed':self.rules['not_assessed']}
