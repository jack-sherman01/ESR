"""Physics-step TwoArmHandover ordering, drop and contact measurements."""
from __future__ import annotations
import math,re

FINGER=re.compile(r'^gripper([01])_finger([12])(?:_pad)?_collision$')

def classify_geometry(name,ancestors):
    name=name or ''
    if name.startswith('hammer_'):return 'hammer_handle' if name=='hammer_handle' else 'hammer_other'
    for n in ancestors:
        if n.startswith('gripper0_'):return 'primary_gripper'
        if n.startswith('gripper1_'):return 'secondary_gripper'
    if any(n.startswith('robot0_') for n in ancestors):return 'primary_link'
    if any(n.startswith('robot1_') for n in ancestors):return 'secondary_link'
    if 'table' in ancestors:return 'table'
    return 'floor' if name=='floor' else 'other'


def validate_rules(r):
    if (r.get('schema')!='esrb.handover_safety/1' or r.get('task')!='two_arm_handover'
        or r.get('physics_timestep_s')!=.002 or r.get('settle_control_steps')!=20):raise ValueError('unsupported handover rules')
    for key in ['positive_contact_epsilon_n','grasp_dwell_s','release_dwell_s','pickup_rise_m']:
        v=r.get(key)
        if type(v) not in (float,int) or not math.isfinite(v) or v<=0:raise ValueError('invalid '+key)


class HandoverSafety:
    def __init__(self,rules,initial_hammer_height,initial_time=0.):
        validate_rules(rules)
        self.rules,self.initial_height,self.initial_time=rules,initial_hammer_height,initial_time
        self.steps,self.last_end=0,initial_time
        self.events,self.errors,self.precursors=[],[],[]
        self.stages={};self.giver_dwell=self.receiver_dwell=self.release_dwell=0.
        self.release_start=None;self.release_receiver_stable=False;self.picked_up=False
        self.commands={'primary':None,'secondary':None};self.command_events=[]
        self.receiver_close_time=self.giver_open_time=None
        self.receiver_dwell_at_open=None;self.max_receiver_dwell=0.
        self.last_giver=False;self.last_receiver=False;self.last_hammer_z=initial_hammer_height

    def command(self,t,targets,control_index):
        for arm,value in targets.items():
            if value is None:continue
            value=float(value);old=self.commands[arm];self.commands[arm]=value
            changed=old is None or abs(old-value)>1e-6
            if not changed:continue
            row={'time_s':float(t),'control_index':control_index,'arm':arm,'target':value}
            self.command_events.append(row)
            if arm=='secondary' and value<=.2:
                self.receiver_close_time=float(t);self.stages.setdefault('receiver_close_command',row)
            if arm=='primary' and value>=.8 and self.picked_up:
                self.giver_open_time=float(t);self.receiver_dwell_at_open=self.receiver_dwell
                self.stages.setdefault('giver_open_command',row)
                if self.receiver_dwell<self.rules['grasp_dwell_s']-1e-9:
                    self.precursors.append({'kind':'open_command_before_receiver_dwell','time_s':float(t),
                        'receiver_continuous_grasp_s':self.receiver_dwell,'control_index':control_index})

    def _event(self,kind,start,end,contact=None,**extra):
        if any(e['kind']==kind for e in self.events):return
        e={'kind':kind,'physics_index':self.steps-1,'time_start_s':start,'time_end_s':end,**extra}
        if contact is not None:e['contact']=contact
        self.events.append(e)

    def observe(self,start,end,hammer,contacts):
        if not all(type(v) in (float,int) and math.isfinite(v) for v in [start,end,*hammer]):raise ValueError('nonfinite physical sample')
        dt=self.rules['physics_timestep_s']
        if abs(start-self.last_end)>1e-9 or abs(end-start-dt)>1e-9:raise ValueError('missing or nonuniform physics sample')
        self.steps+=1;self.last_end=end;pads=[set(),set()];support=None
        before=len(self.events)
        for c in contacts:
            f=c['normal_force_n']
            if not math.isfinite(f):raise ValueError('nonfinite contact force')
            if f<=self.rules['positive_contact_epsilon_n']:continue
            g=set(c['groups']);names=c['names']
            if any(x.startswith('primary_') for x in g) and any(x.startswith('secondary_') for x in g):
                self._event('inter_arm_collision',start,end,c)
            if g&{'table','floor'} and g&{'primary_gripper','secondary_gripper','primary_link','secondary_link'}:
                self._event('robot_support_collision',start,end,c)
            if 'other' in g and g&{'hammer_handle','hammer_other','primary_gripper','secondary_gripper','primary_link','secondary_link'}:
                raise ValueError('unclassified loaded contact geometry')
            if g&{'hammer_handle','hammer_other'} and g&{'table','floor'}:support=c
            for name,other in [names,names[::-1]]:
                finger=FINGER.fullmatch(name or '')
                if finger and (other or '').startswith('hammer_') and (finger[1]=='0' or other=='hammer_handle'):
                    pads[int(finger[1])].add(finger[2])
        giver,receiver=(len(p)==2 for p in pads)
        self.giver_dwell=self.giver_dwell+dt if giver else 0.
        receiver_before=self.receiver_dwell
        self.receiver_dwell=self.receiver_dwell+dt if receiver else 0.
        self.max_receiver_dwell=max(self.max_receiver_dwell,self.receiver_dwell)
        lifted=hammer[2]-self.initial_height>=self.rules['pickup_rise_m']
        if self.giver_dwell>=self.rules['grasp_dwell_s']-1e-9:self.stages.setdefault('giver_grasp',start)
        if giver and lifted and support is None and self.giver_dwell>=self.rules['grasp_dwell_s']-1e-9:
            self.picked_up=True;self.stages.setdefault('pickup',start)
        if self.picked_up and receiver and self.receiver_dwell>=self.rules['grasp_dwell_s']-1e-9:
            self.stages.setdefault('receiver_grasp_stable',start)
        if self.picked_up and giver and receiver:self.stages.setdefault('overlap_start',start)
        if self.picked_up and not giver:
            if self.release_dwell==0:
                self.release_start=start
                self.release_onset_index=self.steps-1
                self.release_receiver_stable=receiver and receiver_before>=self.rules['grasp_dwell_s']-1e-9
            self.release_dwell+=dt
            if self.release_dwell>=self.rules['release_dwell_s']-1e-9 and 'giver_release' not in self.stages:
                self.stages['giver_release']=self.release_start
                if not self.release_receiver_stable:
                    self._event('premature_release',self.release_start,end,receiver_stable_before_loss=False,onset_physics_index=self.release_onset_index)
        else:self.release_dwell=0.
        if self.picked_up and support is not None and not giver and not receiver:
            self._event('hammer_drop',start,end,support)
        self.last_giver,self.last_receiver,self.last_hammer_z=giver,receiver,hammer[2]
        return self.events[before:]

    def report(self,*,elapsed_physics_steps,records_written):
        coverage=self.steps>0 and self.steps==elapsed_physics_steps==records_written and not self.errors
        close=self.stages.get('receiver_close_command');opened=self.stages.get('giver_open_command')
        wait=(opened['time_s']-close['time_s']) if opened and close else None
        overlap=(self.stages['giver_release']-self.stages['overlap_start']) if 'giver_release' in self.stages and 'overlap_start' in self.stages else None
        return {'status':'VIOLATION' if self.events else 'PASS' if coverage else 'UNKNOWN',
            'coverage_complete':coverage,'physics_steps':self.steps,'elapsed_physics_steps':elapsed_physics_steps,
            'records_written':records_written,'events':self.events,'errors':self.errors,'precursors':self.precursors,
            'command_events':self.command_events,'stages':self.stages,'pickup_observed':self.picked_up,
            'receiver_wait_before_open_s':wait,'receiver_grasp_dwell_at_open_s':self.receiver_dwell_at_open,
            'overlap_before_release_s':overlap,'maximum_receiver_grasp_dwell_s':self.max_receiver_dwell,
            'scope':self.rules['scope'],'not_assessed':self.rules['not_assessed']}
