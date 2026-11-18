"""Wipe single-execution admission gate. Loads are measured, never thresholded."""
from __future__ import annotations
import math

GROUPS = {'tool', 'robot_link', 'table', 'floor', 'marker', 'other'}


def classify_geometry(name, ancestors):
    name = name or ''
    # Only native wiping surfaces and corners qualify as the contact tool.
    if name.startswith(('gripper0_wiping_surface','gripper0_wiping_corner')):
        return 'tool'
    if name == 'floor':
        return 'floor'
    if 'table' in ancestors:
        return 'table'
    if any((n or '').startswith(('robot0_', 'gripper0_')) for n in ancestors):
        return 'robot_link'
    if name.startswith('marker'):
        return 'marker'
    return 'other'


def validate_rules(rules):
    if (rules.get('schema') != 'esrb.wipe_safety/1' or rules.get('task') != 'spill_wipe'
            or rules.get('physics_timestep_s') != .002 or rules.get('settle_control_steps') != 20
            or rules.get('online_stop') is not True or rules.get('load_overload_assessed') is not False):
        raise ValueError('unsupported Wipe safety rules')
    value = rules.get('positive_contact_epsilon_n')
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError('invalid contact numerical tolerance')
    pairs = rules.get('forbidden_contact_pairs')
    if {frozenset(p) for p in pairs} != {frozenset(p) for p in [('robot_link','table'),('robot_link','floor'),('tool','floor')]}:
        raise ValueError('invalid Wipe forbidden contacts')


class WipeSafety:
    def __init__(self, rules, initial_time=0.):
        validate_rules(rules)
        self.rules, self.last_end = rules, initial_time
        self.forbidden = {frozenset(p) for p in rules['forbidden_contact_pairs']}
        self.steps, self.events, self.errors = 0, [], []
        self.tool_table_steps, self.peak_tool_table_normal_n = 0, 0.
        self.peak_sensor_force_norm_n = self.peak_sensor_torque_norm_nm = 0.

    def observe(self, start, end, contacts, force, torque):
        values = [start, end, *force, *torque]
        if len(force) != 3 or len(torque) != 3 or not all(type(v) in (float,int) and math.isfinite(v) for v in values):
            raise ValueError('missing or nonfinite physical load sample')
        dt = self.rules['physics_timestep_s']
        if abs(start-self.last_end)>1e-9 or abs(end-start-dt)>1e-9:
            raise ValueError('missing or nonuniform physics sample')
        self.steps += 1; self.last_end = end
        total = 0.
        for contact in contacts:
            value = contact['normal_force_n']; groups = frozenset(contact['groups'])
            if not math.isfinite(value) or value < -self.rules['positive_contact_epsilon_n']:
                raise ValueError('invalid contact normal load')
            if not groups <= GROUPS:
                raise ValueError('unknown contact identity')
            if value <= self.rules['positive_contact_epsilon_n']:
                continue
            if 'other' in groups and groups & {'robot_link','tool'}:
                raise ValueError('unclassified robot contact')
            if groups == {'tool','table'}:
                total += value
            if groups in self.forbidden:
                self.events.append({'kind':'forbidden_contact','physics_index':self.steps-1,
                    'time_start_s':start,'time_end_s':end,'contact':contact})
        self.tool_table_steps += int(total > self.rules['positive_contact_epsilon_n'])
        self.peak_tool_table_normal_n = max(self.peak_tool_table_normal_n,total)
        self.peak_sensor_force_norm_n = max(self.peak_sensor_force_norm_n,math.sqrt(sum(x*x for x in force)))
        self.peak_sensor_torque_norm_nm = max(self.peak_sensor_torque_norm_nm,math.sqrt(sum(x*x for x in torque)))
        return self.events[-1] if self.events else None

    def report(self, *, elapsed_physics_steps, records_written):
        coverage = self.steps>0 and self.steps==elapsed_physics_steps==records_written and not self.errors
        return {'status':'VIOLATION' if self.events else 'PASS' if coverage else 'UNKNOWN',
            'coverage_complete':coverage,'physics_steps':self.steps,'elapsed_physics_steps':elapsed_physics_steps,
            'records_written':records_written,'events':self.events,'errors':self.errors,
            'tool_table_contact_physics_steps':self.tool_table_steps,
            'peak_tool_table_normal_force_n':self.peak_tool_table_normal_n,
            'peak_raw_sensor_force_norm_n':self.peak_sensor_force_norm_n,
            'peak_raw_sensor_torque_norm_nm':self.peak_sensor_torque_norm_nm,
            'load_overload_assessed':False,'scope':self.rules['scope'],'not_assessed':self.rules['not_assessed']}


def decide(result, *, active, binding_ok, expected_settle_steps):
    safety=result.get('safety') or {}; reason='accepted'
    if not binding_ok: reason='binding_or_integrity_error'
    elif result.get('status')!='MEASURED': reason='execution_evidence_unavailable'
    elif safety.get('status')=='VIOLATION': reason='observed_safety_violation'
    elif safety.get('status')!='PASS' or safety.get('coverage_complete') is not True: reason='unknown_safety_evidence'
    elif result.get('program_ok') is not True or result.get('native_task_success') is not True: reason='task_not_qualified'
    elif not (result.get('terminal_native_success') is True and result.get('native_success_terminated') is True):
        if result.get('settle_steps')!=expected_settle_steps or result.get('settle_success_steps')!=expected_settle_steps:
            reason='terminal_goal_not_maintained_or_missing_hold'
    accepted=reason=='accepted'
    return {'accepted':accepted,'decision':'PROMOTE' if accepted else 'REJECT','reason':reason,
        'parent_before':active,'retained_parent':not accepted and active is not None,
        'observed_safety_regression':bool(active is not None and binding_ok and safety.get('status')=='VIOLATION'),
        'task_success':result.get('native_task_success'),'safety_status':safety.get('status','UNKNOWN'),
        'terminal_hold_status':result.get('terminal_hold_status','UNKNOWN'),
        'scope':'same-seed single-execution engineering gate; no independent retest; loads recorded without overload assessment'}
