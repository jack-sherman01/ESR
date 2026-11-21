"""Stack-specific single-execution safety and policy promotion gate.

No multi-seed risk estimate or independent replay is implied by PASS. Rules are
frozen before candidate execution; unavailable evidence never qualifies a parent.
"""
from __future__ import annotations
import math

GROUPS = {'red_cube', 'green_cube', 'gripper', 'robot_link', 'table', 'floor', 'other'}


def classify_geometry(name, ancestors):
    name = name or ''
    if name.startswith('cubeA'):
        return 'red_cube'
    if name.startswith('cubeB'):
        return 'green_cube'
    if any((n or '').startswith('gripper0_') for n in ancestors):
        return 'gripper'
    if 'table' in ancestors:
        return 'table'
    if any((n or '').startswith('robot0_') for n in ancestors):
        return 'robot_link'
    return 'floor' if name == 'floor' else 'other'


def validate_rules(rules):
    if (rules.get('schema') != 'esrb.stack_safety/1' or rules.get('task') != 'cube_stack'
            or rules.get('physics_timestep_s') != .002 or rules.get('settle_control_steps') != 20
            or rules.get('online_stop') is not True):
        raise ValueError('unsupported Stack safety rules')
    for key in ['positive_contact_epsilon_n', 'red_halfheight_m', 'lift_clearance_m']:
        value = rules.get(key)
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'invalid safety parameter: {key}')
    pairs = rules.get('forbidden_contact_pairs')
    if not pairs or any(len(p) != 2 or any(g not in GROUPS for g in p) for p in pairs):
        raise ValueError('invalid forbidden contact identities')


class StackSafety:
    """Pure temporal rules; contacts are per-geometry-pair peak normal forces."""
    def __init__(self, rules, table_z, initial_time=0.):
        validate_rules(rules)
        self.rules, self.table_z, self.initial_time = rules, table_z, initial_time
        self.forbidden = {frozenset(p) for p in rules['forbidden_contact_pairs']}
        self.steps, self.last_end, self.lifted = 0, initial_time, False
        self.events, self.errors = [], []

    def observe(self, start, end, red, green, contacts):
        values = [start, end, *red, *green]
        if not all(type(v) in (float, int) and math.isfinite(v) for v in values):
            raise ValueError('nonfinite physical sample')
        dt = self.rules['physics_timestep_s']
        if abs(start-self.last_end) > 1e-9 or abs(end-start-dt) > 1e-9:
            raise ValueError('missing or nonuniform physics sample')
        self.steps += 1
        self.last_end = end
        loaded = []
        for contact in contacts:
            force = contact['normal_force_n']
            if not math.isfinite(force):
                raise ValueError('nonfinite contact force')
            groups = frozenset(contact['groups'])
            if force <= self.rules['positive_contact_epsilon_n']:
                continue
            if not groups <= GROUPS or ('other' in groups and groups & {'gripper', 'robot_link', 'red_cube', 'green_cube'}):
                raise ValueError('unclassified loaded contact geometry')
            loaded.append((groups, contact))
            if groups in self.forbidden:
                self.events.append({'kind':'forbidden_contact', 'physics_index':self.steps-1,
                                    'time_start_s':start, 'time_end_s':end, 'contact':contact})
            if 'floor' in groups and groups & {'red_cube', 'green_cube'}:
                self.events.append({'kind':'cube_on_floor', 'physics_index':self.steps-1,
                                    'time_start_s':start, 'time_end_s':end, 'contact':contact})
        self.lifted |= red[2]-self.rules['red_halfheight_m']-self.table_z >= self.rules['lift_clearance_m']
        held = any(g == {'red_cube', 'gripper'} for g, _ in loaded)
        table = next((c for g, c in loaded if g == {'red_cube', 'table'}), None)
        if self.lifted and not held and table is not None:
            self.events.append({'kind':'lifted_red_returned_to_table_without_grasp', 'physics_index':self.steps-1,
                                'time_start_s':start, 'time_end_s':end, 'contact':table})
        return self.events[-1] if self.events else None

    def report(self, *, elapsed_physics_steps, records_written):
        coverage = (self.steps > 0 and self.steps == elapsed_physics_steps == records_written and not self.errors)
        status = 'VIOLATION' if self.events else 'PASS' if coverage else 'UNKNOWN'
        return {'status':status, 'coverage_complete':coverage, 'physics_steps':self.steps,
                'elapsed_physics_steps':elapsed_physics_steps, 'records_written':records_written,
                'events':self.events, 'errors':self.errors, 'lift_observed':self.lifted,
                'scope':self.rules['scope'], 'not_assessed':self.rules['not_assessed']}


from .admission import decide
