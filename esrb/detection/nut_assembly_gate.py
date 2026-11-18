"""Bounded online NutAssemblySquare contact/drop rules; no load limit."""
import math

GROUPS = {'square_nut', 'square_peg', 'round_peg', 'inactive_round_nut', 'gripper', 'robot_link', 'table', 'floor', 'other'}


def validate_rules(rules):
    if (rules.get('schema') != 'esrb.nut_safety/1' or rules.get('task') != 'nut_assembly'
            or rules.get('physics_timestep_s') != .002 or rules.get('settle_control_steps') != 20
            or rules.get('online_stop') is not True):
        raise ValueError('unsupported nut-assembly safety rules')
    for name in ['positive_contact_epsilon_n', 'nut_halfthickness_m', 'lift_clearance_m',
                 'peg_halfwidth_m', 'nut_hole_halfwidth_m', 'seat_tolerance_m', 'max_tilt_rad']:
        if type(rules.get(name)) not in (int, float) or not math.isfinite(rules[name]) or rules[name] <= 0:
            raise ValueError('invalid geometry or timing parameter: ' + name)
    if not rules.get('forbidden_contact_pairs') or any(len(p) != 2 or not set(p) <= GROUPS for p in rules['forbidden_contact_pairs']):
        raise ValueError('invalid contact geometry identity')


class NutSafety:
    def __init__(self, rules, table_z, initial_time=0.):
        validate_rules(rules)
        self.rules, self.table_z, self.initial_time = rules, table_z, initial_time
        self.forbidden = {frozenset(p) for p in rules['forbidden_contact_pairs']}
        self.steps, self.last_end, self.lifted = 0, initial_time, False
        self.events, self.errors = [], []

    def observe(self, start, end, nut, contacts, *, seated):
        if not all(type(v) in (int, float) and math.isfinite(v) for v in [start, end, *nut]):
            raise ValueError('nonfinite physical sample')
        if abs(start-self.last_end) > 1e-9 or abs(end-start-self.rules['physics_timestep_s']) > 1e-9:
            raise ValueError('missing or nonuniform physics sample')
        self.steps += 1
        self.last_end = end
        loaded = []
        for contact in contacts:
            force = contact['normal_force_n']
            if not math.isfinite(force):
                raise ValueError('nonfinite contact force')
            if force <= self.rules['positive_contact_epsilon_n']:
                continue
            groups = frozenset(contact['groups'])
            if not groups <= GROUPS or ('other' in groups and groups & {'gripper', 'robot_link', 'square_nut'}):
                raise ValueError('unclassified loaded contact geometry')
            loaded.append((groups, contact))
            kind = ('forbidden_contact' if groups in self.forbidden else
                    'nut_on_floor' if groups == {'square_nut', 'floor'} else None)
            if kind:
                self.events.append({'kind': kind, 'physics_index': self.steps-1,
                                    'time_start_s': start, 'time_end_s': end, 'contact': contact})
        held = any(g == {'square_nut', 'gripper'} for g, _ in loaded)
        self.lifted |= held and nut[2]-self.rules['nut_halfthickness_m']-self.table_z >= self.rules['lift_clearance_m']
        support = next((c for g, c in loaded if g == {'square_nut', 'table'}), None)
        if self.lifted and not held and support is not None and not seated:
            self.events.append({'kind': 'lifted_nut_returned_to_table_without_grasp', 'physics_index': self.steps-1,
                                'time_start_s': start, 'time_end_s': end, 'contact': support})
        return self.events[-1] if self.events else None

    def report(self, *, elapsed_physics_steps, records_written):
        coverage = self.steps > 0 and self.steps == elapsed_physics_steps == records_written and not self.errors
        return {'status': 'VIOLATION' if self.events else 'PASS' if coverage else 'UNKNOWN',
                'coverage_complete': coverage, 'physics_steps': self.steps,
                'elapsed_physics_steps': elapsed_physics_steps, 'records_written': records_written,
                'events': self.events, 'errors': self.errors, 'lift_observed': self.lifted,
                'scope': self.rules['scope'], 'not_assessed': self.rules['not_assessed']}
