"""Frozen experiment design. Importing this module requires no simulator."""
SCENES = (
    'tool_handling', 'long_horizon_assembly', 'moving_transport', 'moving_assembly',
    'constrained_push', 'force_alignment', 'compliant_assembly', 'stack',
    'pickplacecan', 'two-arm-handover', 'nut-assembly', 'wipe',
)
NATIVE = SCENES[7:]
JAM = ('long_horizon_assembly', 'moving_assembly', 'constrained_push',
       'force_alignment', 'compliant_assembly', 'nut-assembly')
NO_GRASP = ('constrained_push', 'wipe')
SCHEMA = 'measurement-validation/1'


def event_profile(version=6):
    import json
    from esrb.runtime.paths import project_root
    if version not in (5,6,7):raise ValueError('unsupported measurement event profile')
    return json.loads((project_root()/f'configs/evaluation/measurement_validation_v{version}.json').read_text())


def primary(scene):
    return 'robot_contact' if scene in NO_GRASP else 'drop'


def case_status(scene,case,measurement,limits):
    if case in ('baseline','seated_release','task_failure','legal_hold'):
        return measurement['status']
    target=primary(scene) if case=='primary' else case
    if measurement['missing']:return 'UNKNOWN'
    # A witnessed crossing of its frozen calibrated threshold remains known
    # even when unrelated load channels have no qualified limit.
    targets={target,'grasp_loss'} if target=='drop' else {target}
    if any(e['kind'] in targets for e in measurement['events']):return 'TRIGGERED'
    if case in ('overload','eccentric_torque','jam') and limits.get('status')!='QUALIFIED':return 'UNKNOWN'
    return 'NOT_TRIGGERED'


def cases(scene):
    # baseline and the primary intervention double as main diagnostic episodes.
    values = ['baseline', 'primary', 'task_failure', 'robot_contact', 'overload', 'eccentric_torque']
    if primary(scene) == 'robot_contact':
        values.remove('robot_contact')
    if scene not in NO_GRASP:
        values.append('seated_release')
    if scene in JAM:
        values.extend(('jam', 'legal_hold'))
    return values


def protocol(scenes, workers):
    return dict(schema=SCHEMA, event_profile=event_profile(), scenes=list(scenes), model_calls=0,
                scene_event_profiles={'wipe':event_profile(7)} if 'wipe' in scenes else {},
                generations=3, repetitions=5, alpha=.10, epsilon=.09,
                main_seed_formula='42 + 1000 * generation + repetition; generation=0,1,2',
                diagnostic_seeds=list(range(12)), calibration_seeds=[42,43,44],
                calibration_max_per_scene=36, workers=workers,
                nominal_limit_multiplier=1.5, sustained_window_s=.1,
                minimum_jam_s=1., main_execution='scene-parallel; generations and episodes sequential within scene',
                primary_rules={s:primary(s) for s in scenes},
                historical_metrics='esrb.metrics.core legacy v1; controlled-label detection',
                privilege='deterministic engineering controllers may read native state and Jacobians',
                scope='simulation engineering envelopes, not material damage or real robot limits')
