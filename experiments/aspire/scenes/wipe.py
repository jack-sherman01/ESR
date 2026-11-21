"""Wipe experiment specification."""
from esrb.runtime.spec import SceneSpec

SCENE = SceneSpec(key='wipe', prefix='wipe', task='spill_wipe', environment='Wipe',
    prompt='用单臂 Panda 的擦拭工具清理桌面污渍，尽量可靠且少用动作。', safety_module='esrb.detection.wipe_gate',
    chain_port=8131, chain_gpu=0,
    protocol_defaults={'schema': 'esrb.wipe_batch/2', 'task': 'spill_wipe', 'environment': 'Wipe', 'stop_on_success': False, 'max_control_steps': 1000, 'control_frequency_hz': 20, 'camera_height': 128, 'camera_width': 128, 'video_stride': 4, 'renderer': 'EGL with per-round GPU assignment', 'controller': 'JOINT_POSITION', 'physics_timestep_s': 0.002, 'independent_retest': False, 'offline_physics_audit': False, 'telemetry': 'every physics interval: all contacts, tool-table normal load, raw tool force and torque; 20Hz native cleaning progress', 'generator': 'current interactive assistant; host-feedback access, unblinded engineering', 'perception': 'public calibrated RGB-D color/geometry', 'ik': 'local CPU PyRoki', 'planner': 'official joint interpolation'}, input_module=None)
