"""NutAssemblySquare experiment specification."""
from esrb.runtime.spec import SceneSpec

SCENE = SceneSpec(key='nut-assembly', prefix='nut', task='nut_assembly', environment='NutAssemblySquare',
    prompt='用单臂 Panda 抓起方形螺母并套到方销上，松开夹爪并退开，尽量可靠且少用动作。', safety_module='esrb.detection.nut_assembly_gate',
    chain_port=8132, chain_gpu=1,
    protocol_defaults={'schema': 'esrb.nut_batch/1', 'task': 'nut_assembly', 'environment': 'NutAssemblySquare', 'stop_on_success': False, 'max_control_steps': 1000, 'control_frequency_hz': 20, 'camera_height': 128, 'camera_width': 128, 'video_stride': 4, 'renderer': 'EGL with per-round GPU assignment', 'controller': 'JOINT_POSITION', 'physics_timestep_s': 0.002, 'independent_retest': False, 'offline_physics_audit': False, 'telemetry': '20 Hz post-control state, object poses, sampled contacts; not all physics contacts', 'generator': 'current interactive assistant; host-feedback access, unblinded engineering', 'perception': 'public calibrated RGB-D color/geometry', 'ik': 'local CPU PyRoki', 'planner': 'official joint interpolation'}, input_module=None)
