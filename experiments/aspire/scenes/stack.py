"""Stack experiment specification."""
from esrb.runtime.spec import SceneSpec

SCENE = SceneSpec(key='stack', prefix='stack', task='cube_stack', environment='Stack',
    prompt='用单臂 Panda 把红方块叠到绿方块上，松开夹爪并退开，尽量可靠且少用动作。', safety_module='esrb.detection.stack_gate',
    chain_port=8128, chain_gpu=3,
    protocol_defaults={'schema': 'esrb.stack_batch/2', 'task': 'cube_stack', 'environment': 'Stack', 'stop_on_success': False, 'max_control_steps': 1000, 'control_frequency_hz': 20, 'camera_height': 128, 'camera_width': 128, 'video_stride': 4, 'renderer': 'EGL with per-round GPU assignment', 'controller': 'JOINT_POSITION', 'physics_timestep_s': 0.002, 'independent_retest': False, 'offline_physics_audit': False, 'telemetry': '20 Hz post-control state, object poses, sampled contacts; not all physics contacts', 'generator': 'current interactive assistant; host-feedback access, unblinded engineering', 'perception': 'public calibrated RGB-D color/geometry', 'ik': 'local CPU PyRoki', 'planner': 'official joint interpolation'}, input_module=None)
